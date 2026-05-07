# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.models.rwkv7.model import RWKV7ForCausalLM
from torchtitan.tools.logging import logger


def parallelize_rwkv7(
    model: RWKV7ForCausalLM,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    if parallel_dims.tp_enabled:
        raise NotImplementedError("RWKV7 v1 does not support tensor parallelism")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("RWKV7 v1 does not support pipeline parallelism")

    if parallel_dims.cp_enabled:
        if parallelism.context_parallel_load_balancer is not None:
            raise ValueError(
                "RWKV7 CP requires context_parallel_load_balancer=None because "
                "the recurrence must preserve contiguous token order."
            )
        if compile_config.enable and "model" in compile_config.components:
            logger.warning(
                "RWKV7 CP with torch.compile is experimental. TorchTitan will "
                "compile RWKV blocks with fullgraph=False so FLA custom kernels "
                "and CP communication can graph-break if needed."
            )
        total_tokens = training.local_batch_size * training.seq_len
        if total_tokens % parallel_dims.cp != 0:
            raise ValueError(
                f"RWKV7 CP requires local_batch_size * seq_len ({total_tokens}) "
                f"to be divisible by CP degree ({parallel_dims.cp})"
            )
        model.set_cp_process_group(parallel_dims.get_mesh("cp").get_group())

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model.llm,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_rwkv_compile(model.llm, compile_config)

    if parallel_dims.fsdp_enabled or parallel_dims.dp_replicate_enabled:
        names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to RWKV7")
    elif parallel_dims.fsdp_enabled:
        logger.info("Applied FSDP to RWKV7")
    else:
        logger.info("Running RWKV7 without data-parallel wrapping")
    return model


def apply_rwkv_compile(model: nn.Module, compile_config: CompileConfig) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )
    for block in model.layers.values():
        block.compile(backend=compile_config.backend, fullgraph=False)
    logger.info("Compiled each RWKV7Block with torch.compile(fullgraph=False)")


def apply_fsdp(
    model: RWKV7ForCausalLM,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled=False,
    )

    fully_shard(
        model.llm.embeddings,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for block in model.llm.layers.values():
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model.llm.norm,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )
    fully_shard(
        model.lm_head,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )
    fully_shard(model.llm, **fsdp_config)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
