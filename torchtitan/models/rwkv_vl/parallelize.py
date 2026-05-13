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
from torchtitan.models.rwkv_vl.model import RWKV7VLForConditionalGeneration
from torchtitan.tools.logging import logger


def parallelize_rwkv_vl(
    model: RWKV7VLForConditionalGeneration,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    trainable_roots = getattr(model, "_trainable_roots", None)
    if trainable_roots is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "RWKV-VL module_lrs=%s enables roots=%s and %s / %s trainable parameters",
            model.config.root_lrs,
            list(trainable_roots),
            f"{trainable_params:,}",
            f"{total_params:,}",
        )

    if parallel_dims.tp_enabled:
        raise NotImplementedError("RWKV-VL v1 does not support tensor parallelism")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("RWKV-VL v1 does not support pipeline parallelism")

    if parallel_dims.cp_enabled:
        if parallelism.context_parallel_load_balancer is not None:
            raise ValueError("RWKV-VL CP requires context_parallel_load_balancer=None")
        if compile_config.enable and "model" in compile_config.components:
            logger.warning(
                "RWKV-VL CP with torch.compile is experimental. TorchTitan will "
                "compile RWKV blocks with fullgraph=False so FLA custom kernels "
                "and CP communication can graph-break if needed."
            )
        total_tokens = training.local_batch_size * training.seq_len
        if total_tokens % parallel_dims.cp != 0:
            raise ValueError(
                f"RWKV-VL CP requires local_batch_size * seq_len ({total_tokens}) "
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
            logger.info("Applied HSDP to RWKV-VL")
        else:
            logger.info("Applied FSDP to RWKV-VL")
    else:
        logger.info("Running RWKV-VL without data-parallel wrapping")
    return model


def apply_rwkv_compile(model: nn.Module, compile_config: CompileConfig) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = (
        True  # pyrefly: ignore [bad-assignment]
    )
    for block in model.layers.values():
        block.compile(backend=compile_config.backend, fullgraph=False)
    logger.info("Compiled each RWKV7Block with torch.compile(fullgraph=False)")


def _has_trainable_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def _fully_shard_if_trainable(
    module: nn.Module,
    *,
    module_name: str,
    skipped_frozen_modules: list[str],
    **kwargs,
) -> None:
    if _has_trainable_params(module):
        fully_shard(module, **kwargs)
    else:
        skipped_frozen_modules.append(module_name)


def apply_fsdp(
    model: RWKV7VLForConditionalGeneration,
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
    frozen_params = {p for p in model.parameters() if not p.requires_grad}
    if frozen_params:
        fsdp_config["ignored_params"] = frozen_params
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled=False,
    )

    skipped_frozen_modules: list[str] = []
    _fully_shard_if_trainable(
        model.vision_encoder,
        module_name="vision_encoder",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    _fully_shard_if_trainable(
        model.proj,
        module_name="proj",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    _fully_shard_if_trainable(
        model.llm.embeddings,
        module_name="llm.embeddings",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    _fully_shard_if_trainable(
        model.llm.pre_norm,
        module_name="llm.pre_norm",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for layer_idx, block in model.llm.layers.items():
        _fully_shard_if_trainable(
            block,
            module_name=f"llm.layers.{layer_idx}",
            skipped_frozen_modules=skipped_frozen_modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    _fully_shard_if_trainable(
        model.llm.norm,
        module_name="llm.norm",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )
    _fully_shard_if_trainable(
        model.lm_head,
        module_name="lm_head",
        skipped_frozen_modules=skipped_frozen_modules,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward_policy == "always",
    )
    if skipped_frozen_modules:
        logger.info(
            "Skipped FSDP wrapping for frozen RWKV-VL modules: %s",
            skipped_frozen_modules,
        )
    fully_shard(model.llm, **fsdp_config)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
