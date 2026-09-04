# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch.distributed._composable import replicate

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from .decoder import RAEDecoder


def parallelize_rae(
    model: RAEDecoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> RAEDecoder:
    if parallel_dims.spmd_backend != "spmd_types":
        raise ValueError("RAE Stage 1 parallelization requires the spmd_types backend")
    if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support tensor parallelism")
    if parallelism.context_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support context parallelism")
    if parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support pipeline parallelism")
    if parallel_dims.dp_shard_enabled:
        raise NotImplementedError(
            "RAE Stage 1 currently supports replicated data parallelism only; "
            "set data_parallel_shard_degree=1"
        )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    predicate = lambda name, parameter: parameter.ndim == 2 and parameter.requires_grad
    dmuon = None
    if model._dmuon_enabled:
        from torchtitan.components.optimizer.dmuon import load_dmuon

        dmuon = load_dmuon()
        if not parallel_dims.dp_replicate_enabled:
            dmuon.dedicate_params_ddp(
                model,
                mesh=parallel_dims.world_mesh,
                predicate=predicate,
            )
            dmuon.replicate(model, mesh=parallel_dims.world_mesh)
            return model

    if not parallel_dims.dp_replicate_enabled:
        return model

    replicate_mesh = parallel_dims.get_mesh("dp_replicate")
    if model._dmuon_enabled:
        assert dmuon is not None
        dmuon.dedicate_params_ddp(
            model,
            mesh=replicate_mesh,
            predicate=predicate,
        )
        dmuon.replicate(model, mesh=replicate_mesh)
    else:
        replicate(model, device_mesh=replicate_mesh)
    return model


__all__ = ["parallelize_rae"]
