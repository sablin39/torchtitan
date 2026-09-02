from __future__ import annotations

from typing import Any

from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    resolve_fsdp_mesh,
)

from .model import RAEDecoder


def parallelize_rae(
    model: RAEDecoder,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: Any,
    dump_folder: str,
) -> RAEDecoder:
    del ac_config, dump_folder
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError("RAE Stage 1 model compilation is not supported yet")
    if parallel_dims.spmd_backend != "spmd_types":
        raise ValueError("RAE Stage 1 parallelization requires the spmd_types backend")
    if parallelism.tensor_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support tensor parallelism")
    if parallelism.context_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support context parallelism")
    if parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError("RAE Stage 1 does not support pipeline parallelism")

    dmuon = None
    if model._dmuon_enabled:
        from torchtitan.components.optimizer.dmuon import load_dmuon

        dmuon = load_dmuon()
        predicate = (
            lambda name, parameter: parameter.ndim == 2 and parameter.requires_grad
        )
        if not parallel_dims.dp_enabled:
            dmuon.dedicate_params_ddp(
                model,
                mesh=parallel_dims.world_mesh,
                predicate=predicate,
            )
            dmuon.replicate(model, mesh=parallel_dims.world_mesh)
            return model

    if not parallel_dims.dp_enabled:
        return model

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        ),
    }
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    if model._dmuon_enabled:
        assert dmuon is not None
        if parallel_dims.dp_shard_enabled:
            shard_mesh = parallel_dims.get_mesh("fsdp")
            replicate_mesh = parallel_dims.get_optional_mesh("dp_replicate")
            dmuon.dedicate_params(
                model,
                mesh=shard_mesh,
                replicate_mesh=replicate_mesh,
                predicate=predicate,
                reshard_after_forward=False,
            )
        else:
            replicate_mesh = parallel_dims.get_mesh("dp_replicate")
            dmuon.dedicate_params_ddp(
                model,
                mesh=replicate_mesh,
                predicate=predicate,
            )

    for layer in model.layers:
        fully_shard(layer, **fsdp_config)
    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
    return model


__all__ = ["parallelize_rae"]
