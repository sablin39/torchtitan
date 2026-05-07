# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.experimental._attention import (
    _context_parallel_shard,
    _enable_context_parallel_dispatcher,
    _HeadTailLoadBalancer,
    _PTRRLoadBalancer,
)
from torch.distributed.tensor.experimental._context_parallel._attention import (
    flex_cp_allgather,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.tools.logging import logger


def apply_cp_to_forward(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Wrap inner attention ``forward`` with CP logic.

    Must be called **before** ``Module.parallelize()`` so the CP wrapper
    is captured inside parallelize's ``local_map`` wrapping.

    The attention type is inferred via isinstance on the first module.

    TODO: This is a temporary workaround that manually allgathers K/V
    (FlexAttention) or wraps inputs as CP-sharded DTensors (SDPA).
    Once all models adopt config-based sharding with full DTensor,
    CP redistribution should be expressed declaratively via
    ShardingConfig and this function should be removed.

    Args:
        attention_modules: Sequence of inner attention modules to apply CP to.
        cp_mesh: Device mesh for context parallel dimension.
    """
    first = attention_modules[0]
    if isinstance(first, FlexAttention):
        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                pg_name = dist._get_process_group_name(mesh.get_group())

                def cp_forward(q, k, v, **kwargs):
                    k = k.contiguous()
                    v = v.contiguous()
                    global_k, global_v = flex_cp_allgather(k, v, 2, pg_name)
                    return orig_fn(q, global_k, global_v, **kwargs)

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, ScaledDotProductAttention):
        _enable_context_parallel_dispatcher()

        for mod in attention_modules:
            original_forward = mod.forward

            def _make_cp_forward(orig_fn, mesh):
                placement = [Shard(2)]

                def cp_forward(q, k, v, **kwargs):
                    if not isinstance(q, DTensor):
                        q = DTensor.from_local(q, mesh, placement, run_check=False)
                    if not isinstance(k, DTensor):
                        k = DTensor.from_local(k, mesh, placement, run_check=False)
                    if not isinstance(v, DTensor):
                        v = DTensor.from_local(v, mesh, placement, run_check=False)
                    output = orig_fn(q, k, v, **kwargs)
                    return output.to_local() if isinstance(output, DTensor) else output

                return cp_forward

            mod.forward = _make_cp_forward(original_forward, cp_mesh)

    elif isinstance(first, VarlenAttention):
        raise NotImplementedError("Variable-length attention CP is not yet supported")
    else:
        raise NotImplementedError(
            f"Context Parallel forward wrapping is not supported for "
            f"{type(first).__name__}"
        )

    logger.info("Applied Context Parallel (forward wrapping) to the model")


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Shard inputs, labels, positions, and attention masks for Context Parallel.

    The caller must provide ``extra_kwargs["positions"]`` before calling this
    function.  Position resolution (per-document vs sequential) is handled
    upstream in ``post_dataloading_process``.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary containing 'positions' (required) and
            optionally 'attention_masks' to be sharded.
        cp_mesh: Device mesh for context parallel dimension
        device: Device for the tensors
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """
    attention_masks = extra_kwargs.get("attention_masks", None)
    positions = extra_kwargs["positions"]
    (inputs, labels, positions), attention_masks = cp_shard(
        cp_mesh,
        (inputs, labels, positions),
        attention_masks,
        load_balancer_type,
    )
    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


def _build_flattened_cu_seqlens(
    *,
    batch_size: int,
    seq_len: int,
    positions: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if positions is None:
        return torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.long,
            device=device,
        )

    if positions.shape != (batch_size, seq_len):
        raise ValueError(
            f"RWKV/FLA CP expected positions shape {(batch_size, seq_len)}, "
            f"got {tuple(positions.shape)}"
        )

    starts: list[int] = [0]
    positions_cpu = positions.detach().to("cpu")
    for batch_idx in range(batch_size):
        row = positions_cpu[batch_idx]
        row_offset = batch_idx * seq_len
        if batch_idx > 0:
            starts.append(row_offset)
        nonzero_positions = torch.nonzero(row, as_tuple=False).flatten()
        if nonzero_positions.numel() == 0:
            continue
        padding_start = int(nonzero_positions[-1].item()) + 1
        for idx in range(1, seq_len):
            if row[idx] > row[idx - 1]:
                continue
            starts.append(row_offset + idx)
            # Collators pad positions with zeros; represent the whole padding
            # tail as one ignored sequence instead of many one-token sequences.
            if idx >= padding_start:
                break
    starts.append(batch_size * seq_len)
    starts = sorted(set(int(x) for x in starts))
    return torch.tensor(starts, dtype=torch.long, device=device)


def prepare_fla_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare contiguous sequence shards for FLA linear-attention CP.

    FLA CP expects a single flattened token stream shaped ``[1, total_tokens]``.
    The global cumulative sequence lengths are built before partitioning and are
    kept replicated so the model can call ``fla.ops.cp.build_cp_context``.
    """
    positions = extra_kwargs.pop("positions", None)
    batch_size, seq_len = inputs.shape
    total_tokens = batch_size * seq_len
    cp_world_size = cp_mesh.size(0)
    if total_tokens % cp_world_size != 0:
        raise ValueError(
            f"FLA CP requires total flattened tokens ({total_tokens}) to be "
            f"divisible by CP degree ({cp_world_size})"
        )

    cu_seqlens_global = _build_flattened_cu_seqlens(
        batch_size=batch_size,
        seq_len=seq_len,
        positions=positions,
        device=device,
    )
    extra_kwargs["cu_seqlens_global"] = cu_seqlens_global
    extra_kwargs["cu_seqlens_global_cpu"] = cu_seqlens_global.detach().to("cpu")

    # Optional v1 helper for multimodal models: keep a replicated copy so local
    # CP shards can map image placeholder spans back to global vision item order.
    if extra_kwargs.get("fla_cp_keep_global_input_ids", False):
        extra_kwargs["fla_cp_global_input_ids"] = inputs.reshape(1, total_tokens)
    extra_kwargs.pop("fla_cp_keep_global_input_ids", None)

    rank = dist.get_rank(cp_mesh.get_group())
    part_len = total_tokens // cp_world_size
    extra_kwargs["fla_cp_global_start"] = torch.tensor(
        rank * part_len,
        dtype=torch.long,
        device=device,
    )

    flat_inputs = inputs.reshape(1, total_tokens)
    flat_labels = labels.reshape(1, total_tokens)
    (flat_inputs, flat_labels), _ = cp_shard(
        cp_mesh,
        (flat_inputs, flat_labels),
        attention_masks=None,
        load_balancer_type=None,
        input_seq_dim=1,
    )
    return flat_inputs, flat_labels, extra_kwargs


def prepare_fla_varlen_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Prepare unsharded FLA variable-length input.

    RWKV/FLA models can train without CP on normal ``[B, S]`` tensors, but
    packed samples need document boundaries passed to ``token_shift`` and the
    DPLR recurrent kernel.  FLA represents those boundaries with
    ``cu_seqlens`` and expects the token stream to be flattened to
    ``[1, B * S]`` for varlen mode.
    """
    positions = extra_kwargs.pop("positions", None)
    batch_size, seq_len = inputs.shape
    total_tokens = batch_size * seq_len

    cu_seqlens_global = _build_flattened_cu_seqlens(
        batch_size=batch_size,
        seq_len=seq_len,
        positions=positions,
        device=device,
    )
    extra_kwargs["cu_seqlens_global"] = cu_seqlens_global
    extra_kwargs["cu_seqlens_global_cpu"] = cu_seqlens_global.detach().to("cpu")

    return (
        inputs.reshape(1, total_tokens),
        labels.reshape(1, total_tokens),
        extra_kwargs,
    )


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | None]:
    """
    Shard inputs and attention masks across the context parallel mesh.

    This function distributes input tensors across devices in the CP mesh
    along the sequence dimension, enabling efficient processing. It optionally
    uses a load balancer to handle uneven computation workload.

    Args:
        cp_mesh: Device mesh for context parallel dimension
        inputs: Tuple of input tensors to be sharded along the sequence
            dimension
        attention_masks: Attention masks to be sharded. Supports None,
            BlockMask, or dict[str, BlockMask]
        load_balancer_type: Type of load balancer to use. Options:
            - "headtail": Use HeadTailLoadBalancer (for SDPA)
            - "ptrr": Use PTRRLoadBalancer (for FlexAttention)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.

    Returns:
        Tuple of (sharded_inputs, attention_masks) where:
            - sharded_inputs: Tuple of input tensors sharded along the
              sequence dimension
            - attention_masks: Sharded attention masks (BlockMask or
              dict[str, BlockMask]) or None

    Raises:
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None or a dict
    """
    seq_len = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)

    load_balancer = None
    if load_balancer_type:
        match load_balancer_type:
            case "headtail":
                # For SDPA, we use the _HeadTailLoadBalancer.
                load_balancer = _HeadTailLoadBalancer(
                    seq_len, cp_world_size, cp_mesh.device_type
                )
            case "ptrr":
                # For FlexAttention, we use _PTRRLoadBalancer.
                # _PTRRLoadBalancer requires attention_masks to be a BlockMask.
                # For dict[str, BlockMask], _PTRRLoadBalancer currently doesn't
                # support the case where there are multiple masks.
                if attention_masks is None or isinstance(attention_masks, dict):
                    raise ValueError(
                        "PTRRLoadBalancer requires attention_masks to be a "
                        "BlockMask, but got None or dict[str, BlockMask]"
                    )
                if not isinstance(attention_masks, BlockMask):
                    raise ValueError(
                        f"PTRRLoadBalancer requires attention_masks to be a "
                        f"BlockMask, but got {type(attention_masks)}"
                    )
                load_balancer = _PTRRLoadBalancer(attention_masks, cp_world_size)
            case _:
                raise ValueError(
                    f"Invalid load_balancer_type '{load_balancer_type}'. "
                    f"Must be one of: 'headtail', 'ptrr', or None"
                )

    inputs = cast(
        tuple[torch.Tensor, ...],
        _context_parallel_shard(
            mesh=cp_mesh,
            buffers=inputs,
            seq_dims=tuple(input_seq_dim for _ in inputs),
            load_balancer=load_balancer,
        ),
    )

    # BlockMask, has shape, [B, H, Q, KV], and we can only shard
    # on the Q seq dimension, not KV.
    MASK_Q_SEQ_DIM = 2
    if attention_masks is not None:
        assert isinstance(attention_masks, (BlockMask, dict))
        masks = (
            [attention_masks]
            if isinstance(attention_masks, BlockMask)
            else list(attention_masks.values())
        )
        masks = _context_parallel_shard(
            mesh=cp_mesh,
            buffers=masks,
            seq_dims=(MASK_Q_SEQ_DIM,) * len(masks),
            load_balancer=load_balancer,
        )
        attention_masks = cast(
            (BlockMask | dict[str, BlockMask]),
            (
                masks[0]
                if isinstance(attention_masks, BlockMask)
                else {k: v for k, v in zip(attention_masks.keys(), masks)}
            ),
        )

    return inputs, attention_masks
