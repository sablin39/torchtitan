# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from torchtitan.components.optimizer import ParamGroupConfig
from torchtitan.models.common import Linear, RMSNorm
from torchtitan.models.common.vision_features import (
    _find_vision_spans,
    apply_vision_slices,
)
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.models.rwkv7.model import (
    _output_linear_init,
    _zero_,
    LayerNorm,
    RWKV7Backbone,
)
from torchtitan.models.rwkv7.tokenizer import (
    DEFAULT_IMAGE_TOKEN_ID,
    DEFAULT_VISION_END_TOKEN_ID,
    DEFAULT_VISION_START_TOKEN_ID,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList, Sequential


ReLU = Module.from_nn_module(nn.ReLU)
GELU = Module.from_nn_module(nn.GELU)


NormKind = Literal["layernorm", "rmsnorm"]
FFNKind = Literal["relu", "gelu", "swiglu"]
ProjectorKind = Literal["mlp", "cross_attn"]


# Projector cross-attn shapes are bucketed so ``_cross_attn_flex``
# (compiled with ``dynamic=False``) sees at most O(log N_max) distinct
# (Q_LEN, KV_LEN) pairs. With ``projector_extra_merge_size > 1`` the K/V
# stream can be ``extra_merge_size**2`` larger than Q for the same image,
# so we generate the ladder on the fly instead of maintaining a static one.
_BUCKET_MIN = 128
# Soft ceiling — values past this almost certainly indicate a config bug
# (e.g. unbounded ``max_images_per_batch`` with a huge ``max_pixels``) and
# a single FlexAttention kernel that large would also be a memory hazard.
_BUCKET_MAX = 1 << 24  # 16,777,216

# Target K/V tokens per chunked cross-attn call. Images are packed greedily
# into chunks of at most this many K/V tokens so each FlexAttention call
# allocates Q/K/V padded to ``next_pow2(<= this)`` regardless of how many
# images the batch holds. Matches ``vit_patch_bucket_size`` so the encoder
# and projector see similar per-call shapes on Hopper.
_CROSS_ATTN_CHUNK_KV_TARGET = 65536


def _next_pow2_bucket(n: int) -> int:
    """Round ``n`` up to the next power of two, clamped to ``[_BUCKET_MIN, _BUCKET_MAX]``.

    Raises ``ValueError`` past ``_BUCKET_MAX`` so a runaway config surfaces
    before flex_attention attempts a multi-million-row compile.
    """
    if n <= _BUCKET_MIN:
        return _BUCKET_MIN
    if n > _BUCKET_MAX:
        raise ValueError(
            f"Value {n} exceeds the projector FlexAttention bucket ceiling "
            f"{_BUCKET_MAX}; check max_pixels / max_images_per_batch / "
            "projector_extra_merge_size or pin q_buckets/kv_buckets explicitly."
        )
    return 1 << (n - 1).bit_length()


def _ceil_to_bucket(n: int, ladder: tuple[int, ...]) -> int:
    """Return the smallest ladder entry >= ``n``.

    Only used when ``q_buckets`` / ``kv_buckets`` is explicitly pinned on the
    projector config. Auto-bucketing uses :func:`_next_pow2_bucket`.
    """
    for b in ladder:
        if b >= n:
            return b
    raise ValueError(
        f"Value {n} exceeds the largest configured FlexAttention bucket "
        f"{ladder[-1]}; widen projector q_buckets/kv_buckets or leave them "
        "unset to use auto power-of-two bucketing."
    )


def _pack_images_into_chunks(
    num_kv_per_item: list[int], target_kv: int
) -> list[tuple[int, int]]:
    """Greedily pack images into contiguous ``[img_lo, img_hi)`` ranges whose
    K/V token sum stays within ``target_kv`` whenever possible.

    A single image larger than ``target_kv`` forms its own chunk (it can't be
    split since per-image attention must stay intact).
    """
    if not num_kv_per_item:
        return []
    chunks: list[tuple[int, int]] = []
    cur_lo = 0
    cur_count = 0
    for i, n in enumerate(num_kv_per_item):
        if cur_count > 0 and cur_count + n > target_kv:
            chunks.append((cur_lo, i))
            cur_lo = i
            cur_count = 0
        cur_count += n
    chunks.append((cur_lo, len(num_kv_per_item)))
    return chunks


_compiled_create_block_mask = create_block_mask

# Dedicated ``torch.compile`` of ``flex_attention`` for the cross-attn
# projector. Kept separate from ``FlexAttention._compiled_flex_attn`` (used
# by the vision encoder and the LLM) so the projector's distinct (Q_LEN,
# KV_LEN) shapes don't poison the shared compile cache. ``dynamic=False``
# requires the caller to pad Q/K/V to fixed bucket sizes
# (``q_buckets`` / ``kv_buckets`` on the projector config) so each invocation
# specialises against a single shape. Inductor options mirror the vision
# encoder's FlexAttention settings (max-autotune, coordinate descent
# tuning, TMA descriptors) — safe here because the static buckets give a
# single shape to specialise against.
_cross_attn_flex = torch.compile(
    flex_attention,
    dynamic=False,
    options={
        "max_autotune": True,
        "coordinate_descent_tuning": True,
        "triton.cudagraphs": False,
        "assume_aligned_inputs": True,
    },
)


# Default per-call kernel options forwarded to ``flex_attention`` for the
# cross-attn projector. The vision encoder pins these for ViT bucketing on
# H800; we reuse the same starting point here. Users can override via the
# projector config's ``kernel_options`` field (currently unset by default).
_CROSS_ATTN_KERNEL_OPTIONS_DEFAULT: dict[str, object] = {
    "USE_TMA": True,
    "ROWS_GUARANTEED_SAFE": False,
    "IS_DIVISIBLE": True,
    # Block sizes / stages / warps are left for Inductor's autotune to pick;
    # the supported cross_attn production flavors use head_dim>=128 where
    # autotune finds valid choices. Smaller smoke configs (head_dim=64) hit
    # a Triton "Cannot broadcast" issue with TMA; those callers should
    # override ``kernel_options={"USE_TMA": False, "fwd_BLOCK_M": 64, ...}``
    # via the projector config.
}


def _build_norm(kind: NormKind, dim: int, eps: float) -> Module:
    if kind == "layernorm":
        return LayerNorm(dim, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm.Config(normalized_shape=dim, eps=eps).build()
    raise ValueError(f"Unknown norm kind: {kind!r}; expected layernorm|rmsnorm")


_ROOT_MODULE_NAMES = ("vision_encoder", "proj", "llm")
_ROOT_PARAM_PATTERNS = {
    "vision_encoder": r"^vision_encoder\.",
    "proj": r"^proj\.",
    # lm_head shares the llm LR root; it has no separate LR entry.
    "llm": r"^(llm|lm_head)\.",
}


def _default_root_lrs() -> dict[str, float]:
    return {name: 1.0 for name in _ROOT_MODULE_NAMES}


def _validate_root_lrs(root_lrs: dict[str, float]) -> dict[str, float]:
    missing = set(_ROOT_MODULE_NAMES) - set(root_lrs)
    unknown = set(root_lrs) - set(_ROOT_MODULE_NAMES)
    if missing or unknown:
        raise ValueError(
            "RWKV-VL root_lrs must contain exactly "
            f"{list(_ROOT_MODULE_NAMES)}; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    resolved = {name: float(root_lrs[name]) for name in _ROOT_MODULE_NAMES}
    negative = {name: lr for name, lr in resolved.items() if lr < 0}
    if negative:
        raise ValueError(f"RWKV-VL module LRs must be non-negative, got {negative}")
    if not any(lr > 0 for lr in resolved.values()):
        raise ValueError("At least one RWKV-VL module LR must be greater than 0")
    return resolved


def _resolve_root_lrs(module_lrs: Any, default_lr: float) -> dict[str, float]:
    if default_lr <= 0:
        raise ValueError(
            "RWKV-VL module LR config requires --optimizer.lr to be greater than 0"
        )

    resolved = {}
    for name in _ROOT_MODULE_NAMES:
        value = getattr(module_lrs, name)
        resolved[name] = default_lr if value is None else float(value)

    return _validate_root_lrs(resolved)


def _configure_optimizer_param_groups(
    optimizer_config: Any,
    root_lrs: dict[str, float],
):
    base_lr = float(optimizer_config.lr)
    if base_lr <= 0:
        raise ValueError(
            "RWKV-VL module LR config requires --optimizer.lr to be greater than 0"
        )

    optimizer_config.param_groups = [
        ParamGroupConfig(
            pattern=_ROOT_PARAM_PATTERNS[name],
            lr_multiplier=lr / base_lr,
        )
        for name, lr in root_lrs.items()
        if lr > 0
    ]


def _validate_backbone_chunk_size(chunk_size: int) -> int:
    chunk_size = int(chunk_size)
    if chunk_size < 16:
        raise ValueError(
            f"RWKV-VL backbone_chunk_size must be at least 16; got {chunk_size}"
        )
    if chunk_size & (chunk_size - 1):
        raise ValueError(
            f"RWKV-VL backbone_chunk_size must be a power of two; got {chunk_size}"
        )
    return chunk_size


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear:
    init = {"weight": _zero_, "bias": _zero_} if bias else {"weight": _zero_}
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=init,
    ).build()


def _projector_linear_config(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear.Config:
    init = {
        "weight": partial(nn.init.trunc_normal_, std=0.02),
        **({"bias": _zero_} if bias else {}),
    }
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        param_init=init,
    )


def _projector_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
) -> Linear:
    return _projector_linear_config(in_features, out_features, bias=bias).build()


class _SwiGLUFFN(Module):
    """SwiGLU feed-forward used inside the visual projector.

    Uses param names ``w1``/``w2``/``w3`` to match ``common.FeedForward``.
    """

    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int, bias: bool):
        super().__init__()
        self.w1 = _projector_linear(in_dim, hidden_dim, bias=bias)
        self.w2 = _projector_linear(hidden_dim, out_dim, bias=bias)
        self.w3 = _projector_linear(in_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _build_ffn(
    kind: FFNKind,
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    bias: bool = True,
) -> Module:
    """Build the projector's inner feed-forward block.

    For ``relu``/``gelu`` the structure is ``Linear -> activation -> Linear``,
    keeping state_dict keys ``mlp.0.*`` / ``mlp.2.*`` (back-compat with the
    pre-refactor projector when ``kind='relu'``).
    For ``swiglu`` the module has ``w1``/``w2``/``w3`` Linear params.
    """
    if kind == "relu":
        return Sequential(
            _projector_linear(in_dim, hidden_dim, bias=bias),
            ReLU(),
            _projector_linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "gelu":
        return Sequential(
            _projector_linear(in_dim, hidden_dim, bias=bias),
            GELU(),
            _projector_linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "swiglu":
        return _SwiGLUFFN(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, bias=bias
        )
    raise ValueError(f"Unknown ffn kind: {kind!r}; expected relu|gelu|swiglu")


class _VisualStreamProjector(Module):
    """MLP-based visual stream projector.

    When ``merge_size > 1`` the projector also performs the spatial merge:
    inputs are pre-shuffle-normalized, ``merge_size**2`` adjacent tokens are
    concatenated along the channel axis, then the MLP maps the merged
    ``encoder_dim * merge_size**2`` channels to ``project_dim``. This is the
    same structure as ``Qwen3VLVisionModel.PatchMerger`` and removes the
    need for a separate ``extra_merger`` module on the projector main path.
    """

    def __init__(
        self,
        *,
        encoder_dim: int,
        hidden_dim: int,
        project_dim: int,
        norm_eps: float,
        norm: NormKind = "layernorm",
        ffn: FFNKind = "relu",
        merge_size: int = 1,
    ):
        super().__init__()
        if merge_size < 1:
            raise ValueError(f"merge_size must be >= 1; got {merge_size}")
        self.merge_size = merge_size
        self.merge_unit = merge_size**2
        self.encoder_dim = encoder_dim
        in_dim = encoder_dim * self.merge_unit
        # Pre-shuffle norm when merging (matches ``PatchMerger``'s pattern).
        # When merge_size == 1 the in_norm is omitted entirely so the
        # state_dict matches the original ``relu``/``layernorm`` projector
        # bit-identically (back-compat).
        self.in_norm = (
            _build_norm(norm, encoder_dim, norm_eps) if merge_size > 1 else None
        )
        self.pre_norm = _build_norm(norm, project_dim, norm_eps)
        self.mlp = _build_ffn(
            ffn,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=project_dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_size > 1:
            # x: ``(N, encoder_dim)``. Normalize per token, then reshape
            # ``merge_unit`` adjacent tokens into a single merged token along
            # the channel axis. ``N`` is required to be divisible by
            # ``merge_unit`` — enforced upstream by the dataloader's
            # ``vit_patch_bucket_unit = lcm(bucket, 128, spatial_merge**2)``.
            n = x.shape[0]
            x = self.in_norm(x)
            x = x.reshape(n // self.merge_unit, self.encoder_dim * self.merge_unit)
        x = self.mlp(x)
        return x + self.pre_norm(x)


def _flatten_visual_features(
    features: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Remove per-image padding while preserving Qwen's native patch order."""
    count_list = [int(count) for count in counts.tolist()]
    if features.dim() == 2:
        return features[: sum(count_list)]
    if features.dim() != 3 or features.shape[0] != len(count_list):
        raise ValueError(
            "Visual features must be flat or padded per image; got "
            f"shape={tuple(features.shape)} for {len(count_list)} images"
        )
    return torch.cat(
        [features[index, :count] for index, count in enumerate(count_list)], dim=0
    )


def _tokenpacker_query_seeds(
    merged_features: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    extra_merge_size: int,
) -> torch.Tensor:
    """Bilinearly resample native merged ViT features onto the query grid."""
    merged_counts = grid_thw.prod(-1) // (spatial_merge_size**2)
    merged_features = _flatten_visual_features(merged_features, merged_counts)
    chunks = merged_features.split([int(n) for n in merged_counts.tolist()])
    query_chunks = []
    for chunk, grid in zip(chunks, grid_thw.tolist(), strict=True):
        temporal, height, width = (int(value) for value in grid)
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        query_height = merged_height // extra_merge_size
        query_width = merged_width // extra_merge_size
        feature_grid = chunk.view(
            temporal, merged_height, merged_width, chunk.shape[-1]
        ).permute(0, 3, 1, 2)
        query_grid = F.interpolate(
            feature_grid.float(),
            size=(query_height, query_width),
            mode="bilinear",
            align_corners=False,
        ).to(chunk.dtype)
        query_chunks.append(query_grid.permute(0, 2, 3, 1).reshape(-1, chunk.shape[-1]))
    return torch.cat(query_chunks, dim=0)


def _query_position_encoding(
    grid_thw: torch.Tensor,
    *,
    dim: int,
    spatial_merge_size: int,
    extra_merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D query position encoding requires dim % 4 == 0, got {dim}")
    quarter_dim = dim // 4
    frequency = torch.exp(
        -torch.arange(quarter_dim, device=device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=device)) / max(quarter_dim - 1, 1))
    )
    chunks = []
    merge = spatial_merge_size * extra_merge_size
    for temporal, height, width in grid_thw.tolist():
        query_height = int(height) // merge
        query_width = int(width) // merge
        y = torch.linspace(0.0, 1.0, query_height, device=device)
        x = torch.linspace(0.0, 1.0, query_width, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        y_phase = y_grid.reshape(-1, 1) * (2 * torch.pi) * frequency.reshape(1, -1)
        x_phase = x_grid.reshape(-1, 1) * (2 * torch.pi) * frequency.reshape(1, -1)
        spatial = torch.cat(
            [y_phase.sin(), y_phase.cos(), x_phase.sin(), x_phase.cos()], dim=-1
        )
        chunks.append(spatial.repeat(int(temporal), 1))
    return torch.cat(chunks, dim=0).to(dtype=dtype)


def _tokenpacker_local_ids(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    extra_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return matching query/KV region ids in native Qwen patch order."""
    device = grid_thw.device
    query_ids = []
    memory_ids = []
    region_offset = 0
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        query_height = height // (spatial_merge_size * extra_merge_size)
        query_width = width // (spatial_merge_size * extra_merge_size)
        ids = torch.arange(
            temporal * query_height * query_width, device=device, dtype=torch.long
        ).view(temporal, query_height, query_width)
        query_ids.append(ids.reshape(-1) + region_offset)
        full_grid = ids.repeat_interleave(
            spatial_merge_size * extra_merge_size, dim=1
        ).repeat_interleave(spatial_merge_size * extra_merge_size, dim=2)
        native_order = (
            full_grid.view(
                temporal,
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1)
        )
        memory_ids.append(native_order + region_offset)
        region_offset += temporal * query_height * query_width
    return torch.cat(query_ids), torch.cat(memory_ids)


class VisualAdapter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        encoder_dim: int = 1024
        vision_dim: int | None = None
        hidden_dim: int | None = None
        project_dim: int = 1024
        num_deepstack: int = 0
        norm_eps: float = 1e-5
        # Selectors shared by both projector kinds. Defaults preserve the
        # original `relu`/`layernorm` projector for back-compat.
        kind: ProjectorKind = "mlp"
        norm: NormKind = "layernorm"
        ffn: FFNKind = "relu"
        # TokenPacker query-grid downsampling ratio when the processor uses a
        # coarser image-token grid than the vision encoder
        # (image_pad count < K/V token count). The ratio is
        # ``processor_spatial_merge_size / vision_spatial_merge_size`` and
        # must be a positive integer. ``1`` keeps the native merged-token
        # resolution. Only meaningful when
        # ``kind == "cross_attn"`` — with ``kind == "mlp"`` the deepstack
        # streams also need to match image_pad length, so we currently
        # require the two merge sizes to be equal in that case.
        extra_merge_size: int = 1
        spatial_merge_size: int = 2
        language_layer_indices: tuple[int, ...] = ()
        num_query_heads: int | None = None
        num_key_value_heads: int | None = None
        tie_qkvo: bool = True
        # Powers-of-two ladders for FlexAttention Q_LEN / KV_LEN buckets. Each
        # forward picks the smallest bucket that fits the current shape and
        # pads the rest with masked rows. ``None`` -> no bucketing (rebuild
        # block_mask per shape).
        q_buckets: tuple[int, ...] | None = None
        kv_buckets: tuple[int, ...] | None = None
        # Optional overrides for the per-call FlexAttention ``kernel_options``.
        # Merged on top of ``_CROSS_ATTN_KERNEL_OPTIONS_DEFAULT``. Pass
        # ``{"USE_TMA": False}`` to disable TMA on small head_dim shapes that
        # trip Triton autotune; pass empty dict to keep defaults.
        kernel_options: dict[str, Any] | None = None

    def __init__(self, config: Config):
        super().__init__()
        if config.extra_merge_size < 1:
            raise ValueError(
                "VisualAdapter.Config.extra_merge_size must be >= 1; "
                f"got {config.extra_merge_size}"
            )
        if config.extra_merge_size > 1 and config.kind != "cross_attn":
            raise ValueError(
                "extra_merge_size > 1 (processor merge != vision merge) is "
                "only supported with kind='cross_attn'; with kind='mlp' the "
                "deepstack streams must also match image_pad length and that "
                "compression path isn't implemented yet."
            )
        self.config = config
        self.encoder_dim = config.encoder_dim
        self.project_dim = config.project_dim
        self.hidden_dim = config.hidden_dim or config.project_dim * 4
        self.num_deepstack = config.num_deepstack
        self.kind = config.kind
        self.extra_merge_size = config.extra_merge_size
        self.language_layer_indices = tuple(config.language_layer_indices)
        self.tie_qkvo = config.tie_qkvo
        if config.kind == "mlp":
            self.main = _VisualStreamProjector(
                encoder_dim=config.encoder_dim,
                hidden_dim=self.hidden_dim,
                project_dim=config.project_dim,
                norm_eps=config.norm_eps,
                norm=config.norm,
                ffn=config.ffn,
                merge_size=config.extra_merge_size,
            )
            self.deepstack = ModuleList(
                [
                    _VisualStreamProjector(
                        encoder_dim=config.encoder_dim,
                        hidden_dim=self.hidden_dim,
                        project_dim=config.project_dim,
                        norm_eps=config.norm_eps,
                        norm=config.norm,
                        ffn=config.ffn,
                    )
                    for _ in range(config.num_deepstack)
                ]
            )
            return
        if config.kind != "cross_attn":
            raise ValueError(f"Unknown projector kind: {config.kind!r}")

        if config.norm != "layernorm":
            raise ValueError("cross_attn uses separate LayerNorms at every depth")
        self.vision_dim = config.vision_dim or config.encoder_dim
        self.num_query_heads = config.num_query_heads or 1
        self.num_key_value_heads = config.num_key_value_heads or self.num_query_heads
        if self.vision_dim % self.num_query_heads != 0:
            raise ValueError(
                f"vision_dim={self.vision_dim} must be divisible by "
                f"num_query_heads={self.num_query_heads}"
            )
        if self.num_query_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_key_value_heads; got "
                f"{self.num_query_heads} and {self.num_key_value_heads}"
            )
        if len(self.language_layer_indices) != config.num_deepstack:
            raise ValueError(
                "language_layer_indices must have one entry per visual depth; got "
                f"{self.language_layer_indices} for {config.num_deepstack} depths"
            )
        if (
            tuple(sorted(set(self.language_layer_indices)))
            != self.language_layer_indices
        ):
            raise ValueError("language_layer_indices must be unique and increasing")

        self.head_dim = self.vision_dim // self.num_query_heads
        kv_dim = self.num_key_value_heads * self.head_dim
        self.seed_query_norm = LayerNorm(config.encoder_dim, eps=config.norm_eps)
        self.seed_output_norm = LayerNorm(self.vision_dim, eps=config.norm_eps)
        self.query_norms = ModuleList(
            [
                LayerNorm(config.project_dim, eps=config.norm_eps)
                for _ in range(config.num_deepstack)
            ]
        )
        self.query_gate_projs = ModuleList(
            [
                _projector_linear(config.project_dim, self.num_query_heads, bias=False)
                for _ in range(config.num_deepstack)
            ]
        )
        self.memory_norms = ModuleList(
            [
                LayerNorm(self.vision_dim, eps=config.norm_eps)
                for _ in range(config.num_deepstack + 1)
            ]
        )
        self.seed_q_proj = _projector_linear(
            config.encoder_dim, self.vision_dim, bias=False
        )
        if self.tie_qkvo:
            # One projection set is reused at every visual/RWKV depth and by
            # the TokenPacker seed retrieval. GQA independently shares each
            # KV-head slice among a group of query heads.
            self.rwkv_q_proj = _projector_linear(
                config.project_dim, self.vision_dim, bias=False
            )
            self.k_proj = _projector_linear(self.vision_dim, kv_dim, bias=False)
            self.v_proj = _projector_linear(self.vision_dim, kv_dim, bias=False)
            self.o_proj = _projector_linear(
                self.vision_dim, config.project_dim, bias=False
            )
        else:
            self.rwkv_q_projs = ModuleList(
                [
                    _projector_linear(config.project_dim, self.vision_dim, bias=False)
                    for _ in range(config.num_deepstack)
                ]
            )
            self.k_projs = ModuleList(
                [
                    _projector_linear(self.vision_dim, kv_dim, bias=False)
                    for _ in range(config.num_deepstack + 1)
                ]
            )
            self.v_projs = ModuleList(
                [
                    _projector_linear(self.vision_dim, kv_dim, bias=False)
                    for _ in range(config.num_deepstack + 1)
                ]
            )
            self.o_projs = ModuleList(
                [
                    _projector_linear(self.vision_dim, config.project_dim, bias=False)
                    for _ in range(config.num_deepstack + 1)
                ]
            )
        self.kernel_options = dict(_CROSS_ATTN_KERNEL_OPTIONS_DEFAULT)
        if config.kernel_options:
            self.kernel_options.update(config.kernel_options)

    def _project_memory(
        self, features: torch.Tensor, *, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.memory_norms[depth](features)
        k_proj = self.k_proj if self.tie_qkvo else self.k_projs[depth]
        v_proj = self.v_proj if self.tie_qkvo else self.v_projs[depth]
        keys = k_proj(features).reshape(-1, self.num_key_value_heads, self.head_dim)
        values = v_proj(features).reshape(-1, self.num_key_value_heads, self.head_dim)
        return keys, values

    def _project_query_and_gate(
        self, depth: int, query_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_hidden_states = self.query_norms[depth](query_hidden_states)
        q_proj = self.rwkv_q_proj if self.tie_qkvo else self.rwkv_q_projs[depth]
        queries = q_proj(query_hidden_states)
        gates = torch.sigmoid(self.query_gate_projs[depth](query_hidden_states))
        return queries, gates

    def _project_output(self, depth: int, attended: torch.Tensor) -> torch.Tensor:
        o_proj = self.o_proj if self.tie_qkvo else self.o_projs[depth]
        return o_proj(attended)

    def _attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        block_mask: BlockMask,
        q_bucket: int,
        kv_bucket: int,
    ) -> torch.Tensor:
        q_len, kv_len = queries.shape[0], keys.shape[0]
        queries = F.pad(queries, (0, 0, 0, 0, 0, q_bucket - q_len))
        keys = F.pad(keys, (0, 0, 0, 0, 0, kv_bucket - kv_len))
        values = F.pad(values, (0, 0, 0, 0, 0, kv_bucket - kv_len))
        query_states = queries.transpose(0, 1).unsqueeze(0)
        key_states = keys.transpose(0, 1).unsqueeze(0)
        value_states = values.transpose(0, 1).unsqueeze(0)
        attended = _cross_attn_flex(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=self.num_query_heads != self.num_key_value_heads,
            kernel_options=self.kernel_options,
        )
        return attended[0, :, :q_len].transpose(0, 1).reshape(q_len, self.vision_dim)

    def attend(
        self,
        depth: int,
        query_hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        block_mask: BlockMask,
        q_bucket: int,
        kv_bucket: int,
    ) -> torch.Tensor:
        queries, gates = self._project_query_and_gate(depth, query_hidden_states)
        queries = queries.reshape(-1, self.num_query_heads, self.head_dim)
        attended = self._attention(
            queries,
            keys,
            values,
            block_mask=block_mask,
            q_bucket=q_bucket,
            kv_bucket=kv_bucket,
        )
        attended = attended.reshape(-1, self.num_query_heads, self.head_dim)
        attended = (attended * gates.unsqueeze(-1)).reshape(-1, self.vision_dim)
        return self._project_output(depth, attended)

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None = None,
        *,
        grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        """Project the main vision stream and prepare each deepstack stream.

        For ``kind='mlp'`` each deepstack entry is the final projected feature
        tensor (added residually into the LLM hidden state).
        For ``kind='cross_attn'`` each entry is a ``(k, v)`` tuple of shape
        ``(N_kv, num_heads, head_dim)`` — the actual attention runs inside
        the LLM ``after_layer`` callback against the per-layer hidden state.
        """
        if deepstack_features is None:
            deepstack_features = []
        expected_features = self.num_deepstack + (self.kind == "cross_attn")
        if len(deepstack_features) != expected_features:
            raise ValueError(
                f"Expected {expected_features} visual feature tensors, "
                f"got {len(deepstack_features)}."
            )
        if self.kind == "mlp":
            projected_deepstack = [
                projector(feature)
                for projector, feature in zip(self.deepstack, deepstack_features)
            ]
            return self.main(x), projected_deepstack
        if grid_thw is None:
            raise ValueError("grid_thw is required by the cross_attn projector")

        raw_counts = grid_thw.prod(-1)
        flat_memories = [
            _flatten_visual_features(feature, raw_counts)
            for feature in deepstack_features
        ]
        seed_memory = flat_memories[-1]
        seed_features = _tokenpacker_query_seeds(
            x,
            grid_thw,
            spatial_merge_size=self.config.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
        )
        seed_queries = self.seed_q_proj(self.seed_query_norm(seed_features))
        seed_queries = seed_queries + _query_position_encoding(
            grid_thw,
            dim=self.vision_dim,
            spatial_merge_size=self.config.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
            dtype=seed_queries.dtype,
            device=seed_queries.device,
        )
        seed_keys, seed_values = self._project_memory(
            seed_memory, depth=self.num_deepstack
        )
        _, memory_ids = _tokenpacker_local_ids(
            grid_thw,
            spatial_merge_size=self.config.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
        )
        memory_order = torch.argsort(memory_ids, stable=True)
        local_length = (self.config.spatial_merge_size * self.extra_merge_size) ** 2
        seed_keys = seed_keys[memory_order].view(
            seed_queries.shape[0],
            local_length,
            self.num_key_value_heads,
            self.head_dim,
        )
        seed_values = seed_values[memory_order].view_as(seed_keys)
        local_attention = F.scaled_dot_product_attention(
            seed_queries.view(-1, self.num_query_heads, 1, self.head_dim),
            seed_keys.permute(0, 2, 1, 3),
            seed_values.permute(0, 2, 1, 3),
            enable_gqa=self.num_query_heads != self.num_key_value_heads,
        ).reshape(-1, self.vision_dim)
        refined_queries = seed_queries + local_attention
        projected_main = self._project_output(
            self.num_deepstack, self.seed_output_norm(refined_queries)
        )
        projected_deepstack = [
            self._project_memory(feature, depth=depth)
            for depth, feature in enumerate(flat_memories[:-1])
        ]
        return projected_main, projected_deepstack


class RWKV7VLForConditionalGeneration(BaseModel):
    _skip_lm_head: bool = False

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vocab_size: int = 65536
        hidden_size: int = 1024
        llm: RWKV7Backbone.Config
        vision_encoder: Qwen3VLVisionEncoder.Config
        proj: VisualAdapter.Config
        lm_head: Linear.Config | None = None
        image_token_id: int = DEFAULT_IMAGE_TOKEN_ID
        vision_start_token_id: int = DEFAULT_VISION_START_TOKEN_ID
        vision_end_token_id: int = DEFAULT_VISION_END_TOKEN_ID
        uses_fla_context_parallel: bool = True
        # Spatial merge size used by the processor / dataloader when counting
        # ``<image_pad>`` tokens. Must be a positive integer multiple of
        # ``vision_encoder.spatial_merge_size``. When equal to vision merge
        # (the default), one ``<image_pad>`` token corresponds to exactly one
        # vision feature; when larger, each ``<image_pad>`` token represents
        # ``(processor_merge/vision_merge)**2`` vision features and the
        # projector constructs a coarser query grid and locally retrieves from
        # the corresponding native ViT patch regions.
        # Decoupled merge sizes are currently only supported with
        # ``proj.kind == "cross_attn"``.
        processor_spatial_merge_size: int = 2
        root_lrs: dict[str, float] = field(default_factory=_default_root_lrs)

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            parallelism = trainer_config.parallelism
            training = trainer_config.training
            compile_config = getattr(trainer_config, "compile", None)
            module_lrs = trainer_config.module_lrs
            self.root_lrs = _resolve_root_lrs(module_lrs, trainer_config.optimizer.lr)
            _configure_optimizer_param_groups(
                trainer_config.optimizer,
                self.root_lrs,
            )
            self.llm = replace(
                self.llm,
                chunk_size=_validate_backbone_chunk_size(
                    getattr(trainer_config, "backbone_chunk_size", self.llm.chunk_size)
                ),
            )

            # Snap vocab_size to the paired tokenizer (rounded up for matmul
            # alignment). image / vision_start / vision_end token ids are
            # intentionally not auto-synced from the tokenizer here — the VL
            # special-token IDs must be set explicitly per-flavor.
            from torchtitan.models.common.config_utils import (
                align_vocab_size_to_tokenizer,
            )

            tokenizer = kwargs.get("tokenizer")
            new_vocab = align_vocab_size_to_tokenizer(
                declared_vocab_size=self.vocab_size, tokenizer=tokenizer
            )
            if new_vocab != self.vocab_size:
                self.vocab_size = new_vocab
                self.llm = replace(
                    self.llm,
                    vocab_size=new_vocab,
                    embeddings=replace(self.llm.embeddings, num_embeddings=new_vocab),
                )
                if self.lm_head is not None:
                    self.lm_head = replace(self.lm_head, out_features=new_vocab)

            # Optional projector overrides let trainer config replace the
            # flavor-baked projector kind, normalization, FFN, and GQA layout.
            # ``projector_extra_merge_size`` controls the image-token grid only;
            # the collator preserves the frozen ViT's native patch layout.
            proj_overrides: dict[str, Any] = {}
            for src_name, dst_name in (
                ("projector_kind", "kind"),
                ("projector_norm", "norm"),
                ("projector_ffn", "ffn"),
                ("projector_num_query_heads", "num_query_heads"),
                ("projector_num_key_value_heads", "num_key_value_heads"),
                ("tie_projector_qkvo", "tie_qkvo"),
                ("projector_extra_merge_size", "extra_merge_size"),
            ):
                value = getattr(trainer_config, src_name, None)
                if value is not None:
                    proj_overrides[dst_name] = value
            q_bucket = getattr(trainer_config, "projector_q_bucket", None)
            if q_bucket is not None:
                proj_overrides["q_buckets"] = (int(q_bucket),)
            kv_bucket = getattr(trainer_config, "projector_kv_bucket", None)
            if kv_bucket is not None:
                proj_overrides["kv_buckets"] = (int(kv_bucket),)
            if proj_overrides:
                self.proj = replace(self.proj, **proj_overrides)

            visual_layers = getattr(
                trainer_config, "projector_visual_layer_indices", None
            )
            if visual_layers is not None:
                visual_layers = tuple(int(index) for index in visual_layers)
                self.vision_encoder = replace(
                    self.vision_encoder,
                    deepstack_visual_indices=list(visual_layers),
                )
                self.proj = replace(self.proj, num_deepstack=len(visual_layers))
            language_layers = getattr(
                trainer_config, "projector_language_layer_indices", None
            )
            if language_layers is not None:
                self.proj = replace(
                    self.proj,
                    language_layer_indices=tuple(
                        int(index) for index in language_layers
                    ),
                )
            if self.proj.kind == "cross_attn":
                self.vision_encoder = replace(
                    self.vision_encoder, raw_deepstack_features=True
                )

            # Derive the image-token merge size without changing the native
            # ViT patch layout used by the collator. This controls how many
            # ``<image_pad>`` tokens the processor inserts; patch padding and
            # ordering continue to use the vision encoder's native merge size.
            vision_merge = self.vision_encoder.spatial_merge_size
            extra = self.proj.extra_merge_size
            derived_processor_merge = vision_merge * extra
            self.processor_spatial_merge_size = derived_processor_merge
            dataloader_cfg = getattr(trainer_config, "dataloader", None)
            if dataloader_cfg is not None and hasattr(
                dataloader_cfg, "image_token_merge_size"
            ):
                dataloader_cfg.image_token_merge_size = derived_processor_merge

            if parallelism.tensor_parallel_degree > 1:
                raise NotImplementedError(
                    "RWKV-VL v1 does not support tensor parallelism"
                )
            if parallelism.pipeline_parallel_degree > 1:
                raise NotImplementedError(
                    "RWKV-VL v1 does not support pipeline parallelism"
                )
            if parallelism.context_parallel_degree > 1:
                if parallelism.context_parallel_load_balancer is not None:
                    raise ValueError(
                        "RWKV-VL CP requires --parallelism.context_parallel_load_balancer None"
                    )
                total_tokens = training.local_batch_size * training.seq_len
                if total_tokens % parallelism.context_parallel_degree != 0:
                    raise ValueError(
                        f"RWKV-VL CP requires local_batch_size * seq_len "
                        f"({total_tokens}) to be divisible by context_parallel_degree "
                        f"({parallelism.context_parallel_degree})"
                    )
                if (
                    compile_config is not None
                    and compile_config.enable
                    and "model" in compile_config.components
                ):
                    from torchtitan.tools.logging import logger

                    logger.warning(
                        "RWKV-VL CP with torch.compile is experimental and should "
                        "be checked with benchmarks/rwkv7_compile_bench.py before "
                        "large training runs."
                    )

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 6 * nparams

    def __init__(self, config: Config):
        super().__init__()
        vision_merge = config.vision_encoder.spatial_merge_size
        processor_merge = config.processor_spatial_merge_size
        if processor_merge < vision_merge or processor_merge % vision_merge != 0:
            raise ValueError(
                "processor_spatial_merge_size must be a positive integer "
                f"multiple of vision_encoder.spatial_merge_size; got "
                f"processor={processor_merge}, vision={vision_merge}"
            )
        extra_merge_size = processor_merge // vision_merge
        if config.proj.extra_merge_size != extra_merge_size:
            raise ValueError(
                "proj.extra_merge_size must match processor/vision merge ratio; "
                f"got projector={config.proj.extra_merge_size}, ratio={extra_merge_size}"
            )
        if config.proj.kind == "cross_attn":
            visual_indices = tuple(config.vision_encoder.deepstack_visual_indices)
            if tuple(sorted(set(visual_indices))) != visual_indices:
                raise ValueError(
                    "deepstack_visual_indices must be unique and increasing"
                )
            if any(
                index < 0 or index >= config.vision_encoder.n_layers
                for index in visual_indices
            ):
                raise ValueError("deepstack_visual_indices select a missing ViT layer")
            if any(
                index < 0 or index >= config.llm.num_hidden_layers
                for index in config.proj.language_layer_indices
            ):
                raise ValueError("language_layer_indices select a missing RWKV layer")
            config = replace(
                config,
                vision_encoder=replace(
                    config.vision_encoder, raw_deepstack_features=True
                ),
                proj=replace(
                    config.proj,
                    encoder_dim=config.vision_encoder.out_hidden_size,
                    vision_dim=config.vision_encoder.dim,
                    spatial_merge_size=vision_merge,
                    num_deepstack=len(visual_indices),
                ),
            )
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.processor_spatial_merge_size = processor_merge
        self.vision_encoder = config.vision_encoder.build()
        self.proj = config.proj.build()
        self.llm = config.llm.build()
        self.lm_head = (
            config.lm_head
            or Linear.Config(
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                bias=False,
                param_init=_output_linear_init(config.hidden_size),
            )
        ).build()
        self._cp_group = None
        self._vision_patch_sync_group = None
        self._trainable_roots = self._apply_root_lr_selection()

    def _apply_root_lr_selection(self) -> tuple[str, ...]:
        root_lrs = _validate_root_lrs(self.config.root_lrs)
        # lm_head shares the llm LR root: it trains iff the llm trains.
        module_roots = {
            "vision_encoder": (self.vision_encoder,),
            "proj": (self.proj,),
            "llm": (self.llm, self.lm_head),
        }
        for name, modules in module_roots.items():
            trainable = root_lrs[name] > 0
            for module in modules:
                module.requires_grad_(trainable)
        return tuple(name for name in _ROOT_MODULE_NAMES if root_lrs[name] > 0)

    def set_cp_process_group(self, cp_group) -> None:
        self._cp_group = cp_group

    def set_vision_patch_sync_process_group(self, group) -> None:
        self._vision_patch_sync_group = group

    def _build_cp_context(
        self,
        cu_seqlens_global: torch.Tensor | None,
        cu_seqlens_global_cpu: torch.Tensor | None,
    ) -> Any | None:
        if self._cp_group is None:
            return None
        if cu_seqlens_global is None:
            raise ValueError("RWKV-VL CP requires cu_seqlens_global")
        from torchtitan.models.rwkv7.model import _require_fla_ops

        ops = _require_fla_ops()
        return ops.build_cp_context(
            cu_seqlens_global,
            group=self._cp_group,
            cu_seqlens_cpu=cu_seqlens_global_cpu,
        )

    def _sync_flat_vision_patch_bucket(
        self,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, bool]:
        if not dist.is_available() or not dist.is_initialized():
            return pixel_values, grid_thw, False

        group = self._vision_patch_sync_group
        world_size = (
            dist.get_world_size(group) if group is not None else dist.get_world_size()
        )
        if world_size == 1:
            return pixel_values, grid_thw, False
        trainable_roots = getattr(self, "_trainable_roots", None)
        if trainable_roots is not None and not {
            "vision_encoder",
            "proj",
        }.intersection(trainable_roots):
            return pixel_values, grid_thw, False

        has_local = (
            pixel_values is not None
            and grid_thw is not None
            and pixel_values.dim() == 2
        )
        local_num_patch = int(pixel_values.shape[0]) if has_local else 0

        max_num_patch = torch.tensor(local_num_patch, dtype=torch.long, device=device)
        dist.all_reduce(max_num_patch, op=dist.ReduceOp.MAX, group=group)
        target_num_patch = int(max_num_patch.item())
        if target_num_patch == 0 or target_num_patch == local_num_patch:
            return pixel_values, grid_thw, False
        if not has_local:
            # Every rank must enter the FSDP-wrapped vision encoder when any
            # rank has visual input. Run a zero-valued dummy image through the
            # same flat path; the caller discards its features and adds a
            # zero-valued autograd edge so this rank contributes no vision
            # signal while still participating in parameter collectives.
            patch_dim = self.vision_encoder.patch_embed.proj.weight.shape[-1]
            dummy = torch.zeros(
                (target_num_patch, patch_dim),
                device=device,
                dtype=self.vision_encoder.patch_embed.proj.weight.dtype,
            )
            side = max(1, int(target_num_patch**0.5))
            while side > 1 and target_num_patch % side != 0:
                side -= 1
            dummy_grid = torch.tensor(
                [[1, side, target_num_patch // side]],
                dtype=torch.long,
                device=device,
            )
            return dummy, dummy_grid, True
        if target_num_patch < local_num_patch:
            raise RuntimeError(
                f"Rank-synchronized ViT patch bucket target {target_num_patch} "
                f"is smaller than local patch count {local_num_patch}."
            )

        pad_len = target_num_patch - local_num_patch
        pixel_values = torch.cat(
            [
                pixel_values,
                pixel_values.new_zeros((pad_len, pixel_values.shape[1])),
            ],
            dim=0,
        )
        return pixel_values, grid_thw, False

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor, torch.Tensor]:
        """Run the vision encoder + projector and return (
            main, deepstack_per_level, num_tokens_per_item, num_kv_per_item
        ).

        ``num_tokens_per_item`` is the per-image ``<image_pad>`` count set by
        ``processor_spatial_merge_size``. For the cross-attention projector,
        ``num_kv_per_item`` is the raw ViT patch count; for the MLP projector,
        it is the native vision-merged token count.
        """
        pixel_values = pixel_values.to(
            self.vision_encoder.patch_embed.proj.weight.dtype
        )
        merged_embeds, deepstack_features = self.vision_encoder(
            pixel_values,
            grid_thw=grid_thw,
        )
        merged_embeds, deepstack_features = self.proj(
            merged_embeds,
            deepstack_features,
            grid_thw=grid_thw,
        )
        num_kv_per_item = grid_thw.prod(-1)
        if self.proj.kind == "mlp":
            num_kv_per_item = num_kv_per_item // self.vision_encoder.spatial_merge_unit
        processor_unit = self.processor_spatial_merge_size**2
        num_tokens_per_item = grid_thw.prod(-1) // processor_unit
        return merged_embeds, deepstack_features, num_tokens_per_item, num_kv_per_item

    def _apply_vision_features(
        self,
        target: torch.Tensor,
        *,
        features: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
        reduce: str,
    ) -> torch.Tensor:
        if global_input_ids is None:
            global_input_ids = local_tokens
            shard_start = 0
        else:
            shard_start = int(global_start.item()) if global_start is not None else 0
        flat_target = target.view(-1, target.shape[-1])
        spans = _find_vision_spans(
            global_input_ids, num_tokens_per_item, vision_token_id
        )
        apply_vision_slices(
            flat_target,
            features,
            spans,
            num_tokens_per_item,
            shard_start=shard_start,
            shard_length=local_tokens.numel(),
            reduce=reduce,
            cast_to_target=(reduce == "add"),
        )
        return target

    def _add_zero_grad_edge(
        self,
        inputs_embeds: torch.Tensor,
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Add a 0-valued autograd edge from each tensor into ``inputs_embeds``.

        FSDP collective backward requires every rank that wraps the projector /
        vision encoder to enter backward; a CP rank or no-image batch may own
        no real vision contribution, so we splice in a zero-valued edge so the
        graph still passes through those modules.
        """
        for tensor in tensors:
            inputs_embeds = inputs_embeds + tensor.sum().to(inputs_embeds.dtype) * 0.0
        return inputs_embeds

    def _empty_projector_outputs(
        self,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Any]]:
        empty = inputs_embeds.new_zeros((0, self.proj.encoder_dim))
        if self.proj.kind == "mlp":
            empty_deepstack = [
                inputs_embeds.new_zeros((0, self.proj.encoder_dim))
                for _ in range(self.proj.num_deepstack)
            ]
            return self.proj(empty, empty_deepstack)

        empty_visual = inputs_embeds.new_zeros((0, self.proj.vision_dim))
        seed = self.proj.seed_q_proj(self.proj.seed_query_norm(empty))
        main = self.proj._project_output(
            self.proj.num_deepstack, self.proj.seed_output_norm(seed)
        )
        seed_keys, seed_values = self.proj._project_memory(
            empty_visual, depth=self.proj.num_deepstack
        )
        main = main + (seed_keys.sum() + seed_values.sum()).to(main.dtype) * 0.0
        deepstack = []
        for depth in range(self.proj.num_deepstack):
            keys, values = self.proj._project_memory(empty_visual, depth=depth)
            query_edge, gate_edge = self.proj._project_query_and_gate(
                depth,
                inputs_embeds.new_zeros((0, self.proj.project_dim)),
            )
            output_edge = self.proj._project_output(
                depth, inputs_embeds.new_zeros((0, self.proj.vision_dim))
            )
            keys = keys + query_edge.sum().to(keys.dtype) * 0.0
            keys = keys + gate_edge.sum().to(keys.dtype) * 0.0
            keys = keys + output_edge.sum().to(keys.dtype) * 0.0
            deepstack.append((keys, values))
        return main, deepstack

    def _make_mlp_injector(
        self,
        *,
        deepstack_features: list[Any],
        num_tokens_per_item: torch.Tensor | None,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ):
        """Build an ``after_layer`` callback that adds deepstack features into
        ``<image_pad>`` positions (the original additive DeepStack path).
        """

        def inject(idx: int, layer_hidden_states: torch.Tensor) -> torch.Tensor:
            if idx >= len(deepstack_features) or num_tokens_per_item is None:
                return layer_hidden_states
            return self._apply_vision_features(
                layer_hidden_states,
                features=deepstack_features[idx],
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=vision_token_id,
                global_input_ids=global_input_ids,
                global_start=global_start,
                local_tokens=local_tokens,
                reduce="add",
            )

        return inject

    def _make_cross_attn_injector(
        self,
        *,
        deepstack_features: list[Any],
        num_tokens_per_item: torch.Tensor | None,
        num_kv_per_item: torch.Tensor | None,
        vision_token_id: int,
        global_input_ids: torch.Tensor | None,
        global_start: torch.Tensor | None,
        local_tokens: torch.Tensor,
    ):
        """Build an ``after_layer`` callback that runs masked cross-attention
        between local ``<image_pad>`` queries and projected deepstack K/V.

        The block-diagonal mask (each image_pad attends only to its own
        image's K/V) and bucketed lengths are computed **once per forward**
        and reused across all deepstack levels.
        """
        device = local_tokens.device
        empty_callback = self._make_mlp_injector(
            deepstack_features=deepstack_features,
            num_tokens_per_item=num_tokens_per_item,
            vision_token_id=vision_token_id,
            global_input_ids=global_input_ids,
            global_start=global_start,
            local_tokens=local_tokens,
        )
        if (
            num_tokens_per_item is None
            or num_kv_per_item is None
            or len(deepstack_features) == 0
        ):
            return lambda idx, h: h

        if global_input_ids is None:
            global_input_ids_used = local_tokens
            shard_start = 0
        else:
            global_input_ids_used = global_input_ids
            shard_start = int(global_start.item()) if global_start is not None else 0
        shard_length = local_tokens.numel()

        spans = _find_vision_spans(
            global_input_ids_used, num_tokens_per_item, vision_token_id
        )

        # Gather local Q metadata: contiguous slice ranges in the flat local
        # token stream + per-row image_id label.
        local_ranges: list[
            tuple[int, int, int]
        ] = []  # (local_start, local_end, image_id)
        for span in spans:
            span_end = span.start + span.length
            overlap_start = max(span.start, shard_start)
            overlap_end = min(span_end, shard_start + shard_length)
            if overlap_start >= overlap_end:
                continue
            local_ranges.append(
                (
                    overlap_start - shard_start,
                    overlap_end - shard_start,
                    span.item_idx,
                )
            )

        if not local_ranges:
            # No image_pad falls on this rank's shard — fall back to a no-op
            # but still consume deepstack features for the autograd edge in
            # the CP zero-grad path.
            del empty_callback
            return lambda idx, h: h

        q_ladder = self.proj.config.q_buckets
        kv_ladder = self.proj.config.kv_buckets

        def _bucket_q(n: int) -> int:
            return _ceil_to_bucket(n, q_ladder) if q_ladder else _next_pow2_bucket(n)

        def _bucket_kv(n: int) -> int:
            required = n + 1
            return (
                _ceil_to_bucket(required, kv_ladder)
                if kv_ladder
                else _next_pow2_bucket(required)
            )

        # Group local Q ranges by their image_id for fast per-chunk filtering.
        ranges_by_image: dict[int, list[tuple[int, int]]] = {}
        for start, end, image_id in local_ranges:
            ranges_by_image.setdefault(image_id, []).append((start, end))

        num_kv_list = [int(n) for n in num_kv_per_item.tolist()]
        chunk_image_ranges = _pack_images_into_chunks(
            num_kv_list, _CROSS_ATTN_CHUNK_KV_TARGET
        )
        # Cumulative K/V offset per image, so chunk [img_lo, img_hi) maps to
        # a contiguous K/V slice [kv_cum[img_lo], kv_cum[img_hi]).
        kv_cum = [0]
        for n in num_kv_list:
            kv_cum.append(kv_cum[-1] + n)

        chunks: list[dict] = []
        for img_lo, img_hi in chunk_image_ranges:
            kv_start = kv_cum[img_lo]
            kv_end = kv_cum[img_hi]
            kv_real_len_chunk = kv_end - kv_start

            chunk_ranges: list[tuple[int, int, int]] = []
            for image_id in range(img_lo, img_hi):
                for start, end in ranges_by_image.get(image_id, ()):
                    chunk_ranges.append((start, end, image_id - img_lo))

            if not chunk_ranges:
                # No Q rows on this rank touch images in this chunk; skip.
                continue

            q_real_len_chunk = sum(end - start for start, end, _ in chunk_ranges)
            q_bucket_chunk = _bucket_q(q_real_len_chunk)
            kv_bucket_chunk = _bucket_kv(kv_real_len_chunk)

            q_image_id_chunk = torch.full(
                (q_bucket_chunk,), -1, dtype=torch.int32, device=device
            )
            cursor = 0
            for start, end, rel_id in chunk_ranges:
                length = end - start
                q_image_id_chunk[cursor : cursor + length] = rel_id
                cursor += length

            kv_image_id_chunk = torch.full(
                (kv_bucket_chunk,), -1, dtype=torch.int32, device=device
            )
            kv_cursor = 0
            for image_id in range(img_lo, img_hi):
                length = num_kv_list[image_id]
                kv_image_id_chunk[kv_cursor : kv_cursor + length] = image_id - img_lo
                kv_cursor += length

            # Per-chunk closure: mask_mod captures this chunk's id tensors so
            # each chunk gets its own compiled FlexAttention specialisation
            # against (q_bucket_chunk, kv_bucket_chunk).
            def _make_mask_mod(
                qid: torch.Tensor,
                kid: torch.Tensor,
                dummy_index: int,
            ):
                def mask_mod(b, h, q_idx, kv_idx):
                    q_id = qid[q_idx]
                    kv_id = kid[kv_idx]
                    valid_q = q_id >= 0
                    valid_kv = kv_id >= 0
                    same_image = q_id == kv_id
                    padding_dummy = (~valid_q) & (kv_idx == dummy_index)
                    return (same_image & valid_q & valid_kv) | padding_dummy

                return mask_mod

            block_mask_chunk = _compiled_create_block_mask(
                _make_mask_mod(
                    q_image_id_chunk,
                    kv_image_id_chunk,
                    kv_real_len_chunk,
                ),
                B=None,
                H=None,
                Q_LEN=q_bucket_chunk,
                KV_LEN=kv_bucket_chunk,
                device=device,
            )

            q_index_chunk = torch.empty(
                q_real_len_chunk, dtype=torch.long, device=device
            )
            cursor = 0
            for start, end, _ in chunk_ranges:
                length = end - start
                q_index_chunk[cursor : cursor + length] = torch.arange(
                    start, end, device=device
                )
                cursor += length

            chunks.append(
                {
                    "kv_start": kv_start,
                    "kv_end": kv_end,
                    "q_bucket": q_bucket_chunk,
                    "kv_bucket": kv_bucket_chunk,
                    "block_mask": block_mask_chunk,
                    "q_index": q_index_chunk,
                }
            )

        if not chunks:
            return lambda idx, h: h

        depth_by_layer = {
            layer_index: depth
            for depth, layer_index in enumerate(self.proj.language_layer_indices)
        }

        def inject(idx: int, layer_hidden_states: torch.Tensor) -> torch.Tensor:
            depth = depth_by_layer.get(idx)
            if depth is None:
                return layer_hidden_states
            k_all, v_all = deepstack_features[depth]
            flat = layer_hidden_states.reshape(-1, layer_hidden_states.shape[-1])
            for chunk in chunks:
                k_chunk = k_all[chunk["kv_start"] : chunk["kv_end"]]
                v_chunk = v_all[chunk["kv_start"] : chunk["kv_end"]]
                q_real = flat.index_select(0, chunk["q_index"])
                delta = self.proj.attend(
                    depth,
                    q_real,
                    k_chunk,
                    v_chunk,
                    block_mask=chunk["block_mask"],
                    q_bucket=chunk["q_bucket"],
                    kv_bucket=chunk["kv_bucket"],
                )
                # ``index_add`` (non-inplace) returns a fresh tensor; the in-place
                # variant would overwrite ``layer_hidden_states`` while the previous
                # LLM block's autograd graph still references it as a saved input.
                flat = flat.index_add(0, chunk["q_index"], delta.to(flat.dtype))
            return flat.view_as(layer_hidden_states)

        return inject

    def _prepare_inputs_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
        fla_cp_global_input_ids: torch.Tensor | None,
        fla_cp_global_start: torch.Tensor | None,
        vision_is_dummy: bool,
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor | None, torch.Tensor | None, int,]:
        inputs_embeds = self.llm.embeddings(tokens)
        image_token_id = (
            special_tokens.get("image_id", self.config.image_token_id)
            if special_tokens is not None
            else self.config.image_token_id
        )
        deepstack_features: list[Any] = []
        num_tokens_per_item: torch.Tensor | None = None
        num_kv_per_item: torch.Tensor | None = None
        if pixel_values is not None and grid_thw is not None:
            if vision_is_dummy:
                pixel_values = pixel_values.to(
                    self.vision_encoder.patch_embed.proj.weight.dtype
                )
                dummy_main, dummy_deepstack = self.vision_encoder(
                    pixel_values,
                    grid_thw=grid_thw,
                )
                inputs_embeds = self._add_zero_grad_edge(
                    inputs_embeds, dummy_main, *dummy_deepstack
                )

                # Exercise every projector parameter without running fake
                # cross-attention over the dummy image.
                empty_merged, empty_deepstack = self._empty_projector_outputs(
                    inputs_embeds
                )
                empty_tensors = [empty_merged]
                for entry in empty_deepstack:
                    if isinstance(entry, torch.Tensor):
                        empty_tensors.append(entry)
                    else:
                        empty_tensors.extend(entry)
                inputs_embeds = self._add_zero_grad_edge(inputs_embeds, *empty_tensors)
            else:
                (
                    merged_embeds,
                    deepstack_features,
                    num_tokens_per_item,
                    num_kv_per_item,
                ) = self._get_vision_embeds(
                    pixel_values,
                    grid_thw=grid_thw,
                )
                inputs_embeds = self._apply_vision_features(
                    inputs_embeds,
                    features=merged_embeds,
                    num_tokens_per_item=num_tokens_per_item,
                    vision_token_id=image_token_id,
                    global_input_ids=fla_cp_global_input_ids,
                    global_start=fla_cp_global_start,
                    local_tokens=tokens,
                    reduce="set",
                )
            if fla_cp_global_input_ids is not None and not vision_is_dummy:
                # CP v1 computes vision redundantly on every rank, but a rank
                # may own no image placeholder tokens after contiguous sharding.
                zero_grad_tensors = [merged_embeds]
                for entry in deepstack_features:
                    if isinstance(entry, torch.Tensor):
                        zero_grad_tensors.append(entry)
                    else:
                        zero_grad_tensors.extend(entry)
                inputs_embeds = self._add_zero_grad_edge(
                    inputs_embeds, *zero_grad_tensors
                )
        elif "proj" in self._trainable_roots:
            empty_merged, empty_deepstack = self._empty_projector_outputs(inputs_embeds)
            zero_grad_tensors = [empty_merged]
            for entry in empty_deepstack:
                if isinstance(entry, torch.Tensor):
                    zero_grad_tensors.append(entry)
                else:
                    zero_grad_tensors.extend(entry)
            inputs_embeds = self._add_zero_grad_edge(inputs_embeds, *zero_grad_tensors)
        return (
            inputs_embeds,
            deepstack_features,
            num_tokens_per_item,
            num_kv_per_item,
            image_token_id,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
        cu_seqlens_global: torch.Tensor | None = None,
        cu_seqlens_global_cpu: torch.Tensor | None = None,
        fla_cp_global_input_ids: torch.Tensor | None = None,
        fla_cp_global_start: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if pixel_values_videos is not None or grid_thw_videos is not None:
            raise NotImplementedError("RWKV-VL video inputs are not implemented yet")

        pixel_values, grid_thw, vision_is_dummy = self._sync_flat_vision_patch_bucket(
            pixel_values,
            grid_thw,
            device=tokens.device,
        )

        cp_context = self._build_cp_context(cu_seqlens_global, cu_seqlens_global_cpu)
        cu_seqlens = (
            cu_seqlens_global if cp_context is None and tokens.shape[0] == 1 else None
        )
        (
            inputs_embeds,
            deepstack_features,
            num_tokens_per_item,
            num_kv_per_item,
            image_token_id,
        ) = self._prepare_inputs_embeds(
            tokens,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            special_tokens=special_tokens,
            fla_cp_global_input_ids=fla_cp_global_input_ids,
            fla_cp_global_start=fla_cp_global_start,
            vision_is_dummy=vision_is_dummy,
        )

        if self.proj.kind == "cross_attn":
            inject_deepstack = self._make_cross_attn_injector(
                deepstack_features=deepstack_features,
                num_tokens_per_item=num_tokens_per_item,
                num_kv_per_item=num_kv_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
            )
        else:
            inject_deepstack = self._make_mlp_injector(
                deepstack_features=deepstack_features,
                num_tokens_per_item=num_tokens_per_item,
                vision_token_id=image_token_id,
                global_input_ids=fla_cp_global_input_ids,
                global_start=fla_cp_global_start,
                local_tokens=tokens,
            )

        hidden_states = self.llm.forward_embeddings(
            inputs_embeds,
            cp_context=cp_context,
            cu_seqlens=cu_seqlens,
            after_layer=inject_deepstack,
        )
        if self._skip_lm_head:
            return hidden_states
        return self.lm_head(hidden_states)
