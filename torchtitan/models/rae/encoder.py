# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen RAE Stage 1 encoder: Qwen3.5 vision tower plus the RAE adapter.

``QwenVisionEncoder`` is a drop-in replacement for the HF
``Qwen3_5VisionModel`` used as the frozen RAE encoder. It loads the same
safetensors weights (keys without the ``model.visual.`` prefix) and consumes
the same packed patch layout: ``pixels_TD`` of shape
``(T, in_channels * temporal_patch_size * patch_size**2)`` plus a ``grid_thw``
metadata tensor, with tokens packed in spatial-merge block-major order per
image: (block_row, block_col, in_row, in_col).

The tower forward takes only static-shaped tensors (all data-dependent
indexing is precomputed by :meth:`QwenVisionEncoder.build_aux` outside the
compiled graph), so it runs under ``torch.compile(fullgraph=True,
dynamic=False)`` for a fixed token budget.

``FrozenRAEEncoder`` wraps the tower with weight loading, the RAEv2
multi-layer merge, and static token padding (``pad_tokens_to``).
``kind='fixed'`` is a deterministic dependency-free conv projection for smoke
tests.

Tensor shape-suffix legend (letters are scoped to this file):
    T: total packed patch tokens, D: hidden_size, N: num attention heads,
    H: head dim, I: patch pixel dim (in_channels * temporal * patch**2),
    P: num position-embedding taps per token (4 for bilinear).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func


@dataclass(frozen=True, slots=True)
class QwenVisionConfig:
    """Vision tower settings; defaults match the Qwen3.5-0.8B vision config."""

    depth: int = 12
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


@dataclass(slots=True)
class QwenVisionAux:
    """Static-shaped per-batch indexing metadata from ``build_aux``."""

    cu_seqlens: torch.Tensor  # (num_docs + 1,) int32
    max_seqlen: int
    interp_indices: torch.Tensor  # (T, P) long
    interp_weights: torch.Tensor  # (T, P) float32
    position_ids: torch.Tensor  # (T, 2) long, (row, col) patch indices

    def to(self, device: torch.device) -> QwenVisionAux:
        return QwenVisionAux(
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            interp_indices=self.interp_indices.to(device),
            interp_weights=self.interp_weights.to(device),
            position_ids=self.position_ids.to(device),
        )


class QwenVisionPatchEmbed(nn.Module):
    def __init__(self, config: QwenVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        patch_dim = (
            config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
        )
        # Named ``proj`` so the HF conv3d checkpoint keys load unchanged; the
        # kernel == stride, so the conv is exactly this linear.
        self.proj = nn.Linear(patch_dim, self.embed_dim, bias=True)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "proj.weight"
        weight = state_dict.get(key)
        if weight is not None and weight.ndim == 5:
            state_dict[key] = weight.reshape(weight.shape[0], -1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, pixels_TD: torch.Tensor) -> torch.Tensor:
        # Compute through conv3d (kernel == stride) rather than the linear:
        # both are the same math, but this matches the HF model's kernel
        # accumulation order bit-for-bit, and the difference amplifies
        # through the blocks' large activation outliers.
        patch_shape = (
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x_TCTHW = pixels_TD.view(-1, *patch_shape)
        conv_weight = self.proj.weight.view(self.embed_dim, *patch_shape)
        return F.conv3d(
            x_TCTHW, conv_weight, self.proj.bias, stride=patch_shape[1:]
        ).view(-1, self.embed_dim)


class QwenVisionMLP(nn.Module):
    def __init__(self, config: QwenVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x_TD), approximate="tanh"))


class QwenVisionPatchMerger(nn.Module):
    """Pre-shuffle spatial merger; attribute names match the RAE merge helper."""

    def __init__(self, config: QwenVisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = False
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        x_TD = self.norm(x_TD).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x_TD)))


def _rotate_half(x_TNH: torch.Tensor) -> torch.Tensor:
    half = x_TNH.shape[-1] // 2
    return torch.cat((-x_TNH[..., half:], x_TNH[..., :half]), dim=-1)


def _axis_taps_weights(
    src: torch.Tensor, side: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bilinear taps into a ``side``-length table and their hat weights."""
    floor = torch.floor(src)
    frac = src - floor
    low = floor.long().clamp(0, side - 1)
    high = (floor.long() + 1).clamp(0, side - 1)
    weight_low = (1 - frac).clamp(min=0)
    weight_high = frac.clamp(min=0)
    return low, weight_low, high, weight_high


class QwenVisionAttention(nn.Module):
    def __init__(self, config: QwenVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(
        self,
        x_TD: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos_TH: torch.Tensor,
        sin_TH: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        qkv_T3NH = self.qkv(x_TD).view(num_tokens, 3, self.num_heads, self.head_dim)
        q_TNH, k_TNH, v_TNH = qkv_T3NH.unbind(dim=1)
        # RoPE in fp32, cast back to the input dtype, broadcast over heads.
        cos_T1H = cos_TH.unsqueeze(1)
        sin_T1H = sin_TH.unsqueeze(1)
        orig_dtype = q_TNH.dtype
        q_TNH = (q_TNH.float() * cos_T1H + _rotate_half(q_TNH.float()) * sin_T1H).to(
            orig_dtype
        )
        k_TNH = (k_TNH.float() * cos_T1H + _rotate_half(k_TNH.float()) * sin_T1H).to(
            orig_dtype
        )

        if orig_dtype in (torch.float16, torch.bfloat16):
            out_TNH = flash_attn_varlen_func(
                q_TNH,
                k_TNH,
                v_TNH,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=self.scaling,
                causal=False,
            )
        else:
            # flash-attn only supports fp16/bf16; the fp32 path is eager-only
            # (it syncs on cu_seqlens) and exists for high-precision parity
            # checks against the HF sdpa implementation.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outputs = []
            start = 0
            for length in lengths:
                q_1NTH = q_TNH[start : start + length].transpose(0, 1).unsqueeze(0)
                k_1NTH = k_TNH[start : start + length].transpose(0, 1).unsqueeze(0)
                v_1NTH = v_TNH[start : start + length].transpose(0, 1).unsqueeze(0)
                out_1NTH = F.scaled_dot_product_attention(
                    q_1NTH, k_1NTH, v_1NTH, scale=self.scaling
                )
                outputs.append(out_1NTH.squeeze(0).transpose(0, 1))
                start += length
            out_TNH = torch.cat(outputs, dim=0)
        return self.proj(out_TNH.reshape(num_tokens, -1))


class QwenVisionBlock(nn.Module):
    def __init__(self, config: QwenVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = QwenVisionAttention(config)
        self.mlp = QwenVisionMLP(config)

    def forward(
        self,
        x_TD: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cos_TH: torch.Tensor,
        sin_TH: torch.Tensor,
    ) -> torch.Tensor:
        x_TD = x_TD + self.attn(
            self.norm1(x_TD), cu_seqlens, max_seqlen, cos_TH, sin_TH
        )
        return x_TD + self.mlp(self.norm2(x_TD))


class QwenVisionEncoder(nn.Module):
    """Packed varlen Qwen3.5 vision tower with a static-shape forward."""

    def __init__(
        self,
        config: QwenVisionConfig,
        layer_indices: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.config = config
        invalid = [i for i in layer_indices if i < 0 or i >= config.depth]
        if invalid:
            raise ValueError(f"layer_indices out of range: {invalid}")
        self.layer_indices = tuple(sorted(set(layer_indices)))
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(math.isqrt(config.num_position_embeddings))
        if self.num_grid_per_side**2 != config.num_position_embeddings:
            raise ValueError("num_position_embeddings must be a perfect square")
        self.patch_embed = QwenVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.blocks = nn.ModuleList(
            QwenVisionBlock(config) for _ in range(config.depth)
        )
        self.merger = QwenVisionPatchMerger(config)

    def build_aux(
        self,
        grid_thw: torch.Tensor,
        max_seqlen: int | None = None,
        max_docs: int | None = None,
    ) -> QwenVisionAux:
        """Precompute indexing metadata from ``grid_thw`` (CPU or GPU tensor).

        ``max_seqlen`` may be pinned to a fixed upper bound (e.g. the token
        budget) so a compiled forward does not re-specialize when the true
        maximum document length changes; flash-attn only uses it for kernel
        scheduling. ``max_docs`` likewise pads cu_seqlens with zero-length
        trailing entries to a static 1 + max_docs length: token packing fills
        the budget with a variable document count, and an unpadded cu_seqlens
        would re-specialize the compiled graph on every batch.
        """
        grid_thw = torch.as_tensor(grid_thw, dtype=torch.long).reshape(-1, 3)
        merge = self.spatial_merge_size
        side = self.num_grid_per_side
        t_N, h_N, w_N = grid_thw.unbind(dim=1)
        bad = ((h_N % merge) != 0) | ((w_N % merge) != 0)
        if bool(bad.any()):
            index = int(bad.nonzero()[0])
            raise ValueError(
                f"grid ({int(h_N[index])}, {int(w_N[index])}) is not "
                f"divisible by spatial_merge_size {merge}"
            )
        # One entry per temporal frame: docs with t > 1 contribute t frames.
        frame_tokens_F = torch.repeat_interleave(h_N * w_N, t_N)
        h_F = torch.repeat_interleave(h_N, t_N)
        w_F = torch.repeat_interleave(w_N, t_N)
        cu_frames = torch.cat(
            [frame_tokens_F.new_zeros(1), frame_tokens_F.cumsum(dim=0)]
        )
        num_frames = frame_tokens_F.shape[0]
        device = grid_thw.device
        frame_id_T = torch.repeat_interleave(
            torch.arange(num_frames, device=device), frame_tokens_F
        )
        # The arithmetic below replays the exact float32 op sequence of the
        # original per-document loop; elementwise ops are deterministic per
        # element, so the vectorized result is bit-identical.
        within_T = (
            torch.arange(int(cu_frames[-1]), device=device) - cu_frames[:-1][frame_id_T]
        ).to(torch.float32)
        blocks_w_T = (w_F[frame_id_T] // merge).to(torch.float32)
        # Decode the block-major packed index within one frame:
        # (block_row, block_col, in_row, in_col).
        in_col = within_T % merge
        in_row = (within_T // merge) % merge
        block_col = (within_T // (merge * merge)) % blocks_w_T
        block_row = within_T // (merge * merge * blocks_w_T)
        row = block_row * merge + in_row
        col = block_col * merge + in_col
        # Bilinear (align_corners=True) resample of the side x side
        # position table: 2 taps per axis, outer product -> 4 taps.
        h_T = h_F[frame_id_T].to(torch.float32)
        w_T = w_F[frame_id_T].to(torch.float32)
        src_h = row * (side - 1) / (h_T - 1).clamp(min=1)
        src_w = col * (side - 1) / (w_T - 1).clamp(min=1)
        h_low, h_wlow, h_high, h_whigh = _axis_taps_weights(src_h, side)
        w_low, w_wlow, w_high, w_whigh = _axis_taps_weights(src_w, side)
        indices_T4 = torch.stack(
            (
                h_low * side + w_low,
                h_low * side + w_high,
                h_high * side + w_low,
                h_high * side + w_high,
            ),
            dim=-1,
        )
        weights_T4 = torch.stack(
            (
                h_wlow * w_wlow,
                h_wlow * w_whigh,
                h_whigh * w_wlow,
                h_whigh * w_whigh,
            ),
            dim=-1,
        )
        positions_T2 = torch.stack((row.long(), col.long()), dim=-1)
        true_max_seqlen = int((t_N * h_N * w_N).max())
        if max_seqlen is None:
            max_seqlen = true_max_seqlen
        elif max_seqlen < true_max_seqlen:
            raise ValueError(
                f"max_seqlen {max_seqlen} is below the true maximum {true_max_seqlen}"
            )
        if max_docs is not None:
            if num_frames > max_docs:
                raise ValueError(
                    f"batch has {num_frames} documents, above max_docs={max_docs}"
                )
            pad = max_docs - num_frames
            cu_frames = torch.cat([cu_frames, cu_frames[-1:].expand(pad)])
        return QwenVisionAux(
            cu_seqlens=cu_frames.to(torch.int32),
            max_seqlen=max_seqlen,
            interp_indices=indices_T4,
            interp_weights=weights_T4,
            position_ids=positions_T2,
        )

    def _rope_cos_sin(
        self, position_ids_T2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inv_freq is computed fresh from arange instead of stored as a
        # buffer (a buffer would be silently truncated by nn.Module.to).
        # HF stores it as a buffer, so the deployed HF model multiplies
        # position ids by inv_freq in the *module* dtype and takes cos/sin
        # there too; reproduce that quantization here so a bf16 deployment
        # matches HF bit-for-bit (fp32 inv_freq deviates by ~5e-2 norm-rel
        # on the late blocks' large activation outliers).
        compute_dtype = self.pos_embed.weight.dtype
        rope_dim = self.config.hidden_size // self.config.num_heads // 2
        inv_freq = (
            1.0
            / (
                self.config.rope_theta
                ** (
                    torch.arange(
                        0,
                        rope_dim,
                        2,
                        dtype=torch.float32,
                        device=position_ids_T2.device,
                    )
                    / rope_dim
                )
            )
        ).to(compute_dtype)
        rot_T2F = position_ids_T2.unsqueeze(-1) * inv_freq
        rot_TF = rot_T2F.flatten(1)
        emb_TH = torch.cat((rot_TF, rot_TF), dim=-1)
        return emb_TH.cos().float(), emb_TH.sin().float()

    def forward(
        self, pixels_TD: torch.Tensor, aux: QwenVisionAux
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return the final hidden states (T, D) and tapped block outputs."""
        if pixels_TD.shape[0] != aux.interp_indices.shape[0]:
            raise ValueError(
                "pixels_TD token count does not match aux: "
                f"{pixels_TD.shape[0]} != {aux.interp_indices.shape[0]}"
            )
        x_TD = self.patch_embed(pixels_TD)
        # fp32 weighted sum of the 4 interpolation taps, cast to input dtype.
        pos_TD = (
            self.pos_embed(aux.interp_indices).float()
            * aux.interp_weights.unsqueeze(-1)
        ).sum(dim=1)
        x_TD = x_TD + pos_TD.to(x_TD.dtype)
        cos_TH, sin_TH = self._rope_cos_sin(aux.position_ids)
        taps: list[torch.Tensor] = []
        tap_set = set(self.layer_indices)
        for index, block in enumerate(self.blocks):
            x_TD = block(x_TD, aux.cu_seqlens, aux.max_seqlen, cos_TH, sin_TH)
            if index in tap_set:
                taps.append(x_TD)
        return x_TD, taps


def _merge_qwen_hidden_states(
    taps: Sequence[torch.Tensor],
    final_hidden: torch.Tensor,
    merger: nn.Module,
    layer_indices: tuple[int, ...],
    tokens_per_item: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge selected Qwen vision blocks with RAEv2 multi-layer-sum semantics.

    ``taps`` holds the block outputs at ``layer_indices`` (in that order) and
    ``final_hidden`` the last block output. Each selected block output is
    normalized by the merger's LayerNorm (the Qwen vision tower has no
    separate final norm), the selected layers are averaged, and the per-item
    token mean of the final selected layer is added back as a global signal.
    The merger MLP runs once on the combined tokens. ``tokens_per_item``
    holds the pre-merger token counts of each packed media item so the
    global mean stays within its own image.
    """
    if not layer_indices:
        return merger(final_hidden)
    for attribute in ("norm", "linear_fc1", "act_fn", "linear_fc2", "hidden_size"):
        if not hasattr(merger, attribute):
            raise ValueError(
                "Qwen multi-layer merging requires a PatchMerger with norm, "
                "linear_fc1, act_fn, linear_fc2, and hidden_size"
            )
    if getattr(merger, "use_postshuffle_norm", False):
        raise ValueError("Qwen multi-layer merging requires a pre-shuffle merger norm")
    if len(taps) != len(layer_indices):
        raise ValueError(
            f"Qwen vision model returned {len(taps)} taps for "
            f"{len(layer_indices)} requested layer indices"
        )
    normed = [merger.norm(hidden) for hidden in taps]
    merged = torch.stack(normed).mean(dim=0)
    global_signal = normed[-1]
    if tokens_per_item is not None:
        counts = tokens_per_item.reshape(-1).to(
            device=global_signal.device, dtype=torch.long
        )
        if int(counts.sum().item()) != global_signal.shape[0]:
            raise ValueError(
                "Qwen tokens_per_item does not match the packed token count"
            )
        # Deterministic contiguous-segment means: a prefix sum keeps a fixed
        # accumulation order (index_add_ would use non-deterministic atomics).
        prefix = torch.cat(
            [
                global_signal.new_zeros(1, global_signal.shape[-1]),
                global_signal.float().cumsum(dim=0),
            ],
            dim=0,
        )
        ends = counts.cumsum(dim=0)
        starts = ends - counts
        segment_sums = prefix[ends] - prefix[starts]
        means = segment_sums / counts.clamp_min(1).unsqueeze(-1)
        item_ids = torch.repeat_interleave(
            torch.arange(counts.numel(), device=global_signal.device), counts
        )
        global_signal = means[item_ids].to(global_signal.dtype)
    else:
        global_signal = global_signal.mean(dim=0, keepdim=True)
    merged = merged + global_signal
    merged = merger.linear_fc2(
        merger.act_fn(merger.linear_fc1(merged.view(-1, merger.hidden_size)))
    )
    return merged


@dataclass(frozen=True, slots=True)
class RAEEncoderConfig:
    """Frozen encoder settings; Qwen accepts ``image_size=-1`` dynamically."""

    kind: str = "fixed"
    name: str = ""
    latent_dim: int = 768
    image_size: int = 256
    noise_tau: float = 0.8
    normalization_stat_path: str | None = None
    layer_indices: tuple[int, ...] = ()
    merge_size: int = 1
    dtype: str = "float32"
    compile: bool = False
    # Static token budget for kind='qwen': every microbatch is padded with one
    # trailing padding document up to this many patch tokens, so the native
    # Qwen vision encoder runs at a fixed shape and torch.compile never
    # recompiles. Aux tensor VALUES (cu_seqlens, interpolation indices,
    # position ids) still vary per step; only shapes are static.
    pad_tokens_to: int | None = None
    # Static flash-attn max_seqlen pin under pad_tokens_to: must bound the
    # longest varlen segment INCLUDING the padding document. Packing slack is
    # always below one max-size row, so the max pre-merge document length
    # (e.g. 4096) is a valid pin; pinning to the full pad_tokens_to budget
    # instead would make flash-attn schedule splits for a budget-length
    # segment and waste performance. build_aux rejects a pin below the true
    # longest segment, so a wrong value fails loudly.
    max_tokens_per_doc: int | None = None
    # Static cu_seqlens length under pad_tokens_to: token packing fills the
    # budget with a variable document count (~140 for OpenImages), so
    # zero-length trailing entries pad the doc axis to this bound. Hard
    # maximum is pad_tokens_to / min-doc-tokens (min_pixels 256x256 -> 256
    # pre-merge tokens); 2048 covers the 389120/256 = 1520 worst case.
    max_docs_per_microbatch: int = 2048

    def __post_init__(self) -> None:
        if self.image_size == 0 or (self.image_size < -1):
            raise ValueError("encoder.image_size must be -1 or positive")
        if self.kind != "qwen" and self.image_size == -1:
            raise ValueError("encoder.image_size=-1 is only supported for Qwen")
        if self.merge_size <= 0:
            raise ValueError("encoder.merge_size must be positive")
        if self.image_size != -1 and self.image_size % self.merge_size:
            raise ValueError("encoder.image_size must be divisible by merge_size")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported encoder dtype: {self.dtype}")
        if self.pad_tokens_to is not None:
            if self.kind != "qwen":
                raise ValueError("encoder.pad_tokens_to requires kind='qwen'")
            if self.pad_tokens_to <= 0:
                raise ValueError("encoder.pad_tokens_to must be positive")
        if self.max_tokens_per_doc is not None:
            if self.pad_tokens_to is None:
                raise ValueError("encoder.max_tokens_per_doc requires pad_tokens_to")
            if self.max_tokens_per_doc <= 0:
                raise ValueError("encoder.max_tokens_per_doc must be positive")


class FrozenRAEEncoder(nn.Module):
    """Frozen image-to-latent adapter for Stage 1.

    ``kind='qwen'`` loads the local Qwen vision tower and merger.
    ``kind='fixed'`` is deterministic and dependency-free for smoke tests.
    """

    def __init__(self, config: RAEEncoderConfig, device: torch.device) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.latent_dim = config.latent_dim
        self.noise_tau = config.noise_tau
        self.kind = config.kind
        self.layer_indices = config.layer_indices
        self.merge_size = config.merge_size
        self.supervision_image_size = (
            None if config.image_size == -1 else config.image_size // config.merge_size
        )
        self.latent_mean = None
        self.latent_var = None
        if config.normalization_stat_path is not None:
            stats = torch.load(
                config.normalization_stat_path,
                map_location="cpu",
                weights_only=True,
            )
            self.latent_mean = self._format_stat(stats.get("mean"), "mean")
            self.latent_var = self._format_stat(stats.get("var"), "var")
        self.external = None
        self.pad_tokens_to = config.pad_tokens_to
        self.max_tokens_per_doc = config.max_tokens_per_doc
        self.max_docs_per_microbatch = config.max_docs_per_microbatch
        self.last_grid_thw: torch.Tensor | None = None
        self.last_fps: torch.Tensor | None = None
        self.last_temporal_start: torch.Tensor | None = None
        if config.kind == "qwen":
            self._init_qwen(config, device)
        elif config.kind == "fixed":
            self.projection = nn.Conv2d(3, config.latent_dim, kernel_size=1)
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(0)
                self.projection.weight.copy_(
                    torch.randn(
                        self.projection.weight.shape,
                        generator=generator,
                    )
                    / math.sqrt(3)
                )
                self.projection.bias.zero_()
        else:
            raise ValueError(f"Unsupported RAE encoder kind: {config.kind}")
        self.to(device)
        self.eval()
        self.requires_grad_(False)
        if config.compile:
            if self.external is None:
                raise ValueError("encoder.compile requires an external backbone")
            if config.kind == "qwen" and config.pad_tokens_to is not None:
                # Fully static shapes (see RAEEncoderConfig.pad_tokens_to).
                self.external = torch.compile(
                    self.external, dynamic=False, fullgraph=True
                )
            else:
                self.external = torch.compile(self.external, dynamic=True)

    def _init_qwen(self, config: RAEEncoderConfig, device: torch.device) -> None:
        if not config.name:
            raise ValueError("Qwen encoder name is required")
        try:
            from safetensors import safe_open
            from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "encoder.kind='qwen' requires the transformers and safetensors "
                "packages"
            ) from error
        model_directory = Path(config.name).expanduser()
        if not model_directory.is_dir():
            raise ValueError(f"Qwen model directory does not exist: {model_directory}")
        full_config = AutoConfig.from_pretrained(
            str(model_directory), local_files_only=True
        )
        vision_config = getattr(full_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("Qwen checkpoint does not contain a vision_config")
        if vision_config.spatial_merge_size != config.merge_size:
            raise ValueError(
                "encoder.merge_size does not match Qwen vision config: "
                f"{config.merge_size} != {vision_config.spatial_merge_size}"
            )
        if config.image_size != -1 and config.image_size % (
            vision_config.patch_size * config.merge_size
        ):
            raise ValueError(
                "Qwen encoder image_size must be divisible by patch_size * merge_size"
            )
        visual = QwenVisionEncoder(
            QwenVisionConfig(
                depth=int(vision_config.depth),
                hidden_size=int(vision_config.hidden_size),
                num_heads=int(vision_config.num_heads),
                intermediate_size=int(vision_config.intermediate_size),
                patch_size=int(vision_config.patch_size),
                temporal_patch_size=int(vision_config.temporal_patch_size),
                spatial_merge_size=int(vision_config.spatial_merge_size),
                in_channels=int(vision_config.in_channels),
                out_hidden_size=int(vision_config.out_hidden_size),
                num_position_embeddings=int(vision_config.num_position_embeddings),
            ),
            layer_indices=config.layer_indices,
        )
        index_path = model_directory / "model.safetensors.index.json"
        if index_path.is_file():
            import json

            with index_path.open() as index_file:
                weight_map = json.load(index_file)["weight_map"]
            shard_names = sorted(
                {
                    name
                    for key, name in weight_map.items()
                    if key.startswith("model.visual.")
                }
            )
        else:
            shard_names = sorted(model_directory.glob("*.safetensors"))
            shard_names = [path.name for path in shard_names]
        visual_state: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard_path = model_directory / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key.startswith("model.visual."):
                        visual_state[key[len("model.visual.") :]] = shard.get_tensor(
                            key
                        )
        visual.load_state_dict(visual_state, strict=True)
        encoder_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.external = visual.to(device=device, dtype=encoder_dtype).eval()
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(model_directory), local_files_only=True
            )
        except (OSError, ValueError):
            self.processor = AutoImageProcessor.from_pretrained(
                str(model_directory), local_files_only=True
            )
        self._qwen_patch_size = int(vision_config.patch_size)
        self._qwen_depth = int(vision_config.depth)
        external_dim = int(vision_config.out_hidden_size)
        if external_dim != config.latent_dim:
            raise ValueError(
                "Qwen post-merger width does not match decoder latent_dim: "
                f"{external_dim} != {config.latent_dim}"
            )
        invalid_indices = [
            index
            for index in config.layer_indices
            if index < 0 or index >= self._qwen_depth
        ]
        if invalid_indices:
            raise ValueError(
                f"Qwen encoder layer indices are out of range: {invalid_indices}"
            )

    def _format_stat(
        self, value: torch.Tensor | None, name: str
    ) -> torch.Tensor | None:
        if value is None:
            return None
        value = torch.as_tensor(value)
        if value.ndim == 1:
            value = value.view(1, -1, 1, 1)
        elif value.ndim == 3:
            value = value.unsqueeze(0)
        elif value.ndim != 4:
            raise ValueError(
                f"RAE encoder {name} statistics must have shape (C,), (C, H, W), "
                "or (1, C, H, W)"
            )
        if value.shape[1] != self.latent_dim:
            raise ValueError(
                f"RAE encoder {name} statistics have {value.shape[1]} channels, "
                f"expected {self.latent_dim}"
            )
        return value

    @staticmethod
    def _as_btchw(media: Any) -> torch.Tensor:
        """Normalize one image to a single-frame (1, C, H, W) float tensor."""
        if not isinstance(media, torch.Tensor):
            import io

            import numpy as np
            from PIL import Image

            if isinstance(media, (bytes, bytearray)):
                media = Image.open(io.BytesIO(media)).convert("RGB")
            if hasattr(media, "convert"):
                media = np.array(media.convert("RGB"), copy=True)
            media = torch.from_numpy(np.asarray(media))
        media = media.float()
        if media.numel() and media.max() > 1:
            media = media / 255.0
        if media.ndim != 3:
            raise ValueError("RAE encoder images must have CHW or HWC dimensions")
        if media.shape[-1] in (1, 3, 4):
            media = media[..., :3].permute(2, 0, 1)
        elif media.shape[0] not in (1, 3, 4):
            raise ValueError("RAE encoder images must be HWC or CHW with 3 channels")
        else:
            media = media[:3]
        return media.unsqueeze(0)

    def forward(
        self,
        images_BTCHW: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, Any],
        *,
        return_grid_thw: bool = False,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.last_grid_thw = None
        self.last_fps = None
        self.last_temporal_start = None

        def scalar_values(
            value: torch.Tensor | float | None,
            batch_size: int,
            default: float,
        ) -> list[float]:
            if value is None:
                return [default] * batch_size
            if isinstance(value, torch.Tensor):
                values = value.detach().float().flatten().tolist()
            else:
                values = [float(value)]
            if len(values) == 1:
                values *= batch_size
            if len(values) != batch_size:
                raise ValueError("RAE metadata must be scalar or one value per sample")
            return [float(item) for item in values]

        processor_output: Mapping[str, Any] | None = None
        images_BCHW: torch.Tensor | None = None
        batch_size: int
        if isinstance(images_BTCHW, Mapping):
            if self.kind != "qwen":
                raise ValueError(
                    "Preprocessed Qwen mappings require encoder.kind='qwen'"
                )
            processor_output = images_BTCHW
            pixel_key = next(
                (
                    key
                    for key in ("pixel_values", "input")
                    if processor_output.get(key) is not None
                ),
                None,
            )
            grid_key = next(
                (
                    key
                    for key in ("image_grid_thw", "grid_thw")
                    if processor_output.get(key) is not None
                ),
                None,
            )
            if pixel_key is None or grid_key is None:
                raise ValueError(
                    "Qwen encoder mappings require pixel_values and grid_thw metadata"
                )
            grid_thw = torch.as_tensor(processor_output[grid_key])
            batch_size = int(grid_thw.reshape(-1, 3).shape[0])
        elif isinstance(images_BTCHW, torch.Tensor):
            raw_media = images_BTCHW.float()
            if raw_media.numel() and raw_media.max() > 1:
                raw_media = raw_media / 255.0
            if raw_media.ndim != 4:
                raise ValueError("RAE encoder tensor inputs must be BCHW images")
            images_BCHW = raw_media
            batch_size = raw_media.shape[0]
            if self.kind == "qwen":
                processor_output = self.processor(
                    images=raw_media.detach(),
                    do_rescale=False,
                    return_tensors="pt",
                )
        else:
            media_items = list(images_BTCHW)
            if not media_items:
                raise ValueError("RAE encoder requires at least one image")
            if self.kind == "qwen":
                media_items = [self._as_btchw(item) for item in media_items]
                batch_size = len(media_items)
                processor_output = self.processor(
                    images=[item[0].detach() for item in media_items],
                    do_rescale=False,
                    return_tensors="pt",
                )
            else:
                if any(item.ndim != 3 for item in media_items):
                    raise ValueError("RAE encoder image lists must contain CHW tensors")
                images_BCHW = torch.stack([item.float() for item in media_items])
                batch_size = images_BCHW.shape[0]

        if self.kind != "qwen" and images_BCHW is not None:
            if images_BCHW.shape[1] != 3:
                raise ValueError("RAE images must have three channels")
            if images_BCHW.shape[-2:] != (self.image_size, self.image_size):
                images_BCHW = F.interpolate(
                    images_BCHW,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

        tokens_BLC: torch.Tensor | None = None
        latents_BCHW: torch.Tensor | None = None
        if self.external is not None and self.kind == "qwen":
            assert processor_output is not None
            pixel_key = next(
                key
                for key in ("pixel_values", "input")
                if processor_output.get(key) is not None
            )
            grid_key = next(
                key
                for key in ("image_grid_thw", "grid_thw")
                if processor_output.get(key) is not None
            )
            grid_thw = torch.as_tensor(processor_output[grid_key])
            external_device = next(self.external.parameters()).device
            external_dtype = next(self.external.parameters()).dtype
            model_inputs = {
                name: (
                    value.to(device=external_device, dtype=external_dtype)
                    if name == pixel_key
                    else value.to(device=external_device)
                )
                for name, value in processor_output.items()
                if name == pixel_key or name == grid_key
            }
            pixels_TD = model_inputs[pixel_key]
            grid_thw = grid_thw.reshape(-1, 3).to(dtype=torch.long)
            real_grid_thw = grid_thw
            num_real_tokens = int(grid_thw.prod(dim=-1).sum().item())
            if num_real_tokens != pixels_TD.shape[0]:
                raise ValueError(
                    "Qwen grid_thw token count does not match pixel rows: "
                    f"{num_real_tokens} != {pixels_TD.shape[0]}"
                )
            if self.pad_tokens_to is not None:
                pad_len = self.pad_tokens_to - num_real_tokens
                if pad_len < 0:
                    raise ValueError(
                        f"Qwen microbatch has {num_real_tokens} patch tokens, "
                        f"above encoder.pad_tokens_to={self.pad_tokens_to}"
                    )
                if pad_len > 0:
                    # One trailing padding document keeps the pack at the
                    # static token budget. h=2 requires pad_len divisible by 4
                    # (real documents have even h, w, so their token counts
                    # are multiples of 4 already).
                    if pad_len % 4:
                        raise ValueError(
                            f"Qwen padding of {pad_len} tokens is not "
                            "expressible as an even (h, w) document"
                        )
                    pad_grid = grid_thw.new_tensor([[1, 2, pad_len // 2]])
                    grid_thw = torch.cat([grid_thw, pad_grid], dim=0)
                    pixels_TD = torch.cat(
                        [
                            pixels_TD,
                            pixels_TD.new_zeros(pad_len, pixels_TD.shape[1]),
                        ],
                        dim=0,
                    )
            with torch.no_grad(), torch.autocast(
                device_type=external_device.type, enabled=False
            ):
                # Pin max_seqlen under a static budget so the compiled graph
                # never re-specializes on the python int; flash-attn only
                # uses it for kernel scheduling. Prefer max_tokens_per_doc
                # (a tight bound on the longest segment, padding document
                # included) over the full budget. Aux values vary per step;
                # shapes are static. Autocast is forced off so the compiled
                # frame sees one global state from every caller: training
                # encodes outside autocast while validation wraps the whole
                # encode-decode in autocast, and fullgraph=True hard-fails
                # on the recompile that an autocast-state guard flip causes.
                aux = self.external.build_aux(
                    grid_thw,
                    max_seqlen=self.max_tokens_per_doc or self.pad_tokens_to,
                    max_docs=(
                        self.max_docs_per_microbatch
                        if self.pad_tokens_to is not None
                        else None
                    ),
                )
                final_TD, taps = self.external(pixels_TD, aux.to(external_device))
            merged_hidden_states = _merge_qwen_hidden_states(
                taps,
                final_TD,
                self.external.merger,
                self.layer_indices,
                tokens_per_item=grid_thw.prod(dim=-1),
            )
            grid_thw = real_grid_thw
            if self.pad_tokens_to is not None:
                # Drop the padding document's merged tokens and grid row.
                num_real_merged = num_real_tokens // self.merge_size**2
                merged_hidden_states = merged_hidden_states[:num_real_merged]
            grid_thw = grid_thw.to(
                device=merged_hidden_states.device, dtype=torch.long
            ).reshape(-1, 3)
            tokens_per_item = grid_thw.prod(dim=-1) // self.merge_size**2
            post_merge_grid_thw = grid_thw.clone()
            post_merge_grid_thw[:, 1:] //= self.merge_size
            self.last_grid_thw = post_merge_grid_thw.detach().cpu()
            same_grid = torch.all(post_merge_grid_thw == post_merge_grid_thw[0])
            if torch.all(tokens_per_item == tokens_per_item[0]) and same_grid:
                tokens_BLC = merged_hidden_states.view(
                    batch_size, int(tokens_per_item[0].item()), self.latent_dim
                )
            else:
                tokens_BLC = merged_hidden_states
            fps_source = processor_output.get("fps") if fps is None else fps
            fps_values = scalar_values(fps_source, batch_size, 0.0)
            self.last_fps = torch.tensor(fps_values, device=merged_hidden_states.device)
            start_values = scalar_values(temporal_start, batch_size, 0.0)
            self.last_temporal_start = torch.tensor(
                start_values, device=merged_hidden_states.device
            )
        elif self.kind == "fixed":
            assert images_BCHW is not None
            latent_side = self.image_size // 16
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(latent_side, latent_side),
                mode="bilinear",
                align_corners=False,
            )
            latents_BCHW = self.projection(images_BCHW)

        if self.external is not None:
            if tokens_BLC is None:
                raise RuntimeError("RAE external encoder did not return tokens")
            if tokens_BLC.ndim == 3 and self.kind == "qwen" and return_grid_thw:
                latents = tokens_BLC.reshape(-1, tokens_BLC.shape[-1])
            elif tokens_BLC.ndim == 3:
                side = int(math.sqrt(tokens_BLC.shape[1]))
                if side * side != tokens_BLC.shape[1]:
                    if return_grid_thw:
                        latents = tokens_BLC
                    else:
                        raise ValueError(
                            "Qwen encoder returned a non-square token grid; "
                            "request return_grid_thw=True for variable resolution"
                        )
                else:
                    latents_BCHW = tokens_BLC.transpose(1, 2).reshape(
                        batch_size, self.latent_dim, side, side
                    )
                    latents = latents_BCHW
            else:
                latents = tokens_BLC
        else:
            if latents_BCHW is None:
                raise RuntimeError("RAE encoder did not return image latents")
            latents = latents_BCHW
        if self.latent_mean is not None or self.latent_var is not None:
            if latents.ndim == 2:
                mean_shape = (1, self.latent_dim)
                variance_shape = (1, self.latent_dim)
            elif latents.ndim == 3:
                mean_shape = (1, 1, self.latent_dim)
                variance_shape = (1, 1, self.latent_dim)
            else:
                mean_shape = (
                    self.latent_mean.shape
                    if self.latent_mean is not None
                    else (1, self.latent_dim, 1, 1)
                )
                variance_shape = (
                    self.latent_var.shape
                    if self.latent_var is not None
                    else (1, self.latent_dim, 1, 1)
                )
            latent_mean = self.latent_mean
            if latent_mean is not None:
                latent_mean = latent_mean.to(latents.device, dtype=latents.dtype)
                if latents.ndim < 4 and latent_mean.ndim == 4:
                    latent_mean = latent_mean.mean(dim=(2, 3))
                latent_mean = latent_mean.reshape(mean_shape)
            else:
                latent_mean = 0
            latent_var = self.latent_var
            if latent_var is not None:
                latent_var = latent_var.to(latents.device, dtype=latents.dtype)
                if latents.ndim < 4 and latent_var.ndim == 4:
                    latent_var = latent_var.mean(dim=(2, 3))
                latent_var = latent_var.reshape(variance_shape)
            else:
                latent_var = 1
            latents = (latents - latent_mean) / torch.sqrt(latent_var + 1e-5)
        if return_grid_thw:
            if self.last_grid_thw is None:
                if latents.ndim == 4:
                    self.last_grid_thw = torch.tensor(
                        [[1, latents.shape[-2], latents.shape[-1]]],
                        dtype=torch.long,
                    )
                else:
                    raise ValueError(
                        "return_grid_thw requires Qwen grid metadata for token latents"
                    )
            return latents, self.last_grid_thw.to(latents.device)
        return latents


__all__ = [
    "FrozenRAEEncoder",
    "QwenVisionAux",
    "QwenVisionConfig",
    "QwenVisionEncoder",
    "RAEEncoderConfig",
]
