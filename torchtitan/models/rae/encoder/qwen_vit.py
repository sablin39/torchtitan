# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-native Qwen3.5 vision encoder.

Drop-in replacement for the HF ``Qwen3_5VisionModel`` used as the frozen RAE
encoder. Loads the same safetensors weights (keys without the
``model.visual.`` prefix) and consumes the same packed patch layout:
``pixels_TD`` of shape ``(T, in_channels * temporal_patch_size * patch_size**2)``
plus a ``grid_thw`` metadata tensor, with tokens packed in spatial-merge
block-major order per image: (block_row, block_col, in_row, in_col).

The forward pass takes only static-shaped tensors (all data-dependent indexing
is precomputed by :meth:`QwenVisionEncoder.build_aux` outside the compiled
graph), so it runs under ``torch.compile(fullgraph=True, dynamic=False)`` for
a fixed token budget.

Tensor shape-suffix legend (letters are scoped to this file):
    T: total packed patch tokens, D: hidden_size, N: num attention heads,
    H: head dim, I: patch pixel dim (in_channels * temporal * patch**2),
    P: num position-embedding taps per token (4 for bilinear).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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


__all__ = [
    "QwenVisionAux",
    "QwenVisionConfig",
    "QwenVisionEncoder",
]
