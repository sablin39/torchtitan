# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DINOv3 ViT-B/16 feature discriminator for RAE Stage 1.

``DINOv3ViTBackbone`` reimplements the Hugging Face ``DINOv3ViTModel``
forward pass (transformers ``models/dinov3_vit``) without the HF dependency,
loading the same safetensors checkpoint (keys ``embeddings.*``, ``layer.N.*``,
``norm.*``) with ``load_state_dict(..., strict=True)``. The backbone is a
frozen feature extractor: it always runs in eval mode and accepts a variable
input resolution per call (sides divisible by the patch size).

``RAEFeatureDiscriminator`` stacks trainable spectral-norm heads on the frozen
backbone. Images pass through the backbone unresized (the RoPE generalizes to
any resolution), so the discriminator scores the reconstruction at the
decoder's output resolution. Each head emits one logit per patch token:
forward returns a (B, H, L) tensor for a stacked BCHW batch and a list of
(H, L_i) tensors for a variable-resolution CHW sequence.
``backbone_kind='fixed'`` substitutes a small deterministic conv pyramid for
CPU smoke tests.

Tensor dimension legend (letters are scoped to this file):
    B = batch, C = channels (3 for the image, hidden_size for tokens),
    H = image height, W = image width, L = tokens (prefix + patches),
    P = patch tokens, N = attention heads, D = head dim,
    K = flattened patch pixels (C*ph*pw).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Literal, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from safetensors.torch import load_file

from .discriminator import FrozenImageFeatures
from .perceptual import grouped_per_image_loss


class DINOv3ViTEmbeddings(nn.Module):
    """Patch embedding plus cls/register prefix tokens.

    ``mask_token`` exists in the checkpoint but is only used for masked
    pre-training; it is kept as a parameter so a strict load succeeds and is
    never read at inference.

    The patch embedding is a strided Conv2d in the checkpoint, but is
    implemented here as a pixel-unshuffle reshape plus a Linear: under
    torch.compile(dynamic=True) an inductor convolution backward installs
    shape-equality guards on the frame, re-specializing (and recompiling) for
    every new input resolution, which blows dynamo's recompile limit during
    native-resolution discrimination. Reshape+Linear stays dynamic across
    shapes.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_channels: int,
        patch_size: int,
        num_register_tokens: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_register_tokens, hidden_size)
        )
        self.patch_embeddings = nn.Linear(
            num_channels * patch_size * patch_size, hidden_size
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = prefix + "patch_embeddings.weight"
        weight = state_dict.get(weight_key)
        if weight is not None and weight.ndim == 4:
            # The checkpoint stores the patch embedding as a Conv2d kernel
            # (O, C, ph, pw); flatten to the equivalent Linear weight
            # (O, C*ph*pw), matching the (C, ph, pw) inner order produced by
            # the pixel unshuffle in forward.
            state_dict[weight_key] = weight.flatten(1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, pixel_values_B3HW: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values_B3HW.shape
        patch = self.patch_size
        # Pixel unshuffle: (B, C, Hp, ph, Wp, pw) -> (B, Hp, Wp, C, ph, pw)
        # -> (B, Hp*Wp, C*ph*pw), row-major patch order, (C, ph, pw) inner.
        patches_BPK = (
            pixel_values_B3HW.view(
                batch_size, num_channels, height // patch, patch, width // patch, patch
            )
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, -1, num_channels * patch * patch)
        )
        patches_BPC = self.patch_embeddings(patches_BPK)
        cls_B1C = self.cls_token.expand(batch_size, -1, -1)
        registers_BRC = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat([cls_B1C, registers_BRC, patches_BPC], dim=1)


def _rotate_half(x_BHLD: torch.Tensor) -> torch.Tensor:
    half = x_BHLD.shape[-1] // 2
    return torch.cat((-x_BHLD[..., half:], x_BHLD[..., :half]), dim=-1)


def _apply_rope(
    q_BHLD: torch.Tensor,
    k_BHLD: torch.Tensor,
    cos_PD: torch.Tensor,
    sin_PD: torch.Tensor,
    num_prefix_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # RoPE applies to patch tokens only; the cls/register prefix is untouched.
    q_prefix_BHTD, q_patches_BHPD = q_BHLD.split(
        (num_prefix_tokens, cos_PD.shape[0]), dim=-2
    )
    k_prefix_BHTD, k_patches_BHPD = k_BHLD.split(
        (num_prefix_tokens, cos_PD.shape[0]), dim=-2
    )
    q_patches_BHPD = q_patches_BHPD * cos_PD + _rotate_half(q_patches_BHPD) * sin_PD
    k_patches_BHPD = k_patches_BHPD * cos_PD + _rotate_half(k_patches_BHPD) * sin_PD
    return (
        torch.cat((q_prefix_BHTD, q_patches_BHPD), dim=-2),
        torch.cat((k_prefix_BHTD, k_patches_BHPD), dim=-2),
    )


class DINOv3ViTAttention(nn.Module):
    def __init__(
        self, *, hidden_size: int, num_heads: int, num_prefix_tokens: int
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens
        # The DINOv3 checkpoint carries no key projection bias.
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x_BLC: torch.Tensor,
        cos_PD: torch.Tensor,
        sin_PD: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = x_BLC.shape
        q_BHLD = (
            self.q_proj(x_BLC)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k_BHLD = (
            self.k_proj(x_BLC)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_BHLD = (
            self.v_proj(x_BLC)
            .view(batch_size, num_tokens, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q_BHLD, k_BHLD = _apply_rope(
            q_BHLD, k_BHLD, cos_PD, sin_PD, self.num_prefix_tokens
        )
        out_BHLD = F.scaled_dot_product_attention(
            q_BHLD, k_BHLD, v_BHLD, scale=self.scaling
        )
        out_BLC = out_BHLD.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.o_proj(out_BLC)

    def forward_varlen(
        self,
        x_TC: torch.Tensor,
        cos_T1D: torch.Tensor,
        sin_T1D: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Block-diagonal attention over packed per-image documents.

        ``cos_T1D``/``sin_T1D`` cover every packed token; prefix rows carry
        cos=1/sin=0 so the cls/register tokens stay untouched, matching the
        prefix skip in the dense ``forward``.
        """
        num_tokens = x_TC.shape[0]
        q_TND = self.q_proj(x_TC).view(num_tokens, self.num_heads, self.head_dim)
        k_TND = self.k_proj(x_TC).view(num_tokens, self.num_heads, self.head_dim)
        v_TND = self.v_proj(x_TC).view(num_tokens, self.num_heads, self.head_dim)
        q_TND = q_TND * cos_T1D + _rotate_half(q_TND) * sin_T1D
        k_TND = k_TND * cos_T1D + _rotate_half(k_TND) * sin_T1D
        if q_TND.dtype in (torch.float16, torch.bfloat16):
            out_TND = flash_attn_varlen_func(
                q_TND,
                k_TND,
                v_TND,
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
            # checks against the dense sdpa implementation.
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            outputs = []
            start = 0
            for length in lengths:
                q_1NTD = q_TND[start : start + length].transpose(0, 1).unsqueeze(0)
                k_1NTD = k_TND[start : start + length].transpose(0, 1).unsqueeze(0)
                v_1NTD = v_TND[start : start + length].transpose(0, 1).unsqueeze(0)
                out_1NTD = F.scaled_dot_product_attention(
                    q_1NTD, k_1NTD, v_1NTD, scale=self.scaling
                )
                outputs.append(out_1NTD.squeeze(0).transpose(0, 1))
                start += length
            out_TND = torch.cat(outputs, dim=0)
        return self.o_proj(out_TND.reshape(num_tokens, -1))


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        return x_BLC * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        # DINOv3 uses the exact (erf) GELU, the F.gelu default.
        return self.down_proj(F.gelu(self.up_proj(x_BLC)))


class DINOv3ViTLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_prefix_tokens: int,
        intermediate_size: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = DINOv3ViTAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_prefix_tokens=num_prefix_tokens,
        )
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size=hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = DINOv3ViTMLP(
            hidden_size=hidden_size, intermediate_size=intermediate_size
        )
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size=hidden_size)

    def forward(
        self,
        x_BLC: torch.Tensor,
        cos_PD: torch.Tensor,
        sin_PD: torch.Tensor,
    ) -> torch.Tensor:
        x_BLC = x_BLC + self.layer_scale1(
            self.attention(self.norm1(x_BLC), cos_PD, sin_PD)
        )
        x_BLC = x_BLC + self.layer_scale2(self.mlp(self.norm2(x_BLC)))
        return x_BLC

    def forward_varlen(
        self,
        x_TC: torch.Tensor,
        cos_T1D: torch.Tensor,
        sin_T1D: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        x_TC = x_TC + self.layer_scale1(
            self.attention.forward_varlen(
                self.norm1(x_TC), cos_T1D, sin_T1D, cu_seqlens, max_seqlen
            )
        )
        x_TC = x_TC + self.layer_scale2(self.mlp(self.norm2(x_TC)))
        return x_TC


class DINOv3ViTBackbone(nn.Module):
    """Frozen DINOv3 ViT-B/16 feature extractor for the RAE discriminator.

    ``forward`` returns the normed final-block output followed by the normed
    output of each block in ``key_depths`` (0-indexed), each with the prefix
    tokens dropped and transposed to (B, hidden_size, num_patches) -- the
    layout the discriminator heads consume.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        num_channels: int = 3,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 100.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_prefix_tokens = 1 + num_register_tokens
        self.rope_theta = rope_theta
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.embeddings = DINOv3ViTEmbeddings(
            hidden_size=hidden_size,
            num_channels=num_channels,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
        )
        self.layer = nn.ModuleList(
            DINOv3ViTLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_prefix_tokens=self.num_prefix_tokens,
                intermediate_size=intermediate_size,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _rope_cos_sin(
        self, num_patches_h: int, num_patches_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Computed fresh in fp32 every forward so torch.compile keeps the
        # frequencies in fp32 (a stored buffer could be cast to bf16 by an
        # outer mixed-precision wrapper).
        coords_h_1d = (
            torch.arange(0.5, num_patches_h, dtype=torch.float32, device=device)
            / num_patches_h
        )
        coords_w_1d = (
            torch.arange(0.5, num_patches_w, dtype=torch.float32, device=device)
            / num_patches_w
        )
        coords_P2 = torch.stack(
            torch.meshgrid(coords_h_1d, coords_w_1d, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords_P2 = 2.0 * coords_P2 - 1.0
        inv_freq_F = 1.0 / self.rope_theta ** torch.arange(
            0, 1, 4 / self.head_dim, dtype=torch.float32, device=device
        )
        angles_P2F = 2 * math.pi * coords_P2[:, :, None] * inv_freq_F[None, None, :]
        angles_PD = angles_P2F.flatten(1, 2).tile(2)
        return torch.cos(angles_PD), torch.sin(angles_PD)

    def forward(
        self,
        pixel_values_B3HW: torch.Tensor,
        key_depths: tuple[int, ...] = (2, 5, 8, 11),
    ) -> list[torch.Tensor]:
        height, width = pixel_values_B3HW.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "DINOv3 backbone image sides must be divisible by the patch "
                f"size {self.patch_size}, got {(height, width)}"
            )
        if any(not 0 <= depth < len(self.layer) for depth in key_depths):
            raise ValueError(
                f"DINOv3 backbone key_depths must index blocks 0..{len(self.layer) - 1}, "
                f"got {key_depths}"
            )
        cos_PD, sin_PD = self._rope_cos_sin(
            height // self.patch_size,
            width // self.patch_size,
            pixel_values_B3HW.device,
        )
        # Match the module compute dtype (fp32 by default, bf16 when the
        # discriminator casts the frozen backbone): inputs arrive in the
        # caller's dtype, and the fp32 RoPE tables cast at application.
        compute_dtype = self.embeddings.patch_embeddings.weight.dtype
        pixel_values_B3HW = pixel_values_B3HW.to(compute_dtype)
        cos_PD = cos_PD.to(compute_dtype)
        sin_PD = sin_PD.to(compute_dtype)
        x_BLC = self.embeddings(pixel_values_B3HW)
        block_outputs_BLC = []
        for block in self.layer:
            x_BLC = block(x_BLC, cos_PD, sin_PD)
            block_outputs_BLC.append(x_BLC)
        normed_BLC = [self.norm(block_outputs_BLC[-1])]
        normed_BLC += [self.norm(block_outputs_BLC[depth]) for depth in key_depths]
        return [
            activation_BLC[:, self.num_prefix_tokens :].transpose(1, 2)
            for activation_BLC in normed_BLC
        ]

    def forward_varlen(
        self,
        patches_PK: torch.Tensor,
        patch_dest_P: torch.Tensor,
        prefix_rows_R: torch.Tensor,
        cos_TD: torch.Tensor,
        sin_TD: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        key_depths: tuple[int, ...] = (2, 5, 8, 11),
    ) -> list[torch.Tensor]:
        """One packed forward over many images with block-diagonal attention.

        ``patches_PK`` holds every image's flattened, normalized patches
        (row-major, (C, ph, pw) inner order, as the dense pixel unshuffle
        produces). ``patch_dest_P`` maps each packed patch to its row in the
        token sequence; ``prefix_rows_R`` lists the cls/register rows (each
        document contributes ``num_prefix_tokens`` rows, and its RoPE table
        rows are cos=1/sin=0 so the prefix stays untouched). Returns the same
        probed depths as ``forward``, as packed (hidden_size, total_P) patch
        activations; the caller splits per image with ``torch.split``.
        """
        if any(not 0 <= depth < len(self.layer) for depth in key_depths):
            raise ValueError(
                f"DINOv3 backbone key_depths must index blocks 0..{len(self.layer) - 1}, "
                f"got {key_depths}"
            )
        compute_dtype = self.embeddings.patch_embeddings.weight.dtype
        patches_PK = patches_PK.to(compute_dtype)
        cos_TD = cos_TD.to(compute_dtype)
        sin_TD = sin_TD.to(compute_dtype)
        patch_tokens_PC = self.embeddings.patch_embeddings(patches_PK)
        x_TC = patch_tokens_PC.new_zeros((cos_TD.shape[0], self.hidden_size))
        x_TC[patch_dest_P] = patch_tokens_PC
        prefix_TC = torch.cat(
            (self.embeddings.cls_token[0], self.embeddings.register_tokens[0]),
            dim=0,
        )
        x_TC[prefix_rows_R] = prefix_TC.repeat(
            prefix_rows_R.shape[0] // self.num_prefix_tokens, 1
        )
        cos_T1D = cos_TD.unsqueeze(1)
        sin_T1D = sin_TD.unsqueeze(1)
        block_outputs_TC = []
        for block in self.layer:
            x_TC = block.forward_varlen(
                x_TC, cos_T1D, sin_T1D, cu_seqlens, max_seqlen
            )
            block_outputs_TC.append(x_TC)
        normed_TC = [self.norm(block_outputs_TC[-1])]
        normed_TC += [self.norm(block_outputs_TC[depth]) for depth in key_depths]
        return [
            activation_TC[patch_dest_P].transpose(0, 1)
            for activation_TC in normed_TC
        ]


class _ResidualBlock(nn.Module):
    def __init__(self, function: nn.Module) -> None:
        super().__init__()
        self.fn = function
        self.ratio = 1.0 / (2.0**0.5)

    def forward(self, x_BCL: torch.Tensor) -> torch.Tensor:
        return (self.fn(x_BCL).add(x_BCL)).mul_(self.ratio)


class _BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_BCL: torch.Tensor) -> torch.Tensor:
        input_dtype = x_BCL.dtype
        shape = x_BCL.shape
        x_BCL = x_BCL.float()
        grouped_B1CL = x_BCL.view(shape[0], 1, shape[1], shape[2])
        mean_B11L = grouped_B1CL.mean((1, 3), keepdim=True)
        variance_B11L = grouped_B1CL.var((1, 3), keepdim=True, unbiased=False)
        grouped_B1CL = (grouped_B1CL - mean_B11L) / torch.sqrt(variance_B11L + self.eps)
        grouped_B1CL = (
            grouped_B1CL * self.weight[None, :, None] + self.bias[None, :, None]
        )
        # The statistics accumulate in fp32; cast back so the surrounding
        # conv stack keeps the module's compute dtype.
        return grouped_B1CL.view(shape).to(input_dtype)


def _make_head_block(
    channels: int,
    *,
    kernel_size: int,
    norm_type: str,
    norm_eps: float,
    using_spec_norm: bool,
) -> nn.Module:
    if norm_type == "bn":
        normalization = _BatchNormLocal(channels, norm_eps)
    elif norm_type == "gn":
        normalization = nn.GroupNorm(32, channels, eps=norm_eps, affine=True)
    else:
        raise ValueError(f"Unsupported HF vision discriminator norm type: {norm_type}")
    convolution = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
    )
    if using_spec_norm:
        convolution = nn.utils.spectral_norm(convolution)
    return nn.Sequential(
        convolution,
        normalization,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class RAEFeatureDiscriminator(nn.Module):
    """Trainable heads on top of a frozen visual feature extractor.

    ``backbone_kind='hf'`` evaluates a frozen DINOv3 backbone at native image
    resolution (sides must be divisible by the patch size) with one spectral-
    norm head per probed depth. ``backbone_kind='fixed'`` substitutes the
    deterministic conv pyramid from ``discriminator.py`` for CPU smoke tests.
    Each head emits one logit per patch token: forward returns a (B, H, L)
    tensor for a stacked BCHW batch and a list of (H, L_i) tensors for a
    variable-resolution CHW sequence.
    """

    @dataclass(frozen=True, slots=True)
    class Config:
        feature_channels: int = 64
        num_heads: int = 3
        backbone_kind: str = "fixed"
        hf_model_path: str = ""
        hf_key_depths: tuple[int, ...] = (2, 5, 8, 11)
        hf_kernel_size: int = 9
        hf_norm_type: str = "bn"
        hf_using_spec_norm: bool = True
        hf_norm_eps: float = 1e-6
        backbone_batch_size: int = 8
        backbone_dtype: Literal["float32", "bfloat16"] = "float32"

        def __post_init__(self) -> None:
            if self.feature_channels <= 0:
                raise ValueError("discriminator.feature_channels must be positive")
            if self.num_heads <= 0:
                raise ValueError("discriminator.num_heads must be positive")
            if self.backbone_kind == "hf" and not self.hf_model_path:
                raise ValueError(
                    "discriminator.hf_model_path is required for backbone_kind='hf'"
                )
            if self.hf_kernel_size <= 0 or self.hf_kernel_size % 2 == 0:
                raise ValueError(
                    "discriminator.hf_kernel_size must be positive and odd"
                )
            if self.hf_norm_eps <= 0:
                raise ValueError("discriminator.hf_norm_eps must be positive")
            if self.backbone_batch_size <= 0:
                raise ValueError("discriminator.backbone_batch_size must be positive")
            if self.backbone_dtype not in ("float32", "bfloat16"):
                raise ValueError(
                    "discriminator.backbone_dtype must be 'float32' or 'bfloat16', "
                    f"got {self.backbone_dtype!r}"
                )

    def __init__(
        self,
        config: Config | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        config = config or self.Config()
        device = device or torch.device("cpu")
        self._is_hf_model = config.backbone_kind == "hf"
        if config.backbone_kind == "fixed":
            self.backbone = FrozenImageFeatures(config.feature_channels)
            self.heads = nn.ModuleList(
                [
                    nn.Conv1d(config.feature_channels, 1, 1)
                    for _ in range(config.num_heads)
                ]
            )
            if config.num_heads != len(self.backbone.layers):
                raise ValueError(
                    "RAE discriminator num_heads must equal backbone layers"
                )
        elif config.backbone_kind == "hf":
            model_directory = Path(config.hf_model_path).expanduser()
            if not model_directory.is_dir():
                raise ValueError(
                    f"HF model directory does not exist: {model_directory}"
                )
            checkpoint_path = model_directory / "model.safetensors"
            if not checkpoint_path.is_file():
                raise ValueError(f"DINOv3 checkpoint does not exist: {checkpoint_path}")
            backbone = DINOv3ViTBackbone()
            # Weights load in fp32 (the checkpoint dtype); the bf16 cast
            # happens below together with the heads.
            backbone.load_state_dict(load_file(str(checkpoint_path)), strict=True)
            backbone.to(device=device).eval().requires_grad_(False)
            self.backbone = backbone
            self.num_prefix_tokens = backbone.num_prefix_tokens
            hidden_size = backbone.hidden_size
            num_layers = len(backbone.layer)
            self.key_depths = tuple(
                index for index in config.hf_key_depths if 0 <= index < num_layers
            )
            heads = []
            for _ in range(len(self.key_depths) + 1):
                output = nn.Conv1d(hidden_size, 1, kernel_size=1)
                if config.hf_using_spec_norm:
                    output = nn.utils.spectral_norm(output)
                heads.append(
                    nn.Sequential(
                        _make_head_block(
                            hidden_size,
                            kernel_size=1,
                            norm_type=config.hf_norm_type,
                            norm_eps=config.hf_norm_eps,
                            using_spec_norm=config.hf_using_spec_norm,
                        ),
                        _ResidualBlock(
                            _make_head_block(
                                hidden_size,
                                kernel_size=config.hf_kernel_size,
                                norm_type=config.hf_norm_type,
                                norm_eps=config.hf_norm_eps,
                                using_spec_norm=config.hf_using_spec_norm,
                            )
                        ),
                        output,
                    )
                )
            self.heads = nn.ModuleList(heads)
            # DINOv3 preprocessing is fixed ImageNet normalization, applied by
            # _normalized_group_chunks before the backbone call.
            self.image_mean = (0.485, 0.456, 0.406)
            self.image_std = (0.229, 0.224, 0.225)
            self.patch_size = backbone.patch_size
            self.batch_size = config.backbone_batch_size
            backbone_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[config.backbone_dtype]
            if backbone_dtype != torch.float32:
                # bf16 compute halves backbone forward/backward time; the eager
                # heads (spectral-norm power iteration included) run in bf16
                # too so their inputs match the backbone's activations.
                self.to(dtype=backbone_dtype)
            self._compiled_backbone: (
                Callable[[torch.Tensor], list[torch.Tensor]] | None
            ) = None
            self._compiled_backbone_varlen: (
                Callable[..., list[torch.Tensor]] | None
            ) = None
        else:
            raise ValueError(
                f"Unsupported discriminator backbone: {config.backbone_kind}"
            )
        self.set_head_requires_grad(True)

    def train(self, mode: bool = True) -> "RAEFeatureDiscriminator":
        super().train(mode)
        if self._is_hf_model:
            # Keep the frozen DINOv3 backbone in evaluation mode.
            self.backbone.eval()
        return self

    def compile_forward(self, *, backend: str) -> None:
        if not self._is_hf_model:
            return
        # dynamic=True: resolutions vary between microbatches, so the compiled
        # forward keeps batch/height/width symbolic instead of re-specializing
        # (and recompiling) on every new image shape. Buffer donation is
        # disabled globally: an AOT backward compiled with donated buffers
        # rejects the trainer's retain_graph=True adaptive-weight probes.
        torch._functorch.config.donated_buffer = False
        self._compiled_backbone = torch.compile(
            self._backbone_features, backend=backend, dynamic=True
        )
        self._compiled_backbone_varlen = torch.compile(
            self.backbone.forward_varlen, backend=backend, dynamic=True
        )

    def set_head_requires_grad(self, enabled: bool) -> None:
        self.backbone.requires_grad_(False)
        self.heads.requires_grad_(enabled)

    def _varlen_supported(self) -> bool:
        """Whether the packed varlen backbone path applies (flash-attn dtype)."""
        if not self._is_hf_model:
            return False
        weight = self.backbone.embeddings.patch_embeddings.weight
        return weight.device.type == "cuda" and weight.dtype in (
            torch.float16,
            torch.bfloat16,
        )

    def _pack_varlen(
        self, images: list[torch.Tensor]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        list[int],
    ]:
        """Flatten variable-resolution [0, 1] images into one packed varlen call.

        All indexing metadata derives from CPU-side shape knowledge, so
        building the pack never synchronizes the GPU. Returns the packed
        normalized patches, scatter indices, RoPE tables (identity on prefix
        rows), cu_seqlens, max_seqlen, and per-image patch counts.
        """
        backbone = self.backbone
        device = backbone.embeddings.patch_embeddings.weight.device
        patch = self.patch_size
        prefix = backbone.num_prefix_tokens
        patches: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []
        compute_dtype = backbone.embeddings.patch_embeddings.weight.dtype
        for image_CHW in images:
            height, width = image_CHW.shape[-2:]
            mean_311 = image_CHW.new_tensor(self.image_mean).view(3, 1, 1)
            std_311 = image_CHW.new_tensor(self.image_std).view(3, 1, 1)
            normalized_3HW = (image_CHW - mean_311) / std_311
            patches.append(
                normalized_3HW.view(3, height // patch, patch, width // patch, patch)
                .permute(1, 3, 0, 2, 4)
                .reshape(-1, 3 * patch * patch)
                .to(compute_dtype)
            )
            grids.append((height // patch, width // patch))
        sizes = [height * width for height, width in grids]
        patch_dest: list[int] = []
        prefix_rows: list[int] = []
        cu_seqlens = [0]
        max_seqlen = 0
        offset = 0
        for size in sizes:
            doc_len = size + prefix
            patch_dest.extend(range(offset + prefix, offset + doc_len))
            prefix_rows.extend(range(offset, offset + prefix))
            offset += doc_len
            cu_seqlens.append(offset)
            max_seqlen = max(max_seqlen, doc_len)
        patch_dest_P = torch.tensor(patch_dest, dtype=torch.long, device=device)
        head_dim = backbone.head_dim
        cos_TD = torch.ones(offset, head_dim, dtype=torch.float32, device=device)
        sin_TD = torch.zeros(offset, head_dim, dtype=torch.float32, device=device)
        cos_tables, sin_tables = [], []
        for grid_h, grid_w in grids:
            cos_PD, sin_PD = backbone._rope_cos_sin(grid_h, grid_w, device)
            cos_tables.append(cos_PD)
            sin_tables.append(sin_PD)
        cos_TD[patch_dest_P] = torch.cat(cos_tables, dim=0)
        sin_TD[patch_dest_P] = torch.cat(sin_tables, dim=0)
        return (
            torch.cat(patches, dim=0),
            patch_dest_P,
            torch.tensor(prefix_rows, dtype=torch.long, device=device),
            cos_TD,
            sin_TD,
            torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            max_seqlen,
            sizes,
        )

    def _packed_features(
        self, images: list[torch.Tensor], *, use_compiled: bool
    ) -> list[list[torch.Tensor]]:
        """Per-image per-depth (C, L_i) activations from one packed forward."""
        (
            patches_PK,
            patch_dest_P,
            prefix_rows_R,
            cos_TD,
            sin_TD,
            cu_seqlens,
            max_seqlen,
            sizes,
        ) = self._pack_varlen(images)
        backbone_varlen = (
            self._compiled_backbone_varlen
            if use_compiled and self._compiled_backbone_varlen is not None
            else self.backbone.forward_varlen
        )
        activations_CL = backbone_varlen(
            patches_PK,
            patch_dest_P,
            prefix_rows_R,
            cos_TD,
            sin_TD,
            cu_seqlens,
            max_seqlen,
            key_depths=self.key_depths,
        )
        per_depth = [torch.split(activation_CL, sizes, dim=-1) for activation_CL in activations_CL]
        return [
            [per_depth[depth][index] for depth in range(len(per_depth))]
            for index in range(len(images))
        ]

    def _backbone_features(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        """Frozen-backbone activations (B, C, L) at the probed depths."""
        return self.backbone(images_BCHW, key_depths=self.key_depths)

    @overload
    def _forward_group(
        self, images_BCHW: torch.Tensor, *, return_activations: Literal[False] = False
    ) -> torch.Tensor:
        ...

    @overload
    def _forward_group(
        self, images_BCHW: torch.Tensor, *, return_activations: Literal[True]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        ...

    def _forward_group(
        self, images_BCHW: torch.Tensor, *, return_activations: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Per-patch logits (B, H, L) for one same-resolution batch.

        The heads stay eager even when the backbone is compiled: their
        spectral-norm power iteration mutates the u/v buffers in place on
        every training-mode forward, and an AOTAutograd backward captured
        against those buffers fails its version check once a later chunk's
        forward bumps them.
        """
        if self._compiled_backbone is None:
            activations_BCL = self._backbone_features(images_BCHW)
        else:
            activations_BCL = self._compiled_backbone(images_BCHW)
        logits_BHL = [
            head(activation_BCL)
            for head, activation_BCL in zip(self.heads, activations_BCL, strict=True)
        ]
        logits_BHL = torch.cat(logits_BHL, dim=1)
        if return_activations:
            return logits_BHL, activations_BCL
        return logits_BHL

    def _group_by_shape(
        self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[list[torch.Tensor], dict[tuple[int, int], list[int]], bool]:
        if isinstance(images_BCHW, torch.Tensor):
            if images_BCHW.ndim != 4:
                raise ValueError("HF vision discriminator expects BCHW images")
            images = list(images_BCHW.unbind(0))
            return_stacked = True
        else:
            images = list(images_BCHW)
            return_stacked = False
        if not images:
            raise ValueError("HF vision discriminator requires at least one image")
        groups: dict[tuple[int, int], list[int]] = {}
        for index, image_CHW in enumerate(images):
            if image_CHW.ndim != 3 or image_CHW.shape[0] != 3:
                raise ValueError(
                    "HF vision discriminator expects three-channel CHW images"
                )
            height, width = image_CHW.shape[-2:]
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    "HF vision discriminator image sides must be divisible by the "
                    f"backbone patch size {self.patch_size}, got {(height, width)}"
                )
            groups.setdefault((height, width), []).append(index)
        return images, groups, return_stacked

    def _normalized_group_chunks(
        self,
        images: list[torch.Tensor],
        groups: dict[tuple[int, int], list[int]],
    ) -> Generator[tuple[list[int], torch.Tensor]]:
        for indices in groups.values():
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start : start + self.batch_size]
                group_BCHW = torch.stack([images[index] for index in chunk])
                mean_1C11 = group_BCHW.new_tensor(self.image_mean).view(1, -1, 1, 1)
                std_1C11 = group_BCHW.new_tensor(self.image_std).view(1, 3, 1, 1)
                yield chunk, (group_BCHW - mean_1C11) / std_1C11

    def _forward_hf(
        self,
        images_BCHW: torch.Tensor | Sequence[torch.Tensor],
        *,
        return_features: bool = False,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor | list[torch.Tensor], list[list[torch.Tensor]]]
    ):
        images, groups, return_stacked = self._group_by_shape(images_BCHW)
        if self._varlen_supported():
            # One packed varlen backbone forward for the whole call instead of
            # per-shape-group stacked batches; the eager heads re-group the
            # per-image activations by token count.
            activations = self._packed_features(
                images, use_compiled=self._compiled_backbone_varlen is not None
            )
            logits_items = self._logits_from_features(activations)
            logits = torch.stack(logits_items) if return_stacked else logits_items
            if return_features:
                return logits, activations
            return logits
        outputs: list[torch.Tensor | None] = [None] * len(images)
        feature_outputs: list[list[torch.Tensor] | None] | None = (
            [None] * len(images) if return_features else None
        )
        for chunk, group_BCHW in self._normalized_group_chunks(images, groups):
            if feature_outputs is not None:
                logits_BHL, activations_BCL = self._forward_group(
                    group_BCHW, return_activations=True
                )
                for chunk_index, image_index in enumerate(chunk):
                    outputs[image_index] = logits_BHL[chunk_index]
                    feature_outputs[image_index] = [
                        activation_BCL[chunk_index]
                        for activation_BCL in activations_BCL
                    ]
            else:
                logits_BHL = self._forward_group(group_BCHW)
                for chunk_index, image_index in enumerate(chunk):
                    outputs[image_index] = logits_BHL[chunk_index]
        if any(output is None for output in outputs):
            raise RuntimeError("HF vision discriminator did not produce every output")
        final_outputs = cast(list[torch.Tensor], outputs)
        logits = torch.stack(final_outputs) if return_stacked else final_outputs
        if feature_outputs is None:
            return logits
        final_features = cast(list[list[torch.Tensor]], feature_outputs)
        return logits, final_features

    def features(
        self,
        images_BCHW: torch.Tensor | Sequence[torch.Tensor],
        *,
        use_compiled: bool = False,
    ) -> list[list[torch.Tensor]]:
        """Per-image frozen-backbone activations (C, L_i) at each probed depth.

        Eager by default: this path exists for no-gradient metric logging on
        small subsamples, and bypasses the compiled backbone so no extra
        autograd variant gets compiled mid-run. ``use_compiled`` routes the HF
        backbone through the compiled forward for the step-start real-feature
        cache. HF inputs are [0, 1] images; fixed-kind inputs are [-1, 1].
        """
        if not self._is_hf_model:
            items = (
                list(images_BCHW.unbind(0))
                if isinstance(images_BCHW, torch.Tensor)
                else list(images_BCHW)
            )
            groups: dict[tuple[int, int], list[int]] = {}
            for index, image_CHW in enumerate(items):
                height, width = image_CHW.shape[-2:]
                groups.setdefault((height, width), []).append(index)
            outputs: list[list[torch.Tensor] | None] = [None] * len(items)
            for indices in groups.values():
                group_BCHW = torch.stack([items[index] for index in indices])
                group_features = self.backbone(group_BCHW)
                for group_index, image_index in enumerate(indices):
                    outputs[image_index] = [
                        level_BCL[group_index] for level_BCL in group_features
                    ]
            if any(output is None for output in outputs):
                raise RuntimeError("RAE discriminator did not produce every output")
            return outputs  # type: ignore[return-value]
        images, groups, _ = self._group_by_shape(images_BCHW)
        if self._varlen_supported():
            return self._packed_features(images, use_compiled=use_compiled)
        backbone = (
            self._compiled_backbone
            if use_compiled and self._compiled_backbone is not None
            else self._backbone_features
        )
        outputs = [None] * len(images)
        for chunk, group_BCHW in self._normalized_group_chunks(images, groups):
            group_activations = backbone(group_BCHW)
            for chunk_index, image_index in enumerate(chunk):
                outputs[image_index] = [
                    activation_BCL[chunk_index] for activation_BCL in group_activations
                ]
        if any(output is None for output in outputs):
            raise RuntimeError("HF vision discriminator did not produce every output")
        return outputs  # type: ignore[return-value]

    def cache_real_features(
        self, images_01: Sequence[torch.Tensor]
    ) -> list[list[torch.Tensor]]:
        """Step-start cache of per-image real activations for feature matching.

        Takes [0, 1] images (the trainer passes the step's supervision reals
        after the shared augmentation draws, so the discriminator-phase real
        pass keeps its augmented distribution) and returns per-image per-depth
        activations, computed through the compiled HF backbone when available.
        The trainer reuses the cache for the discriminator-phase real pass, so
        the real backbone forward runs once per step instead of twice.
        """
        if self._is_hf_model:
            return self.features(list(images_01), use_compiled=True)
        return self.features([image * 2.0 - 1.0 for image in images_01])

    def feature_matching(
        self,
        real_features: Sequence[Sequence[torch.Tensor]],
        fake_features: Sequence[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """Per-patch feature-matching loss between paired activations.

        Per image and depth: the mean over channels of |fake - real| (diffs in
        fp32, as in feature_distance) gives a per-patch map, reduced to a
        scalar by the patch mean. The loss averages over depths, then images.
        """
        if len(real_features) != len(fake_features) or not real_features:
            raise ValueError("feature_matching expects paired non-empty lists")
        per_image = [
            torch.stack(
                [
                    (fake_CL.float() - real_CL.float()).abs().mean(dim=0).mean()
                    for fake_CL, real_CL in zip(fake_depths, real_depths, strict=True)
                ]
            ).mean()
            for fake_depths, real_depths in zip(
                fake_features, real_features, strict=True
            )
        ]
        return torch.stack(per_image).mean()

    def _heads_from_feature_batch(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Per-patch logits (B, H, L) from one same-shape activation batch."""
        # Fixed-kind pyramid levels shrink by stride 2; pool each head's
        # per-patch logits to the coarsest grid so the heads stack into one
        # (B, H, L) tensor. For the HF backbone every depth shares L and the
        # pool is the identity.
        min_tokens = min(feature.shape[-1] for feature in features)
        return torch.cat(
            [
                F.adaptive_avg_pool1d(head(feature), min_tokens)
                for head, feature in zip(self.heads, features, strict=True)
            ],
            dim=1,
        )

    def _logits_from_features(
        self, features_per_image: Sequence[Sequence[torch.Tensor]]
    ) -> list[torch.Tensor]:
        """Eager head logits (H, L_i) from cached per-image activations."""
        if not features_per_image:
            raise ValueError("RAE discriminator requires at least one image")
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, feature_depths in enumerate(features_per_image):
            groups.setdefault(
                tuple(depth_CL.shape[-1] for depth_CL in feature_depths), []
            ).append(index)
        outputs: list[torch.Tensor | None] = [None] * len(features_per_image)
        for indices in groups.values():
            num_depths = len(features_per_image[indices[0]])
            stacked_BCL = [
                torch.stack([features_per_image[index][depth] for index in indices])
                for depth in range(num_depths)
            ]
            logits_BHL = self._heads_from_feature_batch(stacked_BCL)
            for group_index, image_index in enumerate(indices):
                outputs[image_index] = logits_BHL[group_index]
        if any(output is None for output in outputs):
            raise RuntimeError("RAE discriminator did not produce every output")
        return outputs  # type: ignore[return-value]

    def feature_distance(
        self,
        real_items: Sequence[torch.Tensor],
        fake_items: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Mean multi-depth backbone feature distance, uncalibrated.

        Logging-only LPIPS-style perceptual metric over paired [-1, 1] CHW
        lists: per image, the mean absolute activation difference at each
        probed depth, averaged over depths, then averaged over images.
        """
        if len(real_items) != len(fake_items) or not real_items:
            raise ValueError("feature_distance expects paired non-empty lists")
        if self._is_hf_model:
            real = [(image + 1.0) * 0.5 for image in real_items]
            fake = [(image + 1.0) * 0.5 for image in fake_items]
            real_features = self.features(real)
            fake_features = self.features(fake)
            # The diff accumulates in fp32 so the logged metric keeps the
            # same precision whether the backbone computes in fp32 or bf16.
            per_image = [
                torch.stack(
                    [
                        (fake_depth.float() - real_depth.float()).abs().mean()
                        for fake_depth, real_depth in zip(
                            fake_depths, real_depths, strict=True
                        )
                    ]
                ).mean()
                for fake_depths, real_depths in zip(
                    fake_features, real_features, strict=True
                )
            ]
            return torch.stack(per_image).mean()
        return grouped_per_image_loss(
            self.backbone.features,
            self.backbone.distance,
            list(real_items),
            list(fake_items),
        ).mean()

    @overload
    def _forward_fixed_batch(
        self, images_BCHW: torch.Tensor, *, return_features: Literal[False] = False
    ) -> torch.Tensor:
        ...

    @overload
    def _forward_fixed_batch(
        self, images_BCHW: torch.Tensor, *, return_features: Literal[True]
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        ...

    def _forward_fixed_batch(
        self, images_BCHW: torch.Tensor, *, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """Per-patch logits (B, H, L) for the fixed feature pyramid."""
        features = self.backbone(images_BCHW)
        logits_BHL = self._heads_from_feature_batch(features)
        if not return_features:
            return logits_BHL
        per_image = [
            [level_BCL[index] for level_BCL in features]
            for index in range(images_BCHW.shape[0])
        ]
        return logits_BHL, per_image

    def forward(
        self,
        images_BCHW: torch.Tensor
        | Sequence[torch.Tensor]
        | Sequence[Sequence[torch.Tensor]],
        *,
        return_features: bool = False,
        from_features: bool = False,
    ) -> (
        torch.Tensor
        | list[torch.Tensor]
        | tuple[torch.Tensor | list[torch.Tensor], list[list[torch.Tensor]]]
    ):
        if from_features:
            # Cached per-image backbone activations (see cache_real_features):
            # only the eager heads run, which is what the discriminator-phase
            # real pass needs once the step's real features are cached.
            return self._logits_from_features(images_BCHW)  # type: ignore[arg-type]
        images_input = cast(torch.Tensor | Sequence[torch.Tensor], images_BCHW)
        if self._is_hf_model:
            if isinstance(images_input, torch.Tensor):
                images = (images_input + 1.0) * 0.5
            else:
                images = [(image + 1.0) * 0.5 for image in images_input]
            return self._forward_hf(images, return_features=return_features)
        if isinstance(images_input, torch.Tensor):
            if images_input.ndim != 4:
                raise ValueError("Fixed RAE discriminator expects BCHW images")
            return self._forward_fixed_batch(
                images_input, return_features=return_features
            )
        image_items = list(images_input)
        if not image_items:
            raise ValueError("RAE discriminator requires at least one image")
        for image_CHW in image_items:
            if image_CHW.ndim != 3 or image_CHW.shape[0] != 3:
                raise ValueError("RAE discriminator expects three-channel CHW images")
        outputs: list[torch.Tensor | None] = [None] * len(image_items)
        feature_outputs: list[list[torch.Tensor] | None] | None = (
            [None] * len(image_items) if return_features else None
        )
        groups: dict[tuple[int, int], list[int]] = {}
        for index, image_CHW in enumerate(image_items):
            height, width = image_CHW.shape[-2:]
            groups.setdefault((height, width), []).append(index)
        for indices in groups.values():
            group_BCHW = torch.stack([image_items[index] for index in indices])
            if return_features:
                logits_BHL, group_features = self._forward_fixed_batch(
                    group_BCHW, return_features=True
                )
            else:
                logits_BHL = self._forward_fixed_batch(group_BCHW)
            for group_index, image_index in enumerate(indices):
                outputs[image_index] = logits_BHL[group_index]
                if feature_outputs is not None:
                    feature_outputs[image_index] = group_features[group_index]
        if any(output is None for output in outputs):
            raise RuntimeError("RAE discriminator did not produce every output")
        if feature_outputs is None:
            return outputs  # type: ignore[return-value]
        return outputs, feature_outputs  # type: ignore[return-value]


__all__ = ["DINOv3ViTBackbone", "RAEFeatureDiscriminator"]
