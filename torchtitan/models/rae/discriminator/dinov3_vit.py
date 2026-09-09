# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan-native DINOv3 ViT-B/16 backbone for the RAE discriminator.

Reimplements the Hugging Face ``DINOv3ViTModel`` forward pass (transformers
``models/dinov3_vit``) without the HF dependency, loading the same
safetensors checkpoint (keys ``embeddings.*``, ``layer.N.*``, ``norm.*``)
with ``load_state_dict(..., strict=True)``. The backbone is a frozen
feature extractor: it always runs in fp32 eval mode and accepts a variable
input resolution per call (sides divisible by the patch size).

Tensor dimension legend (letters are scoped to this file):
    B = batch, C = channels (3 for the image, hidden_size for tokens),
    H = image height, W = image width, L = tokens (prefix + patches),
    P = patch tokens, N = attention heads, D = head dim,
    K = flattened patch pixels (C*ph*pw).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


__all__ = ["DINOv3ViTBackbone"]
