# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Tensor dimensions: B=batch, L=patch tokens, C=channel, H=discriminator heads.

from collections.abc import Callable, Generator, Sequence
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .dinov3_vit import DINOv3ViTBackbone


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


class HFModelFeatureDiscriminator(nn.Module):
    """DINOv3 vision feature discriminator evaluated at native image resolution.

    Images pass through the frozen backbone unresized (sides must be
    divisible by the backbone patch size; the backbone's RoPE generalizes to
    any resolution), so the discriminator scores the reconstruction at the
    decoder's output resolution. Each head emits one logit per patch token:
    forward returns a (B, H, L) tensor for a stacked BCHW batch and a list of
    (H, L_i) tensors for a variable-resolution CHW sequence.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: torch.device,
        key_depths: tuple[int, ...],
        kernel_size: int,
        norm_type: str,
        using_spec_norm: bool,
        norm_eps: float,
        batch_size: int = 64,
        backbone_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        model_directory = Path(model_path).expanduser()
        if not model_directory.is_dir():
            raise ValueError(f"HF model directory does not exist: {model_directory}")
        checkpoint_path = model_directory / "model.safetensors"
        if not checkpoint_path.is_file():
            raise ValueError(f"DINOv3 checkpoint does not exist: {checkpoint_path}")
        model = DINOv3ViTBackbone()
        # Weights load in fp32 (the checkpoint dtype); the bf16 cast happens
        # below together with the heads.
        model.load_state_dict(load_file(str(checkpoint_path)), strict=True)
        model.to(device=device).eval().requires_grad_(False)
        self.model = model
        self.num_prefix_tokens = model.num_prefix_tokens
        hidden_size = model.hidden_size
        num_layers = len(model.layer)
        self.key_depths = tuple(
            index for index in key_depths if 0 <= index < num_layers
        )
        heads = []
        for _ in range(len(self.key_depths) + 1):
            output = nn.Conv1d(hidden_size, 1, kernel_size=1)
            if using_spec_norm:
                output = nn.utils.spectral_norm(output)
            heads.append(
                nn.Sequential(
                    _make_head_block(
                        hidden_size,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    _ResidualBlock(
                        _make_head_block(
                            hidden_size,
                            kernel_size=kernel_size,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
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
        self.patch_size = model.patch_size
        if batch_size <= 0:
            raise ValueError("HF vision discriminator batch_size must be positive")
        self.batch_size = batch_size
        if backbone_dtype != torch.float32:
            # bf16 compute halves backbone forward/backward time; the eager
            # heads (spectral-norm power iteration included) run in bf16 too
            # so their inputs match the backbone's activations.
            self.to(dtype=backbone_dtype)
        self._compiled_backbone: (
            Callable[[torch.Tensor], list[torch.Tensor]] | None
        ) = None

    def train(self, mode: bool = True) -> "HFModelFeatureDiscriminator":
        """Keep the frozen DINOv3 backbone in evaluation mode."""
        super().train(mode)
        self.model.eval()
        return self

    def compile_forward(self, *, backend: str) -> None:
        # dynamic=True: resolutions vary between microbatches, so the compiled
        # forward keeps batch/height/width symbolic instead of re-specializing
        # (and recompiling) on every new image shape. Buffer donation is
        # disabled globally: an AOT backward compiled with donated buffers
        # rejects the trainer's retain_graph=True adaptive-weight probes.
        torch._functorch.config.donated_buffer = False
        self._compiled_backbone = torch.compile(
            self._backbone_features, backend=backend, dynamic=True
        )

    def set_head_requires_grad(self, enabled: bool) -> None:
        self.model.requires_grad_(False)
        self.heads.requires_grad_(enabled)

    def _backbone_features(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        """Frozen-backbone activations (B, C, L) at the probed depths."""
        return self.model(images_BCHW, key_depths=self.key_depths)

    def _forward_group(self, images_BCHW: torch.Tensor) -> torch.Tensor:
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
        return torch.cat(logits_BHL, dim=1)

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

    def forward(
        self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:
        images, groups, return_stacked = self._group_by_shape(images_BCHW)
        outputs: list[torch.Tensor | None] = [None] * len(images)
        for chunk, group_BCHW in self._normalized_group_chunks(images, groups):
            logits_BHL = self._forward_group(group_BCHW)
            for chunk_index, image_index in enumerate(chunk):
                outputs[image_index] = logits_BHL[chunk_index]
        if any(output is None for output in outputs):
            raise RuntimeError("HF vision discriminator did not produce every output")
        if return_stacked:
            return torch.stack(outputs)  # type: ignore[arg-type]
        return outputs  # type: ignore[return-value]

    def features(
        self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]
    ) -> list[list[torch.Tensor]]:
        """Per-image frozen-backbone activations (C, L_i) at each probed depth.

        Eager by design: this path exists for no-gradient metric logging on
        small subsamples, and bypasses the compiled backbone so no extra
        autograd variant gets compiled mid-run.
        """
        images, groups, _ = self._group_by_shape(images_BCHW)
        outputs: list[list[torch.Tensor] | None] = [None] * len(images)
        for chunk, group_BCHW in self._normalized_group_chunks(images, groups):
            group_activations = self._backbone_features(group_BCHW)
            for chunk_index, image_index in enumerate(chunk):
                outputs[image_index] = [
                    activation_BCL[chunk_index] for activation_BCL in group_activations
                ]
        if any(output is None for output in outputs):
            raise RuntimeError("HF vision discriminator did not produce every output")
        return outputs  # type: ignore[return-value]


__all__ = ["HFModelFeatureDiscriminator"]
