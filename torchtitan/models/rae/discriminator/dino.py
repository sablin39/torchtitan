# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Tensor dimensions: B=batch, L=patch tokens, C=channel.

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn as nn


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
        shape = x_BCL.shape
        x_BCL = x_BCL.float()
        grouped_B1CL = x_BCL.view(shape[0], 1, shape[1], shape[2])
        mean_B11L = grouped_B1CL.mean((1, 3), keepdim=True)
        variance_B11L = grouped_B1CL.var((1, 3), keepdim=True, unbiased=False)
        grouped_B1CL = (grouped_B1CL - mean_B11L) / torch.sqrt(variance_B11L + self.eps)
        grouped_B1CL = (
            grouped_B1CL * self.weight[None, :, None] + self.bias[None, :, None]
        )
        return grouped_B1CL.view(shape)


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
    """HF vision feature discriminator with variable-resolution image support."""

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
        input_size: int | None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "HFModelFeatureDiscriminator requires the transformers package"
            ) from error
        model_directory = Path(model_path).expanduser()
        if not model_directory.is_dir():
            raise ValueError(f"HF model directory does not exist: {model_directory}")
        model = AutoModel.from_pretrained(
            str(model_directory),
            local_files_only=True,
        )
        model.to(device=device).eval().requires_grad_(False)
        self.model = model
        model_config = model.config
        self.num_prefix_tokens = int(
            getattr(
                model_config,
                "num_prefix_tokens",
                1 + int(getattr(model_config, "num_register_tokens", 0)),
            )
        )
        hidden_size = int(model_config.hidden_size)
        num_layers = int(getattr(model_config, "num_hidden_layers", 0))
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
        try:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(
                str(model_directory), local_files_only=True
            )
        except (ImportError, OSError, ValueError):
            processor = None
        self.image_mean = tuple(
            float(value)
            for value in getattr(processor, "image_mean", (0.485, 0.456, 0.406))
        )
        self.image_std = tuple(
            float(value)
            for value in getattr(processor, "image_std", (0.229, 0.224, 0.225))
        )
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("HF vision processor statistics must have three values")
        self.model_norm = getattr(self.model, "norm", nn.Identity())
        self.patch_size = int(getattr(model_config, "patch_size", 16))
        if self.patch_size <= 0:
            raise ValueError("HF vision model patch_size must be positive")
        if input_size is None:
            input_size = int(getattr(model_config, "image_size", 224))
        if input_size <= 0 or input_size % self.patch_size != 0:
            raise ValueError(
                "HF vision discriminator input_size must be positive and divisible "
                "by the backbone patch size"
            )
        self.input_size = input_size
        self._compiled_forward_group: (
            Callable[[torch.Tensor], torch.Tensor] | None
        ) = None
        self._compile_backend: str | None = None
        self._compile_warmup_pending = False

    def train(self, mode: bool = True) -> "HFModelFeatureDiscriminator":
        """Keep the frozen Hugging Face backbone in evaluation mode."""
        super().train(mode)
        self.model.eval()
        return self

    def compile_forward(self, *, backend: str) -> None:
        self._compile_backend = backend
        self._compile_warmup_pending = True

    def _get_forward_group(self, sample_BCHW: torch.Tensor) -> Callable:
        """Initialize compiled execution after Transformers installs its hooks."""
        if self.training:
            return self._forward_group
        if self._compile_warmup_pending:
            with torch.no_grad():
                self._forward_group(sample_BCHW.detach())
            assert self._compile_backend is not None
            self._compiled_forward_group = torch.compile(
                self._forward_group,
                backend=self._compile_backend,
                fullgraph=True,
            )
            self._compile_warmup_pending = False
        return self._compiled_forward_group or self._forward_group

    def set_head_requires_grad(self, enabled: bool) -> None:
        self.model.requires_grad_(False)
        self.heads.requires_grad_(enabled)

    def _forward_group(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=images_BCHW,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("HF vision model did not return hidden states")
        activations_BCL = [
            outputs.last_hidden_state,
            *[self.model_norm(hidden_states[index + 1]) for index in self.key_depths],
        ]
        activations_BCL = [
            activation_BLC[:, self.num_prefix_tokens :].transpose(1, 2)
            for activation_BLC in activations_BCL
        ]
        logits_B1 = [
            head(activation_BCL).mean(dim=-1)
            for head, activation_BCL in zip(self.heads, activations_BCL)
        ]
        return torch.cat(logits_B1, dim=1)

    def forward_fixed(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        """Evaluate a fixed BCHW batch without runtime grouping or resizing."""
        if images_BCHW.ndim != 4 or images_BCHW.shape[1] != 3:
            raise ValueError(
                "HF vision discriminator fixed path expects BCHW RGB images"
            )
        if images_BCHW.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(
                "HF vision discriminator fixed path expects input_size x input_size"
            )
        mean_1C11 = images_BCHW.new_tensor(self.image_mean).view(1, -1, 1, 1)
        std_1C11 = images_BCHW.new_tensor(self.image_std).view(1, 3, 1, 1)
        return self._forward_group(((images_BCHW + 1.0) * 0.5 - mean_1C11) / std_1C11)

    def _resize_for_backbone(self, image_CHW: torch.Tensor) -> torch.Tensor:
        height, width = image_CHW.shape[-2:]
        if (height, width) == (self.input_size, self.input_size):
            return image_CHW
        scale = min(self.input_size / height, self.input_size / width)
        target_height = max(
            self.patch_size,
            min(
                self.input_size,
                int(height * scale) // self.patch_size * self.patch_size,
            ),
        )
        target_width = max(
            self.patch_size,
            min(
                self.input_size,
                int(width * scale) // self.patch_size * self.patch_size,
            ),
        )
        resized = torch.nn.functional.interpolate(
            image_CHW.unsqueeze(0),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        canvas = image_CHW.new_full(
            (image_CHW.shape[0], self.input_size, self.input_size), 0.5
        )
        top = (self.input_size - target_height) // 2
        left = (self.input_size - target_width) // 2
        canvas[:, top : top + target_height, left : left + target_width] = resized
        return canvas

    def forward(
        self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(images_BCHW, torch.Tensor):
            if images_BCHW.ndim != 4:
                raise ValueError("HF vision discriminator expects BCHW images")
            images = [
                images_BCHW[index : index + 1] for index in range(images_BCHW.shape[0])
            ]
        else:
            images = list(images_BCHW)
        if not images:
            raise ValueError("HF vision discriminator requires at least one image")
        outputs: list[torch.Tensor | None] = [None] * len(images)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, image_CHW in enumerate(images):
            if image_CHW.ndim == 4:
                if image_CHW.shape[0] != 1:
                    raise ValueError("HF image sequences must contain CHW tensors")
                image_CHW = image_CHW[0]
            if image_CHW.ndim != 3 or image_CHW.shape[0] != 3:
                raise ValueError(
                    "HF vision discriminator expects three-channel CHW images"
                )
            image_CHW = self._resize_for_backbone(image_CHW)
            groups.setdefault(tuple(image_CHW.shape[-2:]), []).append(index)
            images[index] = image_CHW
        for indices in groups.values():
            group_BCHW = torch.stack([images[index] for index in indices])
            mean_1C11 = group_BCHW.new_tensor(self.image_mean).view(1, -1, 1, 1)
            std_1C11 = group_BCHW.new_tensor(self.image_std).view(1, 3, 1, 1)
            forward_group = self._get_forward_group(group_BCHW)
            group_logits_BH = forward_group(
                ((group_BCHW + 1.0) * 0.5 - mean_1C11) / std_1C11
            )
            for group_index, image_index in enumerate(indices):
                outputs[image_index] = group_logits_BH[group_index]
        if any(output is None for output in outputs):
            raise RuntimeError("HF vision discriminator did not produce every output")
        return torch.stack(outputs)


__all__ = ["HFModelFeatureDiscriminator"]
