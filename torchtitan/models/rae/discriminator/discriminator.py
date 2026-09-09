# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import HFModelFeatureDiscriminator
from .perceptual import grouped_per_image_loss, LPIPSPerceptualLoss

# Discriminator logits are per patch token: a (B, H, L) tensor for a stacked
# batch, or a list of (H, L_i) tensors for variable-resolution images
# (B=batch, H=discriminator heads, L=patch tokens).
Logits = torch.Tensor | Sequence[torch.Tensor]


class FrozenImageFeatures(nn.Module):
    """Small fixed feature pyramid used by the Stage 1 discriminator.

    The module intentionally has no trainable normalization statistics. This
    makes ``eval()`` and distributed replication deterministic while preserving
    the RAEv2 contract of a frozen visual backbone with trainable heads.
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(3, channels, 3, stride=2, padding=1),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            ]
        )
        generator = torch.Generator(device="cpu").manual_seed(0)
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(
                    torch.randn(
                        layer.weight.shape,
                        generator=generator,
                    )
                    / (layer.in_channels * layer.kernel_size[0] ** 2) ** 0.5
                )
                layer.bias.zero_()
        self.requires_grad_(False)

    def forward(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        features = []
        hidden_BCHW = images_BCHW
        for layer in self.layers:
            hidden_BCHW = F.gelu(layer(hidden_BCHW))
            features.append(hidden_BCHW.flatten(2))
        return features

    def features(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        return self.forward(images_BCHW)

    def distance(
        self,
        input_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Per-image mean absolute feature distance for one batch."""
        return torch.stack(
            [
                (fake - real).abs().flatten(1).mean(dim=1)
                for fake, real in zip(input_features, target_features)
            ]
        ).mean(dim=0)


class RAEFeatureDiscriminator(nn.Module):
    """Trainable heads on top of a frozen visual feature extractor."""

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
        self._is_hf_model = False
        if config.backbone_kind == "fixed":
            self.backbone = FrozenImageFeatures(config.feature_channels)
        elif config.backbone_kind == "hf":
            self.backbone = HFModelFeatureDiscriminator(
                model_path=config.hf_model_path,
                device=device,
                key_depths=config.hf_key_depths,
                kernel_size=config.hf_kernel_size,
                norm_type=config.hf_norm_type,
                using_spec_norm=config.hf_using_spec_norm,
                norm_eps=config.hf_norm_eps,
                batch_size=config.backbone_batch_size,
                backbone_dtype={
                    "float32": torch.float32,
                    "bfloat16": torch.bfloat16,
                }[config.backbone_dtype],
            )
            self._is_hf_model = True
        else:
            raise ValueError(
                f"Unsupported discriminator backbone: {config.backbone_kind}"
            )
        if not self._is_hf_model:
            self._heads = nn.ModuleList(
                [
                    nn.Conv1d(config.feature_channels, 1, 1)
                    for _ in range(config.num_heads)
                ]
            )
        else:
            self._heads = None
        if config.backbone_kind == "fixed" and config.num_heads != len(
            self.backbone.layers
        ):
            raise ValueError("RAE discriminator num_heads must equal backbone layers")
        self.set_head_requires_grad(True)

    @property
    def heads(self) -> nn.Module:
        if self._is_hf_model:
            return self.backbone.heads
        assert self._heads is not None
        return self._heads

    def set_head_requires_grad(self, enabled: bool) -> None:
        if self._is_hf_model:
            self.backbone.set_head_requires_grad(enabled)
        else:
            self.backbone.requires_grad_(False)
            self.heads.requires_grad_(enabled)

    def train(self, mode: bool = True) -> "RAEFeatureDiscriminator":
        super().train(mode)
        if self._is_hf_model:
            self.backbone.model.eval()
        return self

    def compile_forward(self, *, backend: str) -> None:
        if self._is_hf_model:
            self.backbone.compile_forward(backend=backend)

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
            real_features = self.backbone.features(real)
            fake_features = self.backbone.features(fake)
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

    def _forward_fixed_batch(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        """Per-patch logits (B, H, L) for the fixed feature pyramid."""
        features = self.backbone(images_BCHW)
        # Pyramid levels shrink by stride 2; pool each head's per-patch logits
        # to the coarsest grid so the heads stack into one (B, H, L) tensor.
        min_tokens = min(feature.shape[-1] for feature in features)
        return torch.cat(
            [
                F.adaptive_avg_pool1d(head(feature), min_tokens)
                for head, feature in zip(self.heads, features, strict=True)
            ],
            dim=1,
        )

    def forward(self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]) -> Logits:
        if self._is_hf_model:
            if isinstance(images_BCHW, torch.Tensor):
                images = (images_BCHW + 1.0) * 0.5
            else:
                images = [(image + 1.0) * 0.5 for image in images_BCHW]
            return self.backbone(images)
        if isinstance(images_BCHW, torch.Tensor):
            if images_BCHW.ndim != 4:
                raise ValueError("Fixed RAE discriminator expects BCHW images")
            return self._forward_fixed_batch(images_BCHW)
        image_items = list(images_BCHW)
        if not image_items:
            raise ValueError("RAE discriminator requires at least one image")
        for image_CHW in image_items:
            if image_CHW.ndim != 3 or image_CHW.shape[0] != 3:
                raise ValueError("RAE discriminator expects three-channel CHW images")
        outputs: list[torch.Tensor | None] = [None] * len(image_items)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, image_CHW in enumerate(image_items):
            groups.setdefault(tuple(image_CHW.shape[-2:]), []).append(index)
        for indices in groups.values():
            logits_BHL = self._forward_fixed_batch(
                torch.stack([image_items[index] for index in indices])
            )
            for group_index, image_index in enumerate(indices):
                outputs[image_index] = logits_BHL[group_index]
        if any(output is None for output in outputs):
            raise RuntimeError("RAE discriminator did not produce every output")
        return outputs  # type: ignore[return-value]


class RAEPerceptualLoss(nn.Module):
    """Frozen feature-distance loss used as the Stage 1 LPIPS substitute.

    ``resize_long_side`` bounds the long image side before feature extraction.
    VGG/LPIPS is calibrated around 224-256px inputs, so running it at the
    native (often 700px+) reconstruction resolution both leaves the calibrated
    regime and costs FLOPs proportional to H*W for no benefit. Images at or
    below the bound pass through unchanged; 0 disables resizing.
    """

    def __init__(
        self,
        kind: str = "fixed",
        channels: int = 64,
        calibration_checkpoint_path: str = "",
        vgg_checkpoint_path: str | None = None,
        resize_long_side: int = 256,
    ) -> None:
        super().__init__()
        if resize_long_side < 0:
            raise ValueError("perceptual resize_long_side must be non-negative")
        self.resize_long_side = resize_long_side
        if kind == "fixed":
            self.backbone = FrozenImageFeatures(channels)
        elif kind == "lpips":
            self.backbone = LPIPSPerceptualLoss(
                calibration_checkpoint_path=calibration_checkpoint_path,
                vgg_checkpoint_path=vgg_checkpoint_path,
            )
        else:
            raise ValueError(f"Unsupported perceptual loss kind: {kind}")
        self.backbone.eval()
        self.requires_grad_(False)

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        """Downscale (..., H, W) images whose long side exceeds the bound."""
        if self.resize_long_side == 0:
            return images
        height, width = images.shape[-2:]
        long_side = max(height, width)
        if long_side <= self.resize_long_side:
            return images
        scale = self.resize_long_side / long_side
        new_hw = (max(1, round(height * scale)), max(1, round(width * scale)))
        return F.interpolate(images, size=new_hw, mode="bilinear", antialias=True)

    def forward(self, real_BCHW: torch.Tensor, fake_BCHW: torch.Tensor) -> torch.Tensor:
        return self.forward_per_sample(real_BCHW, fake_BCHW).mean()

    def forward_per_sample(
        self, real_BCHW: torch.Tensor, fake_BCHW: torch.Tensor
    ) -> torch.Tensor:
        # The real branch never needs gradients; computing it under no_grad
        # halves the backward work and activation memory.
        with torch.no_grad():
            real_features = self.backbone.features(self._resize(real_BCHW))
        return self.backbone.distance(
            self.backbone.features(self._resize(fake_BCHW)), real_features
        )

    def forward_per_sample_list(
        self,
        real_items: Sequence[torch.Tensor],
        fake_items: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Per-image losses for variable-resolution CHW lists.

        Equal-shape images are stacked into grouped backbone forwards instead
        of one call per image.
        """
        return grouped_per_image_loss(
            lambda images: self.backbone.features(self._resize(images)),
            self.backbone.distance,
            real_items,
            fake_items,
        )


def _image_weighted_mean(
    logits: Logits, per_patch_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Mean of per-patch penalties, weighted per image rather than per token.

    Averaging within each image first keeps the loss scale independent of the
    (variable) patch-token count of every image.
    """
    if isinstance(logits, torch.Tensor):
        return per_patch_fn(logits).flatten(1).mean(dim=1).mean()
    return torch.stack(
        [per_patch_fn(image_logits).mean() for image_logits in logits]
    ).mean()


def gan_generator_loss(logits_fake: Logits, loss_type: str) -> torch.Tensor:
    if loss_type == "hinge":
        return _image_weighted_mean(logits_fake, lambda logits: -logits)
    if loss_type == "vanilla":
        # Non-saturating BCE generator loss: -log D(fake). Unlike the hinge
        # form, its per-patch logit gradient (sigmoid(logit) - 1) vanishes
        # once the generator is winning, instead of pushing without bound.
        return _image_weighted_mean(logits_fake, lambda logits: F.softplus(-logits))
    raise ValueError(f"Unsupported generator GAN loss: {loss_type}")


def gan_discriminator_loss(
    logits_real: Logits, logits_fake: Logits, loss_type: str
) -> torch.Tensor:
    if loss_type == "hinge":
        return 0.5 * (
            _image_weighted_mean(logits_real, lambda logits: F.relu(1.0 - logits))
            + _image_weighted_mean(logits_fake, lambda logits: F.relu(1.0 + logits))
        )
    if loss_type == "vanilla":
        return 0.5 * (
            _image_weighted_mean(logits_real, lambda logits: F.softplus(-logits))
            + _image_weighted_mean(logits_fake, F.softplus)
        )
    raise ValueError(f"Unsupported discriminator GAN loss: {loss_type}")


def gan_logits_per_image(logits: Logits) -> torch.Tensor:
    """Per-image mean logit (over heads and patch tokens), shape (B,)."""
    if isinstance(logits, torch.Tensor):
        return logits.flatten(1).mean(dim=1)
    return torch.stack([image_logits.mean() for image_logits in logits])


def gan_logits_mean(logits: Logits) -> torch.Tensor:
    return gan_logits_per_image(logits).mean()


__all__ = [
    "RAEFeatureDiscriminator",
    "RAEPerceptualLoss",
    "gan_discriminator_loss",
    "gan_generator_loss",
    "gan_logits_mean",
    "gan_logits_per_image",
]
