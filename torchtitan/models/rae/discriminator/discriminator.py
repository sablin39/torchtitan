# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "RAEPerceptualLoss",
    "gan_discriminator_loss",
    "gan_generator_loss",
    "gan_logits_mean",
    "gan_logits_per_image",
]
