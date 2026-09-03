from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino import HFModelFeatureDiscriminator
from .perceptual import LPIPSPerceptualLoss


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

    def forward(
        self, images_BCHW: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor:
        if self._is_hf_model:
            if isinstance(images_BCHW, torch.Tensor):
                images = (images_BCHW + 1.0) * 0.5
            else:
                images = [(image + 1.0) * 0.5 for image in images_BCHW]
            return self.backbone(images)
        if isinstance(images_BCHW, torch.Tensor):
            if images_BCHW.ndim != 4:
                raise ValueError("Fixed RAE discriminator expects BCHW images")
            image_items = list(images_BCHW.unbind(0))
        else:
            image_items = list(images_BCHW)
        if not image_items:
            raise ValueError("RAE discriminator requires at least one image")
        outputs = []
        for image_CHW in image_items:
            if image_CHW.ndim != 3 or image_CHW.shape[0] != 3:
                raise ValueError("RAE discriminator expects three-channel CHW images")
            features = self.backbone(image_CHW.unsqueeze(0))
            logits = [
                head(feature).mean(dim=-1).squeeze(0)
                for head, feature in zip(self.heads, features, strict=True)
            ]
            outputs.append(torch.cat(logits, dim=0))
        return torch.stack(outputs)


class RAEPerceptualLoss(nn.Module):
    """Frozen feature-distance loss used as the Stage 1 LPIPS substitute."""

    def __init__(
        self,
        kind: str = "fixed",
        channels: int = 64,
        calibration_checkpoint_path: str = "",
        vgg_checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
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

    def forward(self, real_BCHW: torch.Tensor, fake_BCHW: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, FrozenImageFeatures):
            real_features = self.backbone(real_BCHW)
            fake_features = self.backbone(fake_BCHW)
            return torch.stack(
                [
                    F.l1_loss(fake, real)
                    for fake, real in zip(fake_features, real_features)
                ]
            ).mean()
        return self.backbone(real_BCHW, fake_BCHW)


def gan_generator_loss(logits_fake: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "hinge":
        return -logits_fake.mean()
    if loss_type == "vanilla":
        return -logits_fake.mean()
    raise ValueError(f"Unsupported generator GAN loss: {loss_type}")


def gan_discriminator_loss(
    logits_real: torch.Tensor, logits_fake: torch.Tensor, loss_type: str
) -> torch.Tensor:
    if loss_type == "hinge":
        return 0.5 * (
            F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()
        )
    if loss_type == "vanilla":
        return 0.5 * (F.softplus(-logits_real).mean() + F.softplus(logits_fake).mean())
    raise ValueError(f"Unsupported discriminator GAN loss: {loss_type}")


__all__ = [
    "RAEFeatureDiscriminator",
    "RAEPerceptualLoss",
    "gan_generator_loss",
    "gan_discriminator_loss",
]
