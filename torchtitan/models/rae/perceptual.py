from __future__ import annotations

# Tensor dimensions: B=batch, C=channel, H=height, W=width.

from pathlib import Path

import torch
import torch.nn as nn


class _ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "shift", torch.tensor((-0.030, -0.088, -0.188))[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.tensor((0.458, 0.448, 0.450))[None, :, None, None]
        )

    def forward(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        return (images_BCHW - self.shift) / self.scale


class _VGG16FeatureExtractor(nn.Module):
    def __init__(self, *, vgg_checkpoint_path: str | None) -> None:
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
        except ImportError as error:
            raise RuntimeError(
                "RAE LPIPS requires torchvision; install a version matching PyTorch"
            ) from error

        if vgg_checkpoint_path is None:
            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        else:
            checkpoint = Path(vgg_checkpoint_path)
            if not checkpoint.is_file():
                raise ValueError(f"VGG16 checkpoint does not exist: {checkpoint}")
            model = vgg16(weights=None)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            features = model.features
        self.slices = nn.ModuleList(
            (
                nn.Sequential(*features[:4]),
                nn.Sequential(*features[4:9]),
                nn.Sequential(*features[9:16]),
                nn.Sequential(*features[16:23]),
                nn.Sequential(*features[23:30]),
            )
        )

    def forward(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        features = []
        hidden_BCHW = images_BCHW
        for layer_slice in self.slices:
            hidden_BCHW = layer_slice(hidden_BCHW)
            features.append(hidden_BCHW)
        return features


class LPIPSPerceptualLoss(nn.Module):
    """Pretrained VGG16 LPIPS loss used by RAE Stage 1."""

    def __init__(
        self,
        *,
        calibration_checkpoint_path: str,
        vgg_checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        calibration_checkpoint = Path(calibration_checkpoint_path)
        if not calibration_checkpoint.is_file():
            raise ValueError(
                "LPIPS calibration checkpoint does not exist: "
                f"{calibration_checkpoint}"
            )
        self.scaling_layer = _ScalingLayer()
        self.net = _VGG16FeatureExtractor(vgg_checkpoint_path=vgg_checkpoint_path)
        channels = (64, 128, 256, 512, 512)
        self.linear_layers = nn.ModuleList(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False) for channel in channels
        )
        calibration_state = torch.load(
            calibration_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        calibration_state = {
            name.replace("lin", "linear_layers.").replace(".model.1", ""): value
            for name, value in calibration_state.items()
        }
        missing, unexpected = self.load_state_dict(calibration_state, strict=False)
        missing = [
            name
            for name in missing
            if not name.startswith("net.") and not name.startswith("scaling_layer.")
        ]
        if missing:
            raise RuntimeError(f"LPIPS checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"LPIPS checkpoint has unexpected keys: {unexpected}")
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _normalize(features_BCHW: torch.Tensor) -> torch.Tensor:
        norm_B1HW = torch.sqrt(torch.sum(features_BCHW.square(), dim=1, keepdim=True))
        return features_BCHW / (norm_B1HW + 1e-10)

    def forward(
        self,
        input_BCHW: torch.Tensor,
        target_BCHW: torch.Tensor,
    ) -> torch.Tensor:
        input_features = self.net(self.scaling_layer(input_BCHW))
        target_features = self.net(self.scaling_layer(target_BCHW))
        distances = []
        for input_BCHW, target_BCHW, calibration in zip(
            input_features,
            target_features,
            self.linear_layers,
        ):
            difference_BCHW = (
                self._normalize(input_BCHW) - self._normalize(target_BCHW)
            ).square()
            distances.append(calibration(difference_BCHW).mean((2, 3), keepdim=True))
        return torch.stack(distances).sum(dim=0).mean()


__all__ = ["LPIPSPerceptualLoss"]
