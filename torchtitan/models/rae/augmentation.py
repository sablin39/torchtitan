from __future__ import annotations

# Tensor dimensions: B=batch, C=channel, H=height, W=width.

import torch
import torch.nn.functional as F


class DiscriminatorAugmentation:
    """Differentiable translation, color, and cutout augmentation."""

    def __init__(self, probability: float = 1.0, cutout: float = 0.0) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if not 0.0 <= cutout <= 1.0:
            raise ValueError("augmentation cutout must be in [0, 1]")
        self.probability = probability
        self.cutout = cutout
        self._grids: dict[
            tuple[int, int, int, torch.device], tuple[torch.Tensor, ...]
        ] = {}

    def _get_grids(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        key = (batch_size, height, width, device)
        if key not in self._grids:
            self._grids[key] = torch.meshgrid(
                torch.arange(batch_size, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self._grids[key]

    def __call__(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        if images_BCHW.dtype != torch.float32:
            images_BCHW = images_BCHW.float()
        if self.probability < 1e-6:
            return images_BCHW

        translate, color, cutout = (torch.rand(3) <= self.probability).tolist()
        if not translate and not color and not cutout:
            return images_BCHW
        batch_size, _, height, width = images_BCHW.shape
        random_B = torch.rand(7, batch_size, 1, 1, device=images_BCHW.device)

        if translate:
            height_delta = round(height * 0.125)
            width_delta = round(width * 0.125)
            height_offset_B11 = (
                random_B[0].mul(2 * height_delta + 1).floor().long() - height_delta
            )
            width_offset_B11 = (
                random_B[1].mul(2 * width_delta + 1).floor().long() - width_delta
            )
            batch_grid_BHW, height_grid_BHW, width_grid_BHW = self._get_grids(
                batch_size, height, width, images_BCHW.device
            )
            height_grid_BHW = (
                (height_grid_BHW + height_offset_B11).add(1).clamp(0, height + 1)
            )
            width_grid_BHW = (
                (width_grid_BHW + width_offset_B11).add(1).clamp(0, width + 1)
            )
            padded_BCHW = F.pad(images_BCHW, (1, 1, 1, 1))
            images_BCHW = padded_BCHW.permute(0, 2, 3, 1)[
                batch_grid_BHW,
                height_grid_BHW,
                width_grid_BHW,
            ].permute(0, 3, 1, 2)

        if color:
            images_BCHW = images_BCHW + random_B[2].unsqueeze(-1) - 0.5
            channel_mean_B1HW = images_BCHW.mean(dim=1, keepdim=True)
            images_BCHW = (images_BCHW - channel_mean_B1HW) * random_B[3].unsqueeze(
                -1
            ).mul(2) + channel_mean_B1HW
            image_mean_B111 = images_BCHW.mean((1, 2, 3), keepdim=True)
            images_BCHW = (images_BCHW - image_mean_B111) * random_B[4].unsqueeze(
                -1
            ).add(0.5) + image_mean_B111

        if cutout and self.cutout > 0:
            cutout_height = round(height * self.cutout)
            cutout_width = round(width * self.cutout)
            height_offset_B11 = (
                random_B[5].mul(height + (1 - cutout_height % 2)).floor().long()
            )
            width_offset_B11 = (
                random_B[6].mul(width + (1 - cutout_width % 2)).floor().long()
            )
            batch_grid_BHW, height_grid_BHW, width_grid_BHW = self._get_grids(
                batch_size,
                cutout_height,
                cutout_width,
                images_BCHW.device,
            )
            height_grid_BHW = (
                (height_grid_BHW + height_offset_B11)
                .sub(cutout_height // 2)
                .clamp(0, height - 1)
            )
            width_grid_BHW = (
                (width_grid_BHW + width_offset_B11)
                .sub(cutout_width // 2)
                .clamp(0, width - 1)
            )
            mask_BHW = torch.ones(
                batch_size,
                height,
                width,
                dtype=images_BCHW.dtype,
                device=images_BCHW.device,
            )
            mask_BHW[batch_grid_BHW, height_grid_BHW, width_grid_BHW] = 0
            images_BCHW = images_BCHW * mask_BHW.unsqueeze(1)

        return images_BCHW.contiguous()


__all__ = ["DiscriminatorAugmentation"]
