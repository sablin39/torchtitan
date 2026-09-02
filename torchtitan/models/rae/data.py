from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import SampleProcessor
from torchtitan.components.data.types import DatasetBuildContext


class RAEImageProcessor(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        image_size: int = 256
        image_key: str = "image"

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        self.image_size = config.image_size
        self.image_key = config.image_key

    def __call__(self, sample: Any, rng) -> torch.Tensor:
        del rng
        if isinstance(sample, dict):
            if self.image_key not in sample:
                raise KeyError(
                    f"RAE dataset sample does not contain {self.image_key!r}"
                )
            image = sample[self.image_key]
        else:
            image = sample
        if not isinstance(image, torch.Tensor):
            import io

            import numpy as np
            from PIL import Image

            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            if hasattr(image, "convert"):
                image = image.convert("RGB")
            image = torch.from_numpy(np.array(image, copy=True))
        if image.ndim == 2:
            image = image.unsqueeze(-1).expand(-1, -1, 3)
        if image.ndim != 3:
            raise ValueError("RAE images must have three dimensions")
        if image.shape[-1] in (1, 3, 4):
            image = image[..., :3].permute(2, 0, 1)
        elif image.shape[0] not in (1, 3, 4):
            raise ValueError("RAE images must be HWC or CHW with three channels")
        else:
            image = image[:3]
        image = image.float()
        if image.max() > 1:
            image = image / 255.0
        height, width = image.shape[-2:]
        scale = self.image_size / min(height, width)
        resized_height = max(self.image_size, round(height * scale))
        resized_width = max(self.image_size, round(width * scale))
        image = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        top = (resized_height - self.image_size) // 2
        left = (resized_width - self.image_size) // 2
        image = image[:, top : top + self.image_size, left : left + self.image_size]
        return image.clamp(0, 1)


class RAEImageCollator(Collator):
    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        batch_size: int = 8

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        if config.batch_size <= 0:
            raise ValueError("RAE image batch_size must be positive")
        self.batch_size = config.batch_size

    def num_rows_per_batch(self) -> int:
        return self.batch_size

    def __call__(self, rows: Sequence[torch.Tensor]) -> TrainerBatch:
        images = torch.stack(list(rows))
        labels = torch.zeros(images.shape[0], dtype=torch.long)
        return {"input": images}, labels


__all__ = ["RAEImageProcessor", "RAEImageCollator"]
