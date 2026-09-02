from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import SampleProcessor
from torchtitan.components.data.types import DatasetBuildContext


class RAEImageProcessor(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        image_size: int | None = 256
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
        if self.image_size is None:
            return image.clamp(0, 1)
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


class RAEQwenProcessor(SampleProcessor):
    """Qwen3.5 image processor with a packed-media output contract.

    The processor is intentionally separate from ``RAEImageProcessor`` because
    Qwen returns normalized flattened patch vectors and ``image_grid_thw``
    metadata rather than a padded BCHW image. The ``media`` field always keeps
    the original item as a one-item ``(B, T, C, H, W)`` tensor; an image uses
    ``T=1`` and ``fps=0``. A downstream decoder can pass the packed vectors
    directly to the RAE with FA2 varlen attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        model_name: str
        image_key: str = "image"
        video_key: str = "video"
        do_resize: bool = True
        do_rescale: bool = False
        do_normalize: bool = True
        image_size: int | None = None
        min_pixels: int | None = None
        max_pixels: int | None = None
        do_sample_frames: bool = False

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        try:
            from transformers import AutoImageProcessor, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "RAEQwenProcessor requires the optional transformers package"
            ) from error
        model_name = str(Path(config.model_name).expanduser())
        self.processor: Any
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except (OSError, ValueError):
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                local_files_only=True,
            )
        self.image_key = config.image_key
        self.video_key = config.video_key
        self.do_resize = config.do_resize
        self.do_rescale = config.do_rescale
        self.do_normalize = config.do_normalize
        if config.image_size is not None and config.image_size <= 0:
            raise ValueError("RAE Qwen image_size must be positive when provided")
        self.image_size = config.image_size
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        self.do_sample_frames = config.do_sample_frames

    @staticmethod
    def _as_btchw(media: Any) -> torch.Tensor:
        if not isinstance(media, torch.Tensor):
            import numpy as np
            from PIL import Image

            if isinstance(media, (bytes, bytearray)):
                import io

                media = Image.open(io.BytesIO(media)).convert("RGB")
            if hasattr(media, "convert"):
                media = np.array(media.convert("RGB"), copy=True)
            if isinstance(media, (list, tuple)):
                frames = [RAEQwenProcessor._as_btchw(frame)[0] for frame in media]
                return torch.stack(frames, dim=0)
            media = torch.from_numpy(np.asarray(media))
        media = media.float()
        if media.numel() and media.max() > 1:
            media = media / 255.0
        if media.ndim == 3:
            if media.shape[-1] in (1, 3, 4):
                media = media[..., :3].permute(2, 0, 1)
            elif media.shape[0] not in (1, 3, 4):
                raise ValueError("RAE media must be HWC or CHW with three channels")
            else:
                media = media[:3]
            return media.unsqueeze(0)
        if media.ndim == 4:
            if media.shape[1] in (1, 3, 4):
                return media[:, :3]
            if media.shape[-1] in (1, 3, 4):
                return media[..., :3].permute(0, 3, 1, 2)
            raise ValueError("RAE videos must use TCHW or THWC layout")
        if media.ndim == 5:
            if media.shape[0] != 1:
                raise ValueError("RAE processor rows must contain one BTCHW media item")
            if media.shape[2] not in (1, 3, 4):
                raise ValueError("RAE videos must use BTCHW layout")
            return media[0, :, :3]
        raise ValueError("RAE media must have CHW, TCHW, or BTCHW dimensions")

    @staticmethod
    def _center_crop_resize(media_BTCHW: torch.Tensor, image_size: int) -> torch.Tensor:
        height, width = media_BTCHW.shape[-2:]
        scale = image_size / min(height, width)
        resized_height = max(image_size, round(height * scale))
        resized_width = max(image_size, round(width * scale))
        media_BTCHW = F.interpolate(
            media_BTCHW,
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        top = (resized_height - image_size) // 2
        left = (resized_width - image_size) // 2
        return media_BTCHW[..., top : top + image_size, left : left + image_size]

    def __call__(self, sample: dict[str, Any], rng) -> dict[str, Any] | None:
        del rng
        if not isinstance(sample, dict):
            raise ValueError("RAEQwenProcessor expects dictionary samples")
        image = sample.get(self.image_key)
        video = sample.get(self.video_key)
        if image is None and video is None:
            raise KeyError(
                f"RAE sample must contain {self.image_key!r} or {self.video_key!r}"
            )
        kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "do_normalize": self.do_normalize,
        }
        if self.min_pixels is not None:
            kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            kwargs["max_pixels"] = self.max_pixels
        media = image if image is not None else video
        media_btchw = self._as_btchw(media)
        if image is not None and self.image_size is not None:
            media_btchw = self._center_crop_resize(media_btchw, self.image_size)
        if image is not None:
            output = self.processor(images=media_btchw, **kwargs)
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            media_kind = "image"
            fps = 0.0
        else:
            video_tchw = media_btchw
            video_kwargs = dict(kwargs)
            video_kwargs["do_sample_frames"] = self.do_sample_frames
            video_kwargs["return_metadata"] = True
            output = self.processor(videos=video_tchw, **video_kwargs)
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            media_kind = "video"
            sample_fps = sample.get("fps")
            metadata = output.get("video_metadata")
            if sample_fps is None and metadata:
                sample_fps = getattr(metadata[0], "fps", None)
            if sample_fps is None or float(sample_fps) <= 0:
                raise ValueError(
                    "Video FPS must be provided with the sample or processor metadata"
                )
            fps = float(sample_fps)
        if grid_key not in output or pixel_key not in output:
            raise ValueError(
                f"Qwen processor did not return {pixel_key} and {grid_key}"
            )
        grid_thw = output[grid_key]
        pixel_values = output[pixel_key]
        processor_for_media = (
            getattr(self.processor, "image_processor", self.processor)
            if image is not None
            else getattr(self.processor, "video_processor", self.processor)
        )
        return {
            "media": media_btchw.unsqueeze(0),
            "media_kind": media_kind,
            "merge_size": int(getattr(processor_for_media, "merge_size", 1)),
            "temporal_patch_size": int(
                getattr(processor_for_media, "temporal_patch_size", 1)
            ),
            "fps": torch.tensor(fps, dtype=torch.float32),
            "temporal_start": torch.tensor(0.0, dtype=torch.float32),
            "pixel_values": pixel_values if image is not None else None,
            "grid_thw": grid_thw if image is not None else None,
            "pixel_values_videos": pixel_values if video is not None else None,
            "grid_thw_videos": grid_thw if video is not None else None,
        }


class RAEImageCollator(Collator):
    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        batch_size: int = 8
        variable_shapes: bool = False

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        if config.batch_size <= 0:
            raise ValueError("RAE image batch_size must be positive")
        self.batch_size = config.batch_size
        self.variable_shapes = config.variable_shapes

    def num_rows_per_batch(self) -> int:
        return self.batch_size

    def __call__(self, rows: Sequence[torch.Tensor]) -> TrainerBatch:
        rows = list(rows)
        if self.variable_shapes:
            labels = torch.zeros(len(rows), dtype=torch.long)
            return {"input": rows}, labels
        images = torch.stack(rows)
        labels = torch.zeros(images.shape[0], dtype=torch.long)
        return {"input": images}, labels


class RAEQwenCollator(Collator):
    """Pack Qwen processor outputs without padding media tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        batch_size: int = 1
        media_kind: Literal["image", "video"] = "image"

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del context
        if config.batch_size <= 0:
            raise ValueError("RAE Qwen batch_size must be positive")
        self.batch_size = config.batch_size
        self.media_kind = config.media_kind

    def num_rows_per_batch(self) -> int:
        return self.batch_size

    def __call__(self, rows: Sequence[dict[str, Any]]) -> TrainerBatch:
        rows = list(rows)
        pixel_key = (
            "pixel_values" if self.media_kind == "image" else "pixel_values_videos"
        )
        grid_key = "grid_thw" if self.media_kind == "image" else "grid_thw_videos"
        pixel_values = [
            row[pixel_key] for row in rows if row.get(pixel_key) is not None
        ]
        grids = [row[grid_key] for row in rows if row.get(grid_key) is not None]
        if len(pixel_values) != len(rows) or len(grids) != len(rows):
            raise ValueError(
                "Every RAE Qwen row must contain one media tensor and grid"
            )
        packed_pixels = torch.cat(pixel_values, dim=0)
        grid_thw = torch.cat([grid.reshape(-1, 3) for grid in grids], dim=0).to(
            dtype=torch.long
        )
        merge_sizes = {int(row.get("merge_size", 1)) for row in rows}
        if len(merge_sizes) != 1:
            raise ValueError("RAE Qwen rows must use the same spatial merge size")
        merge_size = merge_sizes.pop()
        rae_grid_thw = grid_thw.clone()
        rae_grid_thw[:, 1:] //= merge_size
        sequence_lengths = rae_grid_thw.prod(dim=-1)
        media = [row["media"] for row in rows]
        fps = torch.stack([torch.as_tensor(row.get("fps", 0.0)) for row in rows])
        temporal_start = torch.stack(
            [torch.as_tensor(row.get("temporal_start", 0.0)) for row in rows]
        )
        labels = torch.zeros(len(rows), dtype=torch.long)
        return {
            "input": packed_pixels,
            "grid_thw": grid_thw,
            "rae_grid_thw": rae_grid_thw,
            "sequence_lengths": sequence_lengths,
            "media": media,
            "media_kind": self.media_kind,
            "fps": fps,
            "temporal_start": temporal_start,
        }, labels


__all__ = [
    "RAEImageProcessor",
    "RAEQwenProcessor",
    "RAEImageCollator",
    "RAEQwenCollator",
]
