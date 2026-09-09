# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.components.data.collators import Collator, TrainerBatch
from torchtitan.components.data.dataset import SampleProcessor
from torchtitan.components.data.types import DatasetBuildContext


class RAEQwenProcessor(SampleProcessor):
    """Qwen3.5 image processor with a packed-patch output contract.

    Qwen returns normalized flattened patch vectors and ``image_grid_thw``
    metadata rather than a padded BCHW image. The ``media`` field keeps the
    original image as a one-item ``(1, 1, C, H, W)`` tensor with ``fps=0``.
    The decoder consumes the packed vectors directly with FA2 varlen attention.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        model_name: str
        image_key: str = "image"
        do_resize: bool = True
        do_rescale: bool = False
        do_normalize: bool = True
        min_pixels: int | None = None
        max_pixels: int | None = None
        output_dtype: str = "bfloat16"
        """Host-side dtype for pixel_values and media. bf16 halves the host
        payload and device transfer; the frozen encoder consumes bf16 natively
        and supervision targets meet bf16 reconstructions."""

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
        self.do_resize = config.do_resize
        self.do_rescale = config.do_rescale
        self.do_normalize = config.do_normalize
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        if config.output_dtype not in {"bfloat16", "float32"}:
            raise ValueError(
                f"Unsupported RAE Qwen output_dtype: {config.output_dtype}"
            )
        self.output_dtype = (
            torch.bfloat16 if config.output_dtype == "bfloat16" else torch.float32
        )

    @staticmethod
    def _as_btchw(media: Any) -> torch.Tensor:
        """Normalize one image row to a single-frame (1, C, H, W) tensor."""
        if not isinstance(media, torch.Tensor):
            import numpy as np
            from PIL import Image

            if isinstance(media, dict):
                # HF ``Image(decode=False)`` rows carry the encoded file, or
                # just its path for filesystem-backed datasets.
                if media["bytes"] is not None:
                    media = media["bytes"]
                else:
                    with open(media["path"], "rb") as media_file:
                        media = media_file.read()
            if isinstance(media, (bytes, bytearray)):
                import io

                media = Image.open(io.BytesIO(media)).convert("RGB")
            if hasattr(media, "convert"):
                media = np.array(media.convert("RGB"), copy=True)
            media = torch.from_numpy(np.asarray(media))
        media = media.float()
        if media.numel() and media.max() > 1:
            media = media / 255.0
        if media.ndim != 3:
            raise ValueError("RAE processor images must have CHW or HWC dimensions")
        if media.shape[-1] in (1, 3, 4):
            media = media[..., :3].permute(2, 0, 1)
        elif media.shape[0] not in (1, 3, 4):
            raise ValueError("RAE media must be HWC or CHW with three channels")
        else:
            media = media[:3]
        return media.unsqueeze(0)

    def __call__(self, sample: dict[str, Any], rng) -> dict[str, Any] | None:
        del rng
        if not isinstance(sample, dict):
            raise ValueError("RAEQwenProcessor expects dictionary samples")
        image = sample.get(self.image_key)
        if image is None:
            raise KeyError(f"RAE sample must contain {self.image_key!r}")
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
        media_btchw = self._as_btchw(image)
        output = self.processor(images=media_btchw, **kwargs)
        if "image_grid_thw" not in output or "pixel_values" not in output:
            raise ValueError(
                "Qwen processor did not return pixel_values and image_grid_thw"
            )
        grid_thw = output["image_grid_thw"]
        pixel_values = output["pixel_values"].to(self.output_dtype)
        image_processor = getattr(self.processor, "image_processor", self.processor)
        # Bound the host-side media payload: supervision targets are the
        # decoder outputs at half the processed resolution, so the
        # original full-resolution image is never needed. Keep at most the
        # size the vision tower actually saw (grid * patch_size).
        patch_size = int(getattr(image_processor, "patch_size", 1))
        grid_row = grid_thw.reshape(-1, 3)[0]
        processed_size = (
            int(grid_row[1]) * patch_size,
            int(grid_row[2]) * patch_size,
        )
        media_height, media_width = (int(value) for value in media_btchw.shape[-2:])
        if media_height * media_width > processed_size[0] * processed_size[1]:
            media_btchw = F.interpolate(
                media_btchw,
                size=processed_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return {
            "media": media_btchw.to(self.output_dtype).unsqueeze(0),
            "merge_size": int(getattr(image_processor, "merge_size", 1)),
            "fps": torch.tensor(0.0, dtype=torch.float32),
            "temporal_start": torch.tensor(0.0, dtype=torch.float32),
            "pixel_values": pixel_values,
            "grid_thw": grid_thw,
        }


def _pin_for_h2d(value: Any) -> Any:
    """Pin CPU tensors so the trainer's non_blocking H2D copies are async.

    The collator runs on Grain's prefetch thread, so the pin copy stays off
    the trainer's critical path.
    """
    if (
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and torch.cuda.is_available()
    ):
        return value.pin_memory()
    return value


class RAEQwenCollator(Collator):
    """Pack Qwen processor outputs without padding media tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        batch_size: int | None = 1
        token_budget: int | None = None
        max_tokens_per_item: int | None = None

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        if config.batch_size is not None and config.batch_size <= 0:
            raise ValueError("RAE Qwen batch_size must be positive")
        if config.token_budget is not None and config.token_budget <= 0:
            raise ValueError("RAE Qwen token_budget must be positive")
        if config.max_tokens_per_item is not None and config.max_tokens_per_item <= 0:
            raise ValueError("RAE Qwen max_tokens_per_item must be positive")
        if config.batch_size is None and config.max_tokens_per_item is None:
            raise ValueError(
                "RAE Qwen requires batch_size or max_tokens_per_item when token packing"
            )
        context_token_budget = getattr(context, "num_tokens_per_batch", None)
        self.token_budget = config.token_budget or context_token_budget or 0
        self.max_tokens_per_item = config.max_tokens_per_item
        self._enforce_token_budget = (
            config.token_budget is not None or config.max_tokens_per_item is not None
        )
        if config.batch_size is None:
            if self.max_tokens_per_item is None or self.token_budget <= 0:
                raise ValueError(
                    "RAE token-budget batching requires a positive token_budget "
                    "and max_tokens_per_item"
                )
            self._num_rows_per_batch_is_budget_derived = True
            self._num_rows_per_batch = self.token_budget // self.max_tokens_per_item
            if self._num_rows_per_batch <= 0:
                raise ValueError("RAE Qwen token_budget must fit max_tokens_per_item")
        else:
            self._num_rows_per_batch_is_budget_derived = False
            self._num_rows_per_batch = config.batch_size

    def num_rows_per_batch(self) -> int:
        return self._num_rows_per_batch

    def packing_token_budget(self) -> int | None:
        """Budget for token-based batching; fixed row counts return None."""
        if self._num_rows_per_batch_is_budget_derived and self.token_budget > 0:
            return self.token_budget
        return None

    def row_cost(self, row: dict[str, Any]) -> int:
        """Post-merger token count of one processed image row."""
        grid = row.get("grid_thw")
        if grid is None:
            raise ValueError("RAE Qwen rows must contain 'grid_thw'")
        grid = torch.as_tensor(grid).reshape(-1, 3)
        merge_size = int(row.get("merge_size", 1))
        return int((grid.prod(dim=-1) // merge_size**2).sum().item())

    def __call__(self, rows: Sequence[dict[str, Any]]) -> TrainerBatch:
        rows = list(rows)
        pixel_values = [
            row["pixel_values"] for row in rows if row.get("pixel_values") is not None
        ]
        grids = [row["grid_thw"] for row in rows if row.get("grid_thw") is not None]
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
        if self.max_tokens_per_item is not None and torch.any(
            sequence_lengths > self.max_tokens_per_item
        ):
            raise ValueError(
                "RAE Qwen row exceeds max_tokens_per_item; increase the token "
                "ceiling or lower the processor max_pixels"
            )
        packed_tokens = int(sequence_lengths.sum().item())
        if self._enforce_token_budget and packed_tokens > self.token_budget:
            raise ValueError(
                "RAE Qwen packed batch exceeds token_budget; lower max_pixels "
                "or increase token_budget"
            )
        media = [row["media"] for row in rows]
        fps = torch.stack([torch.as_tensor(row.get("fps", 0.0)) for row in rows])
        temporal_start = torch.stack(
            [torch.as_tensor(row.get("temporal_start", 0.0)) for row in rows]
        )
        labels = torch.zeros(len(rows), dtype=torch.long)
        return {
            "input": _pin_for_h2d(packed_pixels),
            "grid_thw": _pin_for_h2d(grid_thw),
            "rae_grid_thw": _pin_for_h2d(rae_grid_thw),
            "sequence_lengths": _pin_for_h2d(sequence_lengths),
            "media": [_pin_for_h2d(item) for item in media],
            "fps": _pin_for_h2d(fps),
            "temporal_start": _pin_for_h2d(temporal_start),
        }, labels


__all__ = [
    "RAEQwenProcessor",
    "RAEQwenCollator",
]
