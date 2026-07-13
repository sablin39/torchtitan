# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Canonical RWKV-VL processor implementation for training and HF export.

This module intentionally has no TorchTitan-specific imports so it can be
copied into HF remote-code exports and used by ``AutoProcessor`` at inference
time.  The key policy is that ``max_pixels`` is a per-sample visual budget:
when one prompt contains multiple images, the resized pixel counts are scaled
together so their sum stays within the configured budget whenever physically
possible.
"""

from __future__ import annotations

import copy

import logging
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import einops as E
import requests
import torch

import torchvision.io

import torchvision.transforms.v2.functional as TVF

from PIL import Image
from transformers import BaseImageProcessor, PreTrainedTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)

try:
    from .tokenizer import (
        CHAT_TEMPLATE,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_VISION_END_TOKEN,
        DEFAULT_VISION_START_TOKEN,
    )
except ImportError:
    from torchtitan.models.rwkv7.tokenizer import (
        CHAT_TEMPLATE,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_VISION_END_TOKEN,
        DEFAULT_VISION_START_TOKEN,
    )


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RWKVVLImageProcessorConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 2097152
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    max_aspect_ratio: float = 50.0

    @property
    def factor(self) -> int:
        return self.patch_size * self.spatial_merge_size


@dataclass(frozen=True, slots=True)
class RWKVVLProcessedImages:
    images: list[torch.Tensor]
    image_token_counts: list[int]
    grid_thw: torch.Tensor
    flat_patches: torch.Tensor


def flatten_images(images: Any) -> list[Any]:
    if images is None:
        return []
    if not isinstance(images, (list, tuple)):
        return [images]

    flat_images = []
    for item in images:
        if isinstance(item, (list, tuple)):
            flat_images.extend(flatten_images(item))
        else:
            flat_images.append(item)
    return flat_images


def get_images_per_text_sample(
    images: Any,
    batch_size: int,
) -> list[list[Any]] | None:
    if images is None:
        return [[] for _ in range(batch_size)]
    if batch_size == 1:
        return [flatten_images(images)]
    if isinstance(images, (list, tuple)) and len(images) == batch_size:
        return [flatten_images(sample_images) for sample_images in images]
    return None


def normalize_image_tags(
    text: str,
    *,
    user_image_tag: str,
    vision_image_token: str,
) -> str:
    return text.replace(user_image_tag, vision_image_token)


def strip_excess_image_tags(
    text: str,
    *,
    user_image_tag: str,
    num_allowed: int,
) -> str:
    count = text.count(user_image_tag)
    if count <= num_allowed:
        return text
    parts = text.split(user_image_tag)
    kept = user_image_tag.join(parts[: num_allowed + 1])
    rest = "".join(parts[num_allowed + 1 :])
    return kept + rest


def append_missing_image_tags(
    text: str,
    *,
    vision_image_token: str,
    num_missing_images: int,
) -> str:
    if num_missing_images <= 0:
        return text
    return text + vision_image_token * num_missing_images


def insert_vision_placeholders(
    input_parts: list[str | None],
    num_vision_tokens: list[int],
    *,
    vision_start_token: str,
    vision_token: str,
    vision_end_token: str,
    eos_token: str = "",
) -> str:
    output_parts: list[str] = []
    vision_index = 0

    for part in input_parts:
        if part is None and vision_index < len(num_vision_tokens):
            output_parts.extend(
                [
                    vision_start_token,
                    *([vision_token] * num_vision_tokens[vision_index]),
                    vision_end_token,
                ]
            )
            vision_index += 1
        else:
            output_parts.append(part)  # pyrefly: ignore [bad-argument-type]

    result = "".join(output_parts).strip()
    if eos_token and not result.endswith(eos_token):
        result += eos_token
    return result


def count_token_occurrences(
    input_ids: list[list[int]],
    token_id: int,
) -> list[int]:
    return [
        sum(1 for token in sample_ids if token == token_id) for sample_ids in input_ids
    ]


def validate_image_token_alignment(
    input_ids: list[list[int]],
    *,
    expected_image_tokens: list[int],
    expected_num_images: list[int],
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> None:
    actual_image_tokens = count_token_occurrences(input_ids, image_token_id)
    actual_vision_starts = count_token_occurrences(input_ids, vision_start_token_id)
    actual_vision_ends = count_token_occurrences(input_ids, vision_end_token_id)

    if actual_image_tokens != expected_image_tokens:
        raise ValueError(
            "Image token count does not match image_grid_thw-derived token count: "
            f"expected {expected_image_tokens}, got {actual_image_tokens}."
        )
    if (
        actual_vision_starts != expected_num_images
        or actual_vision_ends != expected_num_images
    ):
        raise ValueError(
            "Vision boundary token count does not match the number of image placeholders: "
            f"expected {expected_num_images}, got starts={actual_vision_starts}, "
            f"ends={actual_vision_ends}."
        )


def _decode_image(image: str | bytes | Image.Image) -> torch.Tensor:
    """Decode an image to a ``(C, H, W)`` uint8 RGB tensor."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = image["bytes"]
        elif image.get("path") is not None:
            image = image["path"]
        else:
            raise ValueError("Image dict must contain 'bytes' or 'path'.")
    if isinstance(image, str) and image.startswith("http"):
        response = requests.get(image, timeout=10)
        response.raise_for_status()
        image = response.content
    if isinstance(image, bytes):
        raw = torch.frombuffer(bytearray(image), dtype=torch.uint8)
        return torchvision.io.decode_image(raw, mode=torchvision.io.ImageReadMode.RGB)
    if isinstance(image, str):
        return torchvision.io.decode_image(image, mode=torchvision.io.ImageReadMode.RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return TVF.pil_to_tensor(image)


def normalize_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


def get_image_size(image: str | bytes | Image.Image) -> tuple[int, int]:
    """Return ``(height, width)`` without decoding to a tensor when possible."""
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if isinstance(image, bytes):
        with Image.open(BytesIO(image)) as pil_image:
            width, height = pil_image.size
        return height, width
    if isinstance(image, str) and image.startswith("http"):
        response = requests.get(image, timeout=10)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as pil_image:
            width, height = pil_image.size
        return height, width
    if isinstance(image, str):
        with Image.open(image) as pil_image:
            width, height = pil_image.size
        return height, width

    img_tensor = _decode_image(image)
    _, height, width = img_tensor.shape
    return height, width


def _ensure_min_factor_size(
    height: int,
    width: int,
    factor: int,
) -> tuple[int, int]:
    if height < factor or width < factor:
        scale = max(factor / width, factor / height)
        width = int(width * scale)
        height = int(height * scale)
    return height, width


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Compute target ``(height, width)`` with Qwen-style resize constraints."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"Absolute aspect ratio must be smaller than 200, "
            f"got {max(height, width) / min(height, width):.1f}"
        )

    h_bar = max(round(height / factor) * factor, factor)
    w_bar = max(round(width / factor) * factor, factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(math.floor(height / beta / factor) * factor, factor)
        w_bar = max(math.floor(width / beta / factor) * factor, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def resize_sizes_to_total_pixels(
    sizes: list[tuple[int, int]],
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> list[tuple[int, int]]:
    """Resize image sizes with ``max_pixels`` as a shared per-sample budget."""
    if not sizes:
        return []

    resized_sizes = [
        smart_resize(
            *_ensure_min_factor_size(height, width, factor),
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        for height, width in sizes
    ]
    if max_pixels <= 0:
        return resized_sizes

    total_pixels = sum(height * width for height, width in resized_sizes)
    if total_pixels <= max_pixels:
        return resized_sizes

    scale = math.sqrt(max_pixels / total_pixels)
    min_area = factor * factor

    def scale_sizes(scale_factor: float) -> list[tuple[int, int]]:
        scaled = []
        for height, width in resized_sizes:
            scaled_height = max(
                factor, math.floor(height * scale_factor / factor) * factor
            )
            scaled_width = max(
                factor, math.floor(width * scale_factor / factor) * factor
            )
            scaled.append((scaled_height, scaled_width))
        return scaled

    scaled_sizes = scale_sizes(scale)
    for _ in range(8):
        scaled_total = sum(height * width for height, width in scaled_sizes)
        if scaled_total <= max_pixels:
            return scaled_sizes
        if scaled_total <= len(scaled_sizes) * min_area:
            break
        scale *= math.sqrt(max_pixels / scaled_total) * 0.995
        scaled_sizes = scale_sizes(scale)

    scaled_total = sum(height * width for height, width in scaled_sizes)
    if scaled_total > max_pixels:
        logger.warning(
            "Minimum resized image area (%s images * %s pixels) exceeds "
            "max_pixels=%s; using the smallest factor-aligned image sizes.",
            len(scaled_sizes),
            min_area,
            max_pixels,
        )
    return scaled_sizes


def process_image(
    image: str | bytes | Image.Image,
    patch_size: int = 16,
    merge_size: int = 2,
    max_pixels: int = 16777216,
    min_pixels: int = 65536,
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
    resized_size: tuple[int, int] | None = None,
) -> torch.Tensor | None:
    """Decode, resize, rescale, normalize, and return ``(1, H, W, C)``."""
    try:
        img_tensor = _decode_image(image)
        _, original_height, original_width = img_tensor.shape
        factor = patch_size * merge_size

        if resized_size is None:
            original_height, original_width = _ensure_min_factor_size(
                original_height, original_width, factor
            )
            resized_height, resized_width = smart_resize(
                original_height,
                original_width,
                factor=factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            resized_height, resized_width = resized_size

        img_tensor = TVF.resize(
            img_tensor,
            [resized_height, resized_width],
            interpolation=TVF.InterpolationMode.BICUBIC,
            antialias=True,
        )
        img_tensor = TVF.to_dtype(img_tensor, torch.float32, scale=True)
        img_tensor = TVF.normalize(
            img_tensor, list(image_mean), list(image_std), inplace=True
        )
        return img_tensor.permute(1, 2, 0).unsqueeze(0)

    except Exception as exc:
        logger.warning("Error processing image: %s", exc)
        return None


def calculate_vision_tokens(
    num_frames: int,
    height: int,
    width: int,
    patch_size: int,
    spatial_merge_size: int,
    temporal_patch_size: int,
) -> tuple[int, int, int]:
    t_patches = math.ceil(num_frames / temporal_patch_size)
    tokens_per_row = width // (patch_size * spatial_merge_size)
    num_rows = height // (patch_size * spatial_merge_size)
    total_tokens = t_patches * tokens_per_row * num_rows
    return total_tokens, tokens_per_row, num_rows


def vision_to_patches(
    img: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(T, H, W, C)`` to Qwen-compatible flattened patches."""
    T, H, W, C = img.shape
    ps = patch_size
    ts = temporal_patch_size

    if T % ts != 0:
        pad_t = ts - (T % ts)
        img = torch.cat([img, img[-1:].expand(pad_t, -1, -1, -1)], dim=0)
        T = img.shape[0]

    T_patches = T // ts
    H_patches = H // ps
    W_patches = W // ps

    patches = E.rearrange(
        img,
        "(t pt) (bh m ph) (bw n pw) c -> (t bh bw m n) (c pt ph pw)",
        pt=ts,
        ph=ps,
        pw=ps,
        m=merge_size,
        n=merge_size,
    )
    grid_thw = torch.tensor([T_patches, H_patches, W_patches])
    return patches, grid_thw


def process_images(
    images: list[Any],
    config: RWKVVLImageProcessorConfig,
) -> RWKVVLProcessedImages:
    """Process a list of images that share one visual pixel budget."""
    normalized_images = [normalize_image(image) for image in images]
    image_sizes = []
    for image in normalized_images:
        height, width = get_image_size(image)
        if width == 0 or height == 0:
            raise ValueError("Image has zero width or height")
        ratio = max(width / height, height / width)
        if ratio > config.max_aspect_ratio:
            raise ValueError(
                f"Image aspect ratio {ratio:.1f} exceeds {config.max_aspect_ratio}"
            )
        image_sizes.append((height, width))

    resized_sizes = resize_sizes_to_total_pixels(
        image_sizes,
        factor=config.factor,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )

    processed_images = []
    image_token_counts = []
    patch_list = []
    grid_list = []
    for image, resized_size in zip(normalized_images, resized_sizes, strict=True):
        processed = process_image(
            image,
            patch_size=config.patch_size,
            merge_size=config.spatial_merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
            resized_size=resized_size,
        )
        if processed is None:
            raise ValueError("Could not process image")
        num_tokens, _, _ = calculate_vision_tokens(
            num_frames=processed.shape[0],
            height=processed.shape[1],
            width=processed.shape[2],
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
        )
        patches, grid = vision_to_patches(
            processed,
            config.patch_size,
            config.temporal_patch_size,
            config.spatial_merge_size,
        )
        processed_images.append(processed)
        image_token_counts.append(num_tokens)
        patch_list.append(patches)
        grid_list.append(grid)

    if patch_list:
        flat_patches = torch.cat(patch_list, dim=0)
        grid_thw = torch.stack(grid_list, dim=0)
    else:
        patch_dim = 3 * config.temporal_patch_size * config.patch_size**2
        flat_patches = torch.empty(0, patch_dim, dtype=torch.float32)
        grid_thw = torch.empty(0, 3, dtype=torch.long)

    return RWKVVLProcessedImages(
        images=processed_images,
        image_token_counts=image_token_counts,
        grid_thw=grid_thw,
        flat_patches=flat_patches,
    )


def make_image_config_from_processor(
    image_processor: Any,
    *,
    max_aspect_ratio: float = 50.0,
    **overrides: Any,
) -> RWKVVLImageProcessorConfig:
    size = getattr(image_processor, "size", {}) or {}
    if hasattr(size, "to_dict"):
        size = size.to_dict()
    override_size = overrides.pop("size", None) or {}
    if hasattr(override_size, "to_dict"):
        override_size = override_size.to_dict()
    patch_size = int(
        overrides.pop("patch_size", None) or getattr(image_processor, "patch_size", 16)
    )
    temporal_patch_size = int(
        overrides.pop("temporal_patch_size", None)
        or getattr(image_processor, "temporal_patch_size", 2)
    )
    merge_size = int(
        overrides.pop("merge_size", None)
        or overrides.pop("spatial_merge_size", None)
        or getattr(image_processor, "merge_size", 2)
    )
    min_pixels = int(
        overrides.pop("min_pixels", None)
        or overrides.pop("shortest_edge", None)
        or override_size.get("shortest_edge")
        or size.get("shortest_edge")
        or 65536
    )
    max_pixels = int(
        overrides.pop("max_pixels", None)
        or overrides.pop("longest_edge", None)
        or override_size.get("longest_edge")
        or size.get("longest_edge")
        or 2097152
    )
    image_mean = tuple(
        overrides.pop("image_mean", None)
        or getattr(image_processor, "image_mean", (0.5, 0.5, 0.5))
    )
    image_std = tuple(
        overrides.pop("image_std", None)
        or getattr(image_processor, "image_std", (0.5, 0.5, 0.5))
    )
    return RWKVVLImageProcessorConfig(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        image_mean=image_mean,
        image_std=image_std,
        max_aspect_ratio=float(overrides.pop("max_aspect_ratio", max_aspect_ratio)),
    )


class ModRWKVProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
        },
        "images_kwargs": {},
    }


class ModRWKVProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "RwkvTokenizer"
    user_image_tag = "<image>"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        image_processor: BaseImageProcessor = None,
        chat_template=None,
        auto_insert_image_tags: bool = True,
        total_pixels_budget: bool = True,
    ):
        chat_template = CHAT_TEMPLATE if chat_template is None else chat_template
        super().__init__(
            tokenizer=tokenizer,
            image_processor=image_processor,
            chat_template=chat_template,
        )
        self.auto_insert_image_tags = auto_insert_image_tags
        self.total_pixels_budget = total_pixels_budget
        self.image_token = getattr(tokenizer, "image_token", DEFAULT_IMAGE_TOKEN)
        self.vision_start_token = getattr(
            tokenizer, "vision_start_token", DEFAULT_VISION_START_TOKEN
        )
        self.vision_end_token = getattr(
            tokenizer, "vision_end_token", DEFAULT_VISION_END_TOKEN
        )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids(
            self.vision_start_token
        )
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.vision_end_token
        )
        self.vision_image_token = (
            f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
        )

    def to_dict(self):
        output = {}
        if self.image_processor is not None:
            output["image_processor"] = self.image_processor.to_dict()
        auto_map = getattr(self, "auto_map", None)
        if auto_map is not None:
            output["auto_map"] = copy.deepcopy(auto_map)
        output["processor_class"] = self.__class__.__name__
        if not self.auto_insert_image_tags:
            output["auto_insert_image_tags"] = False
        output["total_pixels_budget"] = self.total_pixels_budget
        return output

    def _process_images(self, images, batch_size, images_kwargs):
        image_groups = get_images_per_text_sample(images, batch_size)
        if image_groups is None:
            image_groups = [flatten_images(images)]
            num_images_per_sample = None
        else:
            num_images_per_sample = [len(group) for group in image_groups]

        image_config = make_image_config_from_processor(
            self.image_processor,
            **images_kwargs,
        )
        processed_groups = [
            process_images(group, image_config) for group in image_groups
        ]
        num_image_tokens = [
            count
            for processed in processed_groups
            for count in processed.image_token_counts
        ]
        if not num_image_tokens:
            return {}, None, None, num_images_per_sample

        pixel_values = torch.cat(
            [
                processed.flat_patches
                for processed in processed_groups
                if processed.flat_patches.numel() > 0
            ],
            dim=0,
        )
        image_grid_thw = torch.cat(
            [
                processed.grid_thw
                for processed in processed_groups
                if processed.grid_thw.numel() > 0
            ],
            dim=0,
        )
        return (
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            image_grid_thw,
            num_image_tokens,
            num_images_per_sample,
        )

    def _get_num_multimodal_tokens(self, image_grid_thw=None, **kwargs):
        vision_data = {}
        if image_grid_thw is not None:
            processor_defaults = getattr(self.image_processor, "_defaults", {})
            images_kwargs = dict(processor_defaults.get("images_kwargs", {}))
            images_kwargs.update(kwargs)
            merge_size = (
                images_kwargs.get("merge_size", None) or self.image_processor.merge_size
            )

            num_image_patches = [
                int(grid[0] * grid[1] * grid[2]) for grid in image_grid_thw
            ]
            num_image_tokens = [
                num_patches // merge_size**2 for num_patches in num_image_patches
            ]
            vision_data.update(
                {
                    "num_image_tokens": num_image_tokens,
                    "num_image_patches": num_image_patches,
                }
            )
        return MultiModalData(**vision_data)

    def __call__(
        self, images=None, text=None, **kwargs: Unpack[ModRWKVProcessorKwargs]
    ):
        output_kwargs = self._merge_kwargs(
            ModRWKVProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if not isinstance(text, list):
            text = [text] if text is not None else None

        batch_size = len(text) if text is not None else 1
        if images is not None:
            (
                image_inputs,
                image_grid_thw,
                num_image_tokens,
                num_images_per_sample,
            ) = self._process_images(
                images,
                batch_size,
                output_kwargs["images_kwargs"],
            )
        else:
            image_inputs = {}
            image_grid_thw = None
            num_image_tokens = None
            num_images_per_sample = None

        if text is None:
            return BatchFeature(data=image_inputs)

        text = text.copy()
        expected_image_tokens = [0 for _ in text]
        expected_num_images = [0 for _ in text]
        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                if not self.auto_insert_image_tags:
                    text[i] = text[i].replace(self.user_image_tag, " ")
                else:
                    if num_images_per_sample is not None:
                        text[i] = strip_excess_image_tags(
                            text[i],
                            user_image_tag=self.user_image_tag,
                            num_allowed=num_images_per_sample[i],
                        )
                    text[i] = normalize_image_tags(
                        text[i],
                        user_image_tag=self.user_image_tag,
                        vision_image_token=self.vision_image_token,
                    )

                if self.auto_insert_image_tags and num_images_per_sample is not None:
                    missing = num_images_per_sample[i] - text[i].count(self.image_token)
                    text[i] = append_missing_image_tags(
                        text[i],
                        vision_image_token=self.vision_image_token,
                        num_missing_images=missing,
                    )

                if self.auto_insert_image_tags:
                    placeholder_count = text[i].count(self.vision_image_token)
                    if index + placeholder_count > len(num_image_tokens):
                        raise ValueError(
                            "Number of image placeholders in text exceeds provided images: "
                            f"consumed {index + placeholder_count}, available {len(num_image_tokens)}."
                        )
                    sample_counts = num_image_tokens[index : index + placeholder_count]
                    text[i] = self.tokenizer.expand_image_placeholders(
                        text[i],
                        sample_counts,
                    )
                    expected_image_tokens[i] += sum(sample_counts)
                    expected_num_images[i] += len(sample_counts)
                    index += placeholder_count
                else:
                    while self.image_token in text[i]:
                        if index >= len(num_image_tokens):
                            raise ValueError(
                                "Number of image placeholders in text exceeds provided images: "
                                f"consumed {index + 1}, available {len(num_image_tokens)}."
                            )
                        text[i] = text[i].replace(
                            self.image_token,
                            "<|placeholder|>" * num_image_tokens[index],
                            1,
                        )
                        expected_image_tokens[i] += num_image_tokens[index]
                        expected_num_images[i] += 1
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            if self.auto_insert_image_tags and index != len(num_image_tokens):
                raise ValueError(
                    "Number of image placeholders in text does not match provided images: "
                    f"consumed {index}, available {len(num_image_tokens)}."
                )
        else:
            for i in range(len(text)):
                text[i] = text[i].replace(self.user_image_tag, "")

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if image_grid_thw is not None:
            validate_image_token_alignment(
                text_inputs["input_ids"],
                expected_image_tokens=expected_image_tokens,
                expected_num_images=expected_num_images,
                image_token_id=self.image_token_id,
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
            )
        self._check_special_mm_tokens(text, text_inputs, modalities=["image"])
        return BatchFeature(
            data={**text_inputs, **image_inputs}, tensor_type=return_tensors
        )

    def apply_chat_template(self, conversation, chat_template=None, **kwargs):
        kwargs.setdefault("return_dict", True)
        return super().apply_chat_template(
            conversation,
            chat_template=chat_template,
            **kwargs,
        )


ModRWKVProcessor.register_for_auto_class("AutoProcessor")
