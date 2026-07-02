# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared RWKV-VL image processing helpers.

This module intentionally has no TorchTitan-specific imports so it can be
copied into HF remote-code exports and used by ``AutoProcessor`` at inference
time.  The key policy is that ``max_pixels`` is a per-sample visual budget:
when one prompt contains multiple images, the resized pixel counts are scaled
together so their sum stays within the configured budget whenever physically
possible.
"""

from __future__ import annotations

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


logger = logging.getLogger(__name__)

CHAT_TEMPLATE = r"""
{%- macro render_tools_block(tools) -%}
{{ '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' }}
{{ 'You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' }}
{%- for tool in tools -%}
{{ tool | tojson }}{%- if not loop.last -%}{{ '\n' }}{%- endif -%}
{%- endfor -%}
{{ '\n</tools>\n\nFor each function call, return a json object with function name and arguments ' }}
{{ 'within <tool_call></tool_call> XML tags:\n<tool_call>\n' }}
{{ '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>' }}
{%- endmacro -%}
{%- macro render_content(content) -%}
{%- if content is string -%}
{{ content }}
{%- elif content is none -%}
{%- else -%}
{%- set ns = namespace(explicit_image_tags=0, image_items=0, text_parts=[]) -%}
{%- for item in content -%}
{%- if item['type'] == 'text' -%}
{%- set text = item.get('text', '') -%}
{%- set ns.text_parts = ns.text_parts + [text] -%}
{%- set ns.explicit_image_tags = ns.explicit_image_tags + text.count('<image>') -%}
{%- elif item['type'] == 'image' or item['type'] == 'image_url' -%}
{%- set ns.image_items = ns.image_items + 1 -%}
{%- endif -%}
{%- endfor -%}
{%- for _ in range([ns.image_items - ns.explicit_image_tags, 0] | max) -%}
{{ '<image>' }}
{%- endfor -%}
{{ ns.text_parts | join('') }}
{%- endif -%}
{%- endmacro -%}
{%- macro render_tool_call(tool_call) -%}
{%- set fn = tool_call.get('function', tool_call) -%}
{%- set call = {'name': fn.get('name', ''), 'arguments': fn.get('arguments', {})} -%}
{{ '<tool_call>\n' }}{{ call | tojson }}{{ '\n</tool_call>' }}
{%- endmacro -%}
{%- macro render_tool_response(message) -%}
{{ '<tool_response>\n' }}
{%- if message['content'] is string -%}
{{ message['content'] }}
{%- elif message['content'] is none -%}
{%- else -%}
{{ message['content'] | tojson }}
{%- endif -%}
{{ '\n</tool_response>' }}
{%- endmacro -%}
{%- set ns = namespace(tool_prompt_exists=false) -%}
{%- if messages|length > 0 and messages[0]['role'] == 'system' and messages[0]['content'] is string -%}
{%- set first_system_content = messages[0]['content'] -%}
{%- if '# Tools' in first_system_content and '<tools>' in first_system_content and '<tool_call>' in first_system_content -%}
{%- set ns.tool_prompt_exists = true -%}
{%- endif -%}
{%- endif -%}
{%- set has_tools = tools is defined and tools and not ns.tool_prompt_exists -%}
{%- if has_tools and (messages|length == 0 or messages[0]['role'] != 'system') -%}
{{ 'System✿' }}{{ render_tools_block(tools) }}{{ '✿' }}
{%- if messages|length > 0 or add_generation_prompt -%}{{ '\n' }}{%- endif -%}
{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'tool' -%}
{%- if loop.first or messages[loop.index0 - 1]['role'] != 'tool' -%}
{{ 'User✿' }}
{%- endif -%}
{{ render_tool_response(message) }}
{%- if loop.last or messages[loop.index0 + 1]['role'] != 'tool' -%}
{{ '✿' }}
{%- if not loop.last or add_generation_prompt -%}{{ '\n' }}{%- endif -%}
{%- else -%}
{{ '\n' }}
{%- endif -%}
{%- else -%}
{%- if message['role'] == 'assistant' -%}
{%- set role_name = 'Bot' -%}
{%- elif message['role'] == 'system' -%}
{%- set role_name = 'System' -%}
{%- else -%}
{%- set role_name = 'User' -%}
{%- endif -%}
{{ role_name }}{{ '✿' }}{{ render_content(message['content']) }}
{%- if message['role'] == 'system' and loop.first and has_tools -%}
{%- if message['content'] -%}{{ '\n\n' }}{%- endif -%}
{{ render_tools_block(tools) }}
{%- endif -%}
{%- if message['role'] == 'assistant' and message.get('tool_calls') -%}
{%- if message['content'] -%}{{ '\n' }}{%- endif -%}
{%- for tool_call in message['tool_calls'] -%}
{{ render_tool_call(tool_call) }}{%- if not loop.last -%}{{ '\n' }}{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{ '✿' }}
{%- if not loop.last or add_generation_prompt -%}{{ '\n' }}{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ 'Bot✿' }}
{%- endif -%}
"""


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


def get_num_images_per_text_sample(
    images: Any,
    batch_size: int,
) -> list[int] | None:
    image_groups = get_images_per_text_sample(images, batch_size)
    if image_groups is None:
        return None
    return [len(group) for group in image_groups]


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
