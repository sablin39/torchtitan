# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility imports for multimodal image processing utilities.

The implementation lives in ``processor_core`` so TorchTitan datasets and HF
exported processors use the same resize/token-count policy.
"""

from torchtitan.hf_datasets.multimodal.processor_core import (
    RWKVVLImageProcessorConfig,
    RWKVVLProcessedImages,
    calculate_vision_tokens,
    get_image_size,
    make_image_config_from_processor,
    normalize_image,
    process_image,
    process_images,
    resize_sizes_to_total_pixels,
    smart_resize,
    vision_to_patches,
)


__all__ = [
    "RWKVVLImageProcessorConfig",
    "RWKVVLProcessedImages",
    "calculate_vision_tokens",
    "get_image_size",
    "make_image_config_from_processor",
    "normalize_image",
    "process_image",
    "process_images",
    "resize_sizes_to_total_pixels",
    "smart_resize",
    "vision_to_patches",
]
