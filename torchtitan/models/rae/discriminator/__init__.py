# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen visual backbones and trainable RAE discriminator heads."""

from .dinov3 import DINOv3ViTBackbone, RAEFeatureDiscriminator
from .discriminator import (
    FrozenImageFeatures,
    gan_discriminator_loss,
    gan_generator_loss,
    gan_logits_mean,
    gan_logits_per_image,
    RAEPerceptualLoss,
)
from .perceptual import LPIPSPerceptualLoss

__all__ = [
    "DINOv3ViTBackbone",
    "FrozenImageFeatures",
    "LPIPSPerceptualLoss",
    "RAEFeatureDiscriminator",
    "RAEPerceptualLoss",
    "gan_discriminator_loss",
    "gan_generator_loss",
    "gan_logits_mean",
    "gan_logits_per_image",
]
