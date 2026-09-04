# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RAE Stage 1 training loop, augmentation, and metric logging."""

from .augmentation import DiscriminatorAugmentation
from .graphs import (
    RAEDiscriminatorGraph,
    RAEDiscriminatorGraphOutput,
    RAEGeneratorGraphOutput,
    RAEGeneratorLossGraph,
    StaticCUDAGraph,
)
from .metrics import log_stage1_metrics
from .trainer import RAEGANAugmentConfig, RAEGANConfig, RAEStage1Trainer
from .validation import RAEValidator

__all__ = [
    "DiscriminatorAugmentation",
    "RAEDiscriminatorGraph",
    "RAEDiscriminatorGraphOutput",
    "RAEGeneratorGraphOutput",
    "RAEGeneratorLossGraph",
    "RAEGANAugmentConfig",
    "RAEGANConfig",
    "RAEStage1Trainer",
    "RAEValidator",
    "StaticCUDAGraph",
    "log_stage1_metrics",
]
