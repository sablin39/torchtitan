# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .data import RAEQwenCollator, RAEQwenProcessor
from .decoder import (
    Cosmos3DRotaryPositionEmbedding,
    create_rae_packed_attention_mask,
    create_rae_padding_mask,
    create_rae_static_varlen_metadata,
    create_rae_varlen_metadata,
    RAEAttention,
    RAEBlock,
    RAEDecoder,
    RAEFeedForward,
    unpatchify_packed,
)

__all__ = [
    "RAEDecoder",
    "RAEAttention",
    "RAEBlock",
    "RAEFeedForward",
    "Cosmos3DRotaryPositionEmbedding",
    "create_rae_padding_mask",
    "create_rae_packed_attention_mask",
    "create_rae_static_varlen_metadata",
    "create_rae_varlen_metadata",
    "RAEQwenCollator",
    "RAEQwenProcessor",
    "unpatchify_packed",
]


def __getattr__(name: str):
    if name in {"RAEStage1Trainer", "RAEGANConfig"}:
        from .training import RAEGANConfig, RAEStage1Trainer

        return {"RAEStage1Trainer": RAEStage1Trainer, "RAEGANConfig": RAEGANConfig}[
            name
        ]
    if name in {
        "model_registry",
        "rae_stage1_debug",
        "rae_stage1_dmuon",
        "rae_stage1_openimages",
        "rae_stage1_openimages_static",
        "rae_stage1_dmuon_static",
    }:
        from .config_registry import (
            model_registry,
            rae_stage1_debug,
            rae_stage1_dmuon,
            rae_stage1_dmuon_static,
            rae_stage1_openimages,
            rae_stage1_openimages_static,
        )

        return {
            "model_registry": model_registry,
            "rae_stage1_debug": rae_stage1_debug,
            "rae_stage1_dmuon": rae_stage1_dmuon,
            "rae_stage1_openimages": rae_stage1_openimages,
            "rae_stage1_openimages_static": rae_stage1_openimages_static,
            "rae_stage1_dmuon_static": rae_stage1_dmuon_static,
        }[name]
    raise AttributeError(name)
