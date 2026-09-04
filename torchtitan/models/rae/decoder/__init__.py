# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RAE decoder architecture and variable-media geometry helpers."""

from .decoder import RAEAttention, RAEBlock, RAEDecoder, RAEFeedForward
from .layout import unpatchify_packed
from .packing import (
    create_rae_packed_attention_mask,
    create_rae_padding_mask,
    create_rae_static_varlen_metadata,
    create_rae_varlen_metadata,
)
from .position import Cosmos3DRotaryPositionEmbedding

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
    "unpatchify_packed",
]
