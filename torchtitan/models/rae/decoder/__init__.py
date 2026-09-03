"""RAE decoder architecture and variable-media geometry helpers."""

from .decoder import RAEAttention, RAEBlock, RAEDecoder, RAEFeedForward
from .layout import unpatchify_packed
from .packing import (
    create_rae_packed_attention_mask,
    create_rae_padding_mask,
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
    "create_rae_varlen_metadata",
    "unpatchify_packed",
]
