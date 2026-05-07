try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:
    from configuration_rwkv7 import RWKV7Config

from fla.models.rwkv7.modeling_rwkv7 import (
    RWKV7ForCausalLM as _FLARWKV7ForCausalLM,
    RWKV7Model as _FLARWKV7Model,
)


class RWKV7Model(_FLARWKV7Model):
    """Local HF-exportable RWKV7 base model wrapper."""

    config_class = RWKV7Config


class RWKV7ForCausalLM(_FLARWKV7ForCausalLM):
    """Local HF-exportable RWKV7 CausalLM wrapper."""

    config_class = RWKV7Config
    _tied_weights_keys = {}


__all__ = ["RWKV7Config", "RWKV7ForCausalLM", "RWKV7Model"]
