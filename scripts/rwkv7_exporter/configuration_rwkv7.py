try:
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _FLARWKV7Config
except ImportError:
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _FLARWKV7Config


class RWKV7Config(_FLARWKV7Config):
    """Local HF-exportable RWKV7 config wrapper."""

    model_type = "rwkv7"
    __init__ = _FLARWKV7Config.__init__


__all__ = ["RWKV7Config"]
