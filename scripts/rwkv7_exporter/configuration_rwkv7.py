# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _FLARWKV7Config


class RWKV7Config(_FLARWKV7Config):
    """Local HF-exportable RWKV7 config wrapper."""

    model_type = "rwkv7"
    __init__ = _FLARWKV7Config.__init__


__all__ = ["RWKV7Config"]
