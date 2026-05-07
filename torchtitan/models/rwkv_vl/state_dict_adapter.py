# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import RWKV7VLForConditionalGeneration


class RWKVVLStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: RWKV7VLForConditionalGeneration.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

    def _from_hf_key(self, key: str) -> str | None:
        if key.startswith("model.llm."):
            return "llm." + key.removeprefix("model.llm.")
        if key.startswith("model.proj."):
            return "proj." + key.removeprefix("model.proj.")
        if key == "model.encoder.pos_embed.weight":
            return "vision_encoder.pos_embed"
        if key.startswith("model.encoder.blocks."):
            return "vision_encoder.layers." + key.removeprefix(
                "model.encoder.blocks."
            )
        if key.startswith("model.encoder."):
            return "vision_encoder." + key.removeprefix("model.encoder.")
        if key == "lm_head.weight":
            return key
        return None

    def _to_hf_key(self, key: str) -> str | None:
        if key.startswith("llm."):
            return "model.llm." + key.removeprefix("llm.")
        if key.startswith("proj."):
            return "model.proj." + key.removeprefix("proj.")
        if key == "vision_encoder.pos_embed":
            return "model.encoder.pos_embed.weight"
        if key.startswith("vision_encoder.layers."):
            return "model.encoder.blocks." + key.removeprefix(
                "vision_encoder.layers."
            )
        if key.startswith("vision_encoder."):
            return "model.encoder." + key.removeprefix("vision_encoder.")
        if key == "lm_head.weight":
            return key
        return None

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        for key, value in hf_state_dict.items():
            new_key = self._from_hf_key(key)
            if new_key is None:
                continue
            if key == "model.encoder.patch_embed.proj.weight":
                value = value.reshape(value.shape[0], -1)
            state_dict[new_key] = value
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._to_hf_key(key)
            if new_key is None:
                continue
            if key == "vision_encoder.patch_embed.proj.weight":
                encoder = self.model_config.vision_encoder
                value = value.reshape(
                    value.shape[0],
                    encoder.in_channels,
                    encoder.temporal_patch_size,
                    encoder.patch_size,
                    encoder.patch_size,
                )
            hf_state_dict[new_key] = value
        return hf_state_dict
