# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import RWKV7ForCausalLM


class RWKV7StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: RWKV7ForCausalLM.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

    def _from_hf_key(self, key: str) -> str | None:
        if key.startswith("model.llm."):
            return "llm." + key.removeprefix("model.llm.")
        if key == "lm_head.weight":
            return key
        return None

    def _to_hf_key(self, key: str) -> str | None:
        if key.startswith("llm."):
            return "model.llm." + key.removeprefix("llm.")
        if key == "lm_head.weight":
            return key
        return None

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        for key, value in hf_state_dict.items():
            new_key = self._from_hf_key(key)
            if new_key is not None:
                state_dict[new_key] = value
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._to_hf_key(key)
            if new_key is not None:
                hf_state_dict[new_key] = value
        return hf_state_dict
