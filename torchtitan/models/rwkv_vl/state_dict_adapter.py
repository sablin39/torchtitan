# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch

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
        if key.startswith("model.llm.layers.0.pre_norm."):
            return "llm.pre_norm." + key.removeprefix("model.llm.layers.0.pre_norm.")
        if key.startswith("model.llm."):
            return "llm." + key.removeprefix("model.llm.")
        if key.startswith("model.proj."):
            return "proj." + key.removeprefix("model.proj.")
        if key == "model.encoder.pos_embed.weight":
            return "vision_encoder.pos_embed"
        if key.startswith("model.encoder.patch_embed.proj."):
            return "vision_encoder.patch_embed." + key.removeprefix(
                "model.encoder.patch_embed.proj."
            )
        if key.startswith("model.encoder.deepstack_merger_list."):
            return "vision_encoder.deepstack_merger_list." + key.removeprefix(
                "model.encoder.deepstack_merger_list."
            )
        if key.startswith("model.encoder.blocks."):
            if ".attn.qkv." in key:
                return None
            return "vision_encoder.layers." + key.removeprefix("model.encoder.blocks.")
        if key.startswith("model.encoder."):
            return "vision_encoder." + key.removeprefix("model.encoder.")
        if key == "lm_head.weight":
            return key
        return None

    def _to_hf_key(self, key: str) -> str | None:
        if key.startswith("llm.pre_norm."):
            return "model.llm.layers.0.pre_norm." + key.removeprefix("llm.pre_norm.")
        if key.startswith("llm."):
            return "model.llm." + key.removeprefix("llm.")
        if key.startswith("proj."):
            return "model.proj." + key.removeprefix("proj.")
        if key == "vision_encoder.pos_embed":
            return "model.encoder.pos_embed.weight"
        if key.startswith("vision_encoder.patch_embed."):
            return "model.encoder.patch_embed.proj." + key.removeprefix(
                "vision_encoder.patch_embed."
            )
        if key.startswith("vision_encoder.deepstack_merger_list."):
            return "model.encoder.deepstack_merger_list." + key.removeprefix(
                "vision_encoder.deepstack_merger_list."
            )
        if key.startswith("vision_encoder.layers."):
            if re.search(r"\.attn\.w[qkv]\.(weight|bias)$", key):
                return None
            return "model.encoder.blocks." + key.removeprefix("vision_encoder.layers.")
        if key.startswith("vision_encoder."):
            return "model.encoder." + key.removeprefix("vision_encoder.")
        if key == "lm_head.weight":
            return key
        return None

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        for key, value in hf_state_dict.items():
            qkv_match = re.fullmatch(
                r"model\.encoder\.blocks\.(\d+)\.attn\.qkv\.(weight|bias)",
                key,
            )
            if qkv_match is not None:
                layer_index, suffix = qkv_match.groups()
                query, key_proj, value_proj = value.chunk(3, dim=0)
                state_dict[
                    f"vision_encoder.layers.{layer_index}.attn.wq.{suffix}"
                ] = query
                state_dict[
                    f"vision_encoder.layers.{layer_index}.attn.wk.{suffix}"
                ] = key_proj
                state_dict[
                    f"vision_encoder.layers.{layer_index}.attn.wv.{suffix}"
                ] = value_proj
                continue
            new_key = self._from_hf_key(key)
            if new_key is None:
                continue
            if key == "model.encoder.patch_embed.proj.weight":
                value = value.reshape(value.shape[0], -1)
            state_dict[new_key] = value
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        hf_state_dict = {}
        vision_qkv: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            qkv_match = re.fullmatch(
                r"vision_encoder\.layers\.(\d+)\.attn\.(w[qkv])\.(weight|bias)",
                key,
            )
            if qkv_match is not None:
                layer_index, projection, suffix = qkv_match.groups()
                vision_qkv.setdefault((layer_index, suffix), {})[projection] = value
                continue
            new_key = self._to_hf_key(key)
            if new_key is None:
                continue
            if key == "vision_encoder.patch_embed.weight":
                encoder = self.model_config.vision_encoder
                value = value.reshape(
                    value.shape[0],
                    encoder.in_channels,
                    encoder.temporal_patch_size,
                    encoder.patch_size,
                    encoder.patch_size,
                )
            hf_state_dict[new_key] = value
        for (layer_index, suffix), projections in vision_qkv.items():
            missing = {"wq", "wk", "wv"} - projections.keys()
            if missing:
                raise ValueError(
                    f"Incomplete vision QKV for layer {layer_index}: "
                    f"missing {sorted(missing)} {suffix} tensors"
                )
            hf_state_dict[
                f"model.encoder.blocks.{layer_index}.attn.qkv.{suffix}"
            ] = torch.cat(
                [projections["wq"], projections["wk"], projections["wv"]],
                dim=0,
            )
        return hf_state_dict
