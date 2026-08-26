# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.models.qwen3_5.vision_encoder import PatchMerger, Qwen35VisionEncoder
from torchtitan.protocols.module import ModuleList


class DeepStackPatchMerger(PatchMerger):
    """Qwen patch merger with the legacy DeepStack post-shuffle norm order."""

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        x_MK = x_TD.view(-1, self.merged_hidden_size)
        x_MK = self.norm(x_MK)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x_MK)))


class RWKVVisionEncoder(Qwen35VisionEncoder):
    """Qwen3.5 vision encoder with optional RWKV DeepStack feature taps."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35VisionEncoder.Config):
        deepstack_visual_indices: list[int] = field(default_factory=list)
        deepstack_merger: PatchMerger.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        if len(set(config.deepstack_visual_indices)) != len(
            config.deepstack_visual_indices
        ):
            raise ValueError("deepstack_visual_indices must not contain duplicates")
        invalid_indices = [
            index
            for index in config.deepstack_visual_indices
            if index < 0 or index >= config.num_layers
        ]
        if invalid_indices:
            raise ValueError(
                "deepstack_visual_indices must refer to vision layers, got "
                f"{invalid_indices} for num_layers={config.num_layers}"
            )
        deepstack_merger = config.deepstack_merger
        if config.deepstack_visual_indices:
            if deepstack_merger is None:
                raise ValueError(
                    "deepstack_merger is required when deepstack_visual_indices is set"
                )
            deepstack_mergers = [
                DeepStackPatchMerger(deepstack_merger)
                for _ in config.deepstack_visual_indices
            ]
        else:
            deepstack_mergers = []

        self.deepstack_visual_indices = config.deepstack_visual_indices
        self._deepstack_index_by_layer = {
            layer_index: tap_index
            for tap_index, layer_index in enumerate(config.deepstack_visual_indices)
        }
        self.deepstack_merger_list = ModuleList(deepstack_mergers)

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        deepstack_features: list[torch.Tensor] = []

        def collect_deepstack(
            layer_index: int,
            hidden_states: torch.Tensor,
        ) -> None:
            tap_index = self._deepstack_index_by_layer.get(layer_index)
            if tap_index is not None:
                deepstack_features.append(
                    self.deepstack_merger_list[tap_index](hidden_states)
                )

        merged_features = self._forward_with_layer_callback(
            pixel_values,
            grid_thw=grid_thw,
            layer_callback=collect_deepstack,
        )
        return merged_features, deepstack_features
