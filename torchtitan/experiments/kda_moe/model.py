# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only KDA and mixture-of-experts decoder."""

from dataclasses import dataclass, field

import torch

from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock


class QuantileBalancedMoE(MoE):
    """Common MoE with a persistent bias for quantile-only load balancing."""

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config):
        if config.load_balance_coeff is not None:
            raise ValueError(
                "QuantileBalancedMoE cannot combine sign-based and quantile balancing."
            )
        super().__init__(config)
        del self.expert_bias_E
        self.register_buffer(
            "expert_bias_E",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.router.tokens_per_expert_E.device
        super()._init_self_buffers(buffer_device=buffer_device)
        with torch.device(buffer_device):
            self.expert_bias_E = torch.zeros(
                self.router.num_experts,
                dtype=torch.float32,
            )


class KDAMoEModel(Decoder):
    """Text-only decoder with KDA and quantile-balanced MoE on every layer."""

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.moe import register_moe_quantile_balancing_hook

        register_moe_quantile_balancing_hook(
            optimizers, model_parts, parallelism_context
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[Qwen35TransformerBlock.Config]  # pyrefly: ignore [bad-override]
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(
                regions=[
                    "loss",
                    "gated_rmsnorm",
                    "fused_binary_activation",
                    "fp32_to_bf16_split",
                ]
            )
        )
