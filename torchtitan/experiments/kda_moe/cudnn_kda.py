# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""cuDNN KDA with the shared K3 short-convolution implementation."""

from dataclasses import dataclass

import torch

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention.kda import KDAKernel


class CuDNNKDA(Module):
    """Run packed variable-length KDA through cuDNN."""

    @dataclass(kw_only=True, slots=True)
    class Config(KDAKernel.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound

    def forward(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(
                [0, q_1THK.shape[1]], dtype=torch.int32, device=q_1THK.device
            )
        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(torch.int32)
        from cudnn.linear_attention.ops.kda import kimi_delta_attention

        output_THV = kimi_delta_attention(
            q_1THK.squeeze(0),
            k_1THK.squeeze(0),
            v_1THV.squeeze(0),
            raw_gate_1THK.squeeze(0),
            raw_beta_1TH.squeeze(0),
            cu_seqlens,
            scale=q_1THK.shape[-1] ** -0.5,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            gate_lower_bound=self.lower_bound,
            a_log=A_log_H.float(),
            dt_bias=dt_bias_HK.float(),
            plan_name="kda_frost",
        )[0]
        return output_THV.unsqueeze(0)
