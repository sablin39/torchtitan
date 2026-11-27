# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure KDA MoE model flavors."""

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import (
    Conv1d,
    GatedRMSNorm,
    Linear,
    RowParallelLinear,
    Sigmoid,
    SiTUGLU,
    VocabParallelEmbedding,
)
from torchtitan.models.common.attention.kda import InnerKDA, KDA
from torchtitan.models.common.config_utils import (
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rowwise_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import QuantileBalancedTopKRouter
from torchtitan.models.common.moe.sharding import (
    moe_sharding_config,
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.kimi_k3.sharding import (
    inner_kda_sharding_config,
    kda_conv_sharding_config,
    kda_sharding_config,
)
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock

from .cudnn_kda import CuDNNKDA
from .model import KDAMoEModel, QuantileBalancedMoE

__all__ = ["KDAMoEModel", "MODEL_FLAVORS", "build_model_config"]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_CONV_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_NORM_INIT = {"weight": nn.init.ones_}
_ROUTED_EXPERT_INIT: dict[str, Callable] = {
    "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
    "w2_EDF": partial(nn.init.trunc_normal_, std=0.02),
    "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
}


def _situ_glu() -> SiTUGLU.Config:
    return SiTUGLU.Config(beta=4.0, linear_beta=25.0)


_KDA_HEAD_DIM = 128
_KDA_CONV_KERNEL_SIZE = 4


def _output_linear_init(dim: int) -> dict[str, Callable]:
    scale = dim**-0.5
    return {
        "weight": partial(
            nn.init.trunc_normal_,
            std=scale,
            a=-3 * scale,
            b=3 * scale,
        )
    }


def _a_log_init(parameter: nn.Parameter) -> None:
    with torch.no_grad():
        nn.init.uniform_(parameter, 1.0, 16.0)
        parameter.log_()


def _linear(
    in_features: int,
    out_features: int,
    *,
    param_init: dict[str, Callable] | None = None,
    sharding_config: ShardingConfig,
) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=param_init or _LINEAR_INIT,
        sharding_config=sharding_config,
    )


def _norm(dim: int, *, sharding_config: ShardingConfig) -> RMSNorm.Config:
    return RMSNorm.Config(
        normalized_shape=dim,
        eps=1e-5,
        param_init=_NORM_INIT,
        sharding_config=sharding_config,
    )


def _kda_config(
    *,
    dim: int,
    num_heads: int,
    enable_sp: bool,
) -> KDA.Config:
    projection_dim = num_heads * _KDA_HEAD_DIM

    def conv() -> Conv1d.Config:
        return Conv1d.Config(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=_KDA_CONV_KERNEL_SIZE,
            groups=projection_dim,
            bias=False,
            param_init=_CONV_INIT,
            sharding_config=kda_conv_sharding_config(),
        )

    def colwise_linear(in_features: int, out_features: int) -> Linear.Config:
        return _linear(
            in_features,
            out_features,
            sharding_config=replicated_input_colwise_sharding_config(),
        )

    return KDA.Config(
        n_heads=num_heads,
        head_dim=_KDA_HEAD_DIM,
        conv_kernel_size=_KDA_CONV_KERNEL_SIZE,
        q_proj=colwise_linear(dim, projection_dim),
        k_proj=colwise_linear(dim, projection_dim),
        v_proj=colwise_linear(dim, projection_dim),
        q_conv=conv(),
        k_conv=conv(),
        v_conv=conv(),
        forget_a=_linear(
            dim,
            _KDA_HEAD_DIM,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        forget_b=colwise_linear(_KDA_HEAD_DIM, projection_dim),
        beta=colwise_linear(dim, num_heads),
        output_gate=colwise_linear(dim, projection_dim),
        inner_attention=InnerKDA.Config(
            head_dim=_KDA_HEAD_DIM,
            conv_kernel_size=_KDA_CONV_KERNEL_SIZE,
            kernel=CuDNNKDA.Config(),
            sharding_config=inner_kda_sharding_config(),
        ),
        output_norm=GatedRMSNorm.Config(
            dim=_KDA_HEAD_DIM,
            eps=1e-5,
            activation_fn=Sigmoid.Config(),
            param_init=_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        output_proj=RowParallelLinear.Config(
            in_features=projection_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.zeros_,
        },
        sharding_config=kda_sharding_config(enable_sp=enable_sp),
    )


def _moe_config(
    *,
    dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    shared_expert_hidden_dim: int,
    enable_sp: bool,
    enable_ep: bool,
) -> QuantileBalancedMoE.Config:
    return QuantileBalancedMoE.Config(
        num_experts=num_experts,
        router=QuantileBalancedTopKRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            gate=HiMidLoLinear.Config(
                in_features=dim,
                out_features=num_experts,
                backward_mode="hi_mid_lo",
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=router_gate_sharding_config(),
            ),
            score_func=Sigmoid.Config(),
            route_norm=True,
            route_scale=1.0,
            num_bins=1000,
            sharding_config=router_sharding_config(),
        ),
        routed_experts=replace(
            make_routed_experts_config(
                dim=dim,
                hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init=_ROUTED_EXPERT_INIT,
                enable_ep=enable_ep,
            ),
            activation_fn=_situ_glu(),
        ),
        shared_experts=replace(
            make_shared_expert_ffn_config(
                dim=dim,
                hidden_dim=shared_expert_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_LINEAR_INIT,
                enable_sp=enable_sp,
            ),
            activation_fn=_situ_glu(),
        ),
        load_balance_coeff=None,
        sharding_config=moe_sharding_config(enable_sp=enable_sp),
    )


def _kda_moe_config(
    *,
    max_context_length: int,
    dim: int,
    vocab_size: int,
    num_layers: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    shared_expert_hidden_dim: int,
    parallelism: ParallelismConfig,
) -> KDAMoEModel.Config:
    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    num_heads = dim // _KDA_HEAD_DIM

    layers = [
        Qwen35TransformerBlock.Config(
            attention=_kda_config(dim=dim, num_heads=num_heads, enable_sp=enable_sp),
            feed_forward=_moe_config(
                dim=dim,
                expert_hidden_dim=expert_hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                shared_expert_hidden_dim=shared_expert_hidden_dim,
                enable_sp=enable_sp,
                enable_ep=enable_ep,
            ),
            attention_norm=_norm(dim, sharding_config=norm_sharding),
            ffn_norm=_norm(dim, sharding_config=norm_sharding),
            sharding_config=ShardingConfig(
                input_shardings={
                    "x_TD": residual_activation_placement(enable_sp=enable_sp)
                },
                output_shardings=residual_activation_placement(enable_sp=enable_sp),
            ),
        )
        for _ in range(num_layers)
    ]

    # Tied weights share lm_head's initializer, matching other tied decoders.
    return KDAMoEModel.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=VocabParallelEmbedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            sharding_config=token_embedding_sharding_config(enable_sp=enable_sp),
        ),
        layers=layers,
        norm=_norm(dim, sharding_config=norm_sharding),
        lm_head=_linear(
            dim,
            vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
    )


MODEL_FLAVORS = {
    "debugmodel": dict(
        dim=256,
        vocab_size=2048,
        num_layers=3,
        expert_hidden_dim=128,
        num_experts=8,
        top_k=2,
        shared_expert_hidden_dim=512,
    ),
    "S": dict(
        dim=768,
        vocab_size=163840,
        num_layers=20,
        expert_hidden_dim=384,
        num_experts=60,
        top_k=4,
        shared_expert_hidden_dim=3072,
    ),
    "M": dict(
        dim=1024,
        vocab_size=163840,
        num_layers=26,
        expert_hidden_dim=512,
        num_experts=60,
        top_k=4,
        shared_expert_hidden_dim=4096,
    ),
    "L": dict(
        dim=1280,
        vocab_size=163840,
        num_layers=32,
        expert_hidden_dim=640,
        num_experts=60,
        top_k=4,
        shared_expert_hidden_dim=5120,
    ),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> KDAMoEModel.Config:
    context_len = 2048 if seq_len is None else seq_len
    if not 1 <= context_len <= 262144:
        raise ValueError(
            f"KDA MoE supports seq_len from 1 through 262144, got {context_len}."
        )
    geometry = MODEL_FLAVORS[flavor]
    dim = geometry["dim"]
    tp_degree = parallelism.tensor_parallel_degree
    num_heads = dim // _KDA_HEAD_DIM
    if num_heads % tp_degree != 0:
        raise ValueError(
            f"KDA MoE tensor parallel degree ({tp_degree}) must divide "
            f"the number of KDA heads ({num_heads}) for flavor {flavor}."
        )
    if geometry["shared_expert_hidden_dim"] % tp_degree != 0:
        raise ValueError(
            f"KDA MoE tensor parallel degree ({tp_degree}) must divide "
            f"the shared-expert hidden dimension ({geometry['shared_expert_hidden_dim']})."
        )
    if geometry["num_experts"] % parallelism.expert_parallel_degree != 0:
        raise ValueError(
            f"KDA MoE expert parallel degree ({parallelism.expert_parallel_degree}) "
            f"must divide the number of experts ({geometry['num_experts']})."
        )
    return _kda_moe_config(
        max_context_length=context_len,
        parallelism=parallelism,
        **geometry,
    )
