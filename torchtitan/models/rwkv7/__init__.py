# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import (
    rwkv7_backbone_config,
    rwkv7_causal_lm_from_backbone,
    RWKV7Backbone,
    RWKV7ForCausalLM,
)
from .parallelize import parallelize_rwkv7
from .state_dict_adapter import RWKV7StateDictAdapter

__all__ = [
    "RWKV7ForCausalLM",
    "model_registry",
    "parallelize_rwkv7",
    "rwkv7_backbones",
    "rwkv7_configs",
]


# ---------------------------------------------------------------------------
# RWKV7 text-LM backbone factories
#
# Architectural parameters below mirror the released G1 checkpoints at
# /mnt/raid0_8t/rwkv7-g1 (head_dim=64; vocab=65536; ctx_len is set at the
# trainer level, not the model). These factories are shared by the RWKV7 LM
# (this folder) and the RWKV-VL model, which composes a backbone with vision
# encoder + lm_head.
# ---------------------------------------------------------------------------


def _backbone(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_heads: int,
    intermediate_size: int,
    a_low_rank_dim: int,
    decay_low_rank_dim: int,
    gate_low_rank_dim: int,
    v_low_rank_dim: int,
    vocab_size: int = 65536,
    head_dim: int = 64,
    chunk_size: int = 64,
    skip_embedding_init: bool = False,
    **rwkv7_kwargs,
) -> RWKV7Backbone.Config:
    return rwkv7_backbone_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        value_dim=[hidden_size] * num_hidden_layers,
        norm_eps=1e-5,
        norm_bias=True,
        hidden_act="sqrelu",
        a_low_rank_dim=a_low_rank_dim,
        decay_low_rank_dim=decay_low_rank_dim,
        gate_low_rank_dim=gate_low_rank_dim,
        v_low_rank_dim=v_low_rank_dim,
        chunk_size=chunk_size,
        skip_embedding_init=skip_embedding_init,
        **rwkv7_kwargs,
    )


def _debug_backbone(*, skip_embedding_init: bool = False) -> RWKV7Backbone.Config:
    return rwkv7_backbone_config(
        vocab_size=2048,
        hidden_size=256,
        num_hidden_layers=4,
        num_heads=4,
        head_dim=64,
        intermediate_size=1024,
        a_low_rank_dim=32,
        decay_low_rank_dim=32,
        gate_low_rank_dim=64,
        v_low_rank_dim=32,
        chunk_size=64,
        skip_embedding_init=skip_embedding_init,
    )


def _moe_3b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        vocab_size=151680,
        hidden_size=1536,
        num_hidden_layers=12,
        num_heads=12,
        head_dim=128,
        intermediate_size=8192,
        a_low_rank_dim=96,
        decay_low_rank_dim=96,
        gate_low_rank_dim=320,
        v_low_rank_dim=64,
        moe_channel_mix_start_layer=1,
        moe_channel_mix_layer_freq=1,
        moe_channel_mix_intermediate_size=1024,
        moe_channel_mix_num_experts=64,
        moe_channel_mix_num_shared_experts=2,
        moe_channel_mix_top_k=6,
        moe_channel_mix_score_func="softmax",
        moe_channel_mix_route_norm=False,
        moe_channel_mix_route_scale=1.0,
        moe_channel_mix_num_expert_groups=1,
        moe_channel_mix_num_limited_groups=1,
        moe_channel_mix_load_balance_coeff=1e-3,
        **kwargs,
    )


def _g1d_0_4b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        hidden_size=1024,
        num_hidden_layers=24,
        num_heads=16,
        intermediate_size=4096,
        a_low_rank_dim=64,
        decay_low_rank_dim=64,
        gate_low_rank_dim=128,
        v_low_rank_dim=32,
        **kwargs,
    )


def _g1f_1_5b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        hidden_size=2048,
        num_hidden_layers=24,
        num_heads=32,
        intermediate_size=8192,
        a_low_rank_dim=96,
        decay_low_rank_dim=96,
        gate_low_rank_dim=256,
        v_low_rank_dim=64,
        **kwargs,
    )


def _g1f_2_9b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        hidden_size=2560,
        num_hidden_layers=32,
        num_heads=40,
        intermediate_size=10240,
        a_low_rank_dim=96,
        decay_low_rank_dim=96,
        gate_low_rank_dim=320,
        v_low_rank_dim=64,
        **kwargs,
    )


def _g1f_7_2b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        hidden_size=4096,
        num_hidden_layers=32,
        num_heads=64,
        intermediate_size=16384,
        a_low_rank_dim=128,
        decay_low_rank_dim=128,
        gate_low_rank_dim=480,
        v_low_rank_dim=96,
        **kwargs,
    )


def _g1f_13_3b_backbone(**kwargs) -> RWKV7Backbone.Config:
    return _backbone(
        hidden_size=4096,
        num_hidden_layers=61,
        num_heads=64,
        intermediate_size=16384,
        a_low_rank_dim=192,
        decay_low_rank_dim=192,
        gate_low_rank_dim=384,
        v_low_rank_dim=128,
        **kwargs,
    )


rwkv7_backbones: dict[str, Callable[..., RWKV7Backbone.Config]] = {
    "debugmodel": _debug_backbone,
    "3B-MoE": _moe_3b_backbone,
    "0.4B": _g1d_0_4b_backbone,
    "1.5B": _g1f_1_5b_backbone,
    "2.9B": _g1f_2_9b_backbone,
    "7.2B": _g1f_7_2b_backbone,
    "13.3B": _g1f_13_3b_backbone,
}


def _lm_config(flavor: str) -> Callable[[], RWKV7ForCausalLM.Config]:
    return lambda: rwkv7_causal_lm_from_backbone(rwkv7_backbones[flavor]())


rwkv7_configs: dict[str, Callable[[], RWKV7ForCausalLM.Config]] = {
    flavor: _lm_config(flavor) for flavor in rwkv7_backbones
}


def model_registry(
    flavor: str,
    quantization: list[QuantizationConverter.Config] | None = None,
) -> ModelSpec:
    """Build a ``ModelSpec`` for a registered RWKV7 flavor."""
    config = rwkv7_configs[flavor]()
    if quantization is not None:
        for q in quantization:
            q.build().convert(config)
    return ModelSpec(
        name="rwkv7",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_rwkv7,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=RWKV7StateDictAdapter,
    )
