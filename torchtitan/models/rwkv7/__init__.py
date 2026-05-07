# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.protocols.model_spec import ModelSpec

from .model import RWKV7ForCausalLM, rwkv7_causal_lm_config
from .parallelize import parallelize_rwkv7
from .state_dict_adapter import RWKV7StateDictAdapter

__all__ = [
    "RWKV7ForCausalLM",
    "model_registry",
    "parallelize_rwkv7",
    "rwkv7_configs",
]


def _debugmodel() -> RWKV7ForCausalLM.Config:
    return rwkv7_causal_lm_config(
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
    )


rwkv7_configs = {
    "debugmodel": _debugmodel,
}


def model_registry(
    flavor: str,
    quantization: list[QuantizationConverter.Config] | None = None,
) -> ModelSpec:
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
        post_optimizer_build_fn=None,
        state_dict_adapter=RWKV7StateDictAdapter,
    )
