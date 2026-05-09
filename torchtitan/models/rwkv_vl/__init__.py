# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.quantization import QuantizationConverter
from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.models.rwkv7.model import rwkv7_backbone_config
from torchtitan.protocols.model_spec import ModelSpec

from .model import RWKV7VLForConditionalGeneration, VisualAdapter
from .parallelize import parallelize_rwkv_vl
from .state_dict_adapter import RWKVVLStateDictAdapter

__all__ = [
    "RWKV7VLForConditionalGeneration",
    "model_registry",
    "parallelize_rwkv_vl",
    "rwkv_vl_configs",
]


_DEBUG_SPECIAL_TOKEN_IDS = {
    "pad": 2004,
    "vision_start": 2005,
    "vision_end": 2006,
    "image": 2007,
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}


def _vl_linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


def _vl_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    n_layers: int,
    n_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    deepstack_visual_indices: list[int],
    in_channels: int = 3,
) -> Qwen3VLVisionEncoder.Config:
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    return Qwen3VLVisionEncoder.Config(
        dim=dim,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        deepstack_visual_indices=deepstack_visual_indices,
        patch_embed_proj=_vl_linear(patch_dim, dim),
        attn_qkv=_vl_linear(dim, dim * 3),
        attn_proj=_vl_linear(dim, dim),
        mlp_fc1=_vl_linear(dim, ffn_dim),
        mlp_fc2=_vl_linear(ffn_dim, dim),
        merger_fc1=_vl_linear(merged_hidden_size, merged_hidden_size),
        merger_fc2=_vl_linear(merged_hidden_size, out_hidden_size),
        param_init=_POS_EMBED_INIT,
    )


def _debugmodel() -> RWKV7VLForConditionalGeneration.Config:
    vocab_size = 2048
    hidden_size = 256
    return RWKV7VLForConditionalGeneration.Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        llm=rwkv7_backbone_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_heads=4,
            head_dim=64,
            intermediate_size=1024,
            value_dim=[hidden_size] * 4,
            norm_eps=1e-5,
            norm_bias=True,
            hidden_act="sqrelu",
            a_low_rank_dim=32,
            decay_low_rank_dim=32,
            gate_low_rank_dim=64,
            v_low_rank_dim=32,
            chunk_size=64,
        ),
        vision_encoder=_vl_vision_encoder_config(
            dim=128,
            ffn_dim=512,
            n_layers=2,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=hidden_size,
            num_position_embeddings=1024,
            deepstack_visual_indices=[],
        ),
        proj=VisualAdapter.Config(
            encoder_dim=hidden_size,
            hidden_dim=1024,
            project_dim=hidden_size,
            num_deepstack=0,
            norm_eps=1e-5,
        ),
        lm_head=Linear.Config(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            param_init={
                "weight": partial(
                    nn.init.trunc_normal_,
                    std=hidden_size**-0.5,
                    a=-3 * hidden_size**-0.5,
                    b=3 * hidden_size**-0.5,
                )
            },
        ),
        image_token_id=_DEBUG_SPECIAL_TOKEN_IDS["image"],
        vision_start_token_id=_DEBUG_SPECIAL_TOKEN_IDS["vision_start"],
        vision_end_token_id=_DEBUG_SPECIAL_TOKEN_IDS["vision_end"],
    )


def _qwen3vit_v100m_vision_encoder_config() -> Qwen3VLVisionEncoder.Config:
    return _vl_vision_encoder_config(
        dim=768,
        ffn_dim=3072,
        n_layers=12,
        n_heads=12,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=1024,
        num_position_embeddings=2304,
        deepstack_visual_indices=[],
    )


def _qwen3vit_v400m_vision_encoder_config() -> Qwen3VLVisionEncoder.Config:
    return _vl_vision_encoder_config(
        dim=1024,
        ffn_dim=4096,
        n_layers=24,
        n_heads=16,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indices=[5, 11, 17],
    )


def _rwkv_vl_config(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    num_heads: int,
    intermediate_size: int,
    a_low_rank_dim: int,
    decay_low_rank_dim: int,
    gate_low_rank_dim: int,
    v_low_rank_dim: int,
    vision_encoder: Qwen3VLVisionEncoder.Config,
) -> RWKV7VLForConditionalGeneration.Config:
    vocab_size = 65536
    return RWKV7VLForConditionalGeneration.Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        llm=rwkv7_backbone_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_heads=num_heads,
            head_dim=64,
            intermediate_size=intermediate_size,
            value_dim=[hidden_size] * num_hidden_layers,
            norm_eps=1e-5,
            norm_bias=True,
            hidden_act="sqrelu",
            a_low_rank_dim=a_low_rank_dim,
            decay_low_rank_dim=decay_low_rank_dim,
            gate_low_rank_dim=gate_low_rank_dim,
            v_low_rank_dim=v_low_rank_dim,
            chunk_size=64,
        ),
        vision_encoder=vision_encoder,
        proj=VisualAdapter.Config(
            encoder_dim=vision_encoder.out_hidden_size,
            hidden_dim=None,
            project_dim=hidden_size,
            num_deepstack=len(vision_encoder.deepstack_visual_indices),
            norm_eps=1e-5,
        ),
        lm_head=Linear.Config(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            param_init={
                "weight": partial(
                    nn.init.trunc_normal_,
                    std=hidden_size**-0.5,
                    a=-3 * hidden_size**-0.5,
                    b=3 * hidden_size**-0.5,
                )
            },
        ),
        image_token_id=65532,
        vision_start_token_id=65530,
        vision_end_token_id=65531,
    )


def _g1d_0_4b_v100m() -> RWKV7VLForConditionalGeneration.Config:
    return _rwkv_vl_config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_heads=16,
        intermediate_size=4096,
        a_low_rank_dim=64,
        decay_low_rank_dim=64,
        gate_low_rank_dim=128,
        v_low_rank_dim=32,
        vision_encoder=_qwen3vit_v100m_vision_encoder_config(),
    )


def _g1f_1_5b_v100m() -> RWKV7VLForConditionalGeneration.Config:
    return _rwkv_vl_config(
        hidden_size=2048,
        num_hidden_layers=24,
        num_heads=32,
        intermediate_size=8192,
        a_low_rank_dim=96,
        decay_low_rank_dim=96,
        gate_low_rank_dim=256,
        v_low_rank_dim=64,
        vision_encoder=_qwen3vit_v100m_vision_encoder_config(),
    )


def _g1f_1_5b_v400m() -> RWKV7VLForConditionalGeneration.Config:
    return _rwkv_vl_config(
        hidden_size=2048,
        num_hidden_layers=24,
        num_heads=32,
        intermediate_size=8192,
        a_low_rank_dim=96,
        decay_low_rank_dim=96,
        gate_low_rank_dim=256,
        v_low_rank_dim=64,
        vision_encoder=_qwen3vit_v400m_vision_encoder_config(),
    )


rwkv_vl_configs = {
    "debugmodel": _debugmodel,
    "0.4B-v100M": _g1d_0_4b_v100m,
    "1.5B-v100M": _g1f_1_5b_v100m,
    "1.5B-v400M": _g1f_1_5b_v400m,
}


def model_registry(
    flavor: str,
    quantization: list[QuantizationConverter.Config] | None = None,
) -> ModelSpec:
    config = rwkv_vl_configs[flavor]()
    if quantization is not None:
        for q in quantization:
            q.build().convert(config)
    return ModelSpec(
        name="rwkv_vl",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_rwkv_vl,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=RWKVVLStateDictAdapter,
    )
