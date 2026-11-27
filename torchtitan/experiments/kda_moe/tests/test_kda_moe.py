# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util

import pytest
import torch

from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.kda_moe import build_model_config
from torchtitan.experiments.kda_moe.model import KDAMoEModel, QuantileBalancedMoE
from torchtitan.models.common.attention import LinearAttentionMetadata
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.moe import QuantileBalancedTopKRouter
from torchtitan.models.qwen3_5.model import Qwen35TransformerBlock


class _ParallelismContext:
    ep_enabled = False
    tp = 1

    def get_optional_mesh(self, name: str) -> None:
        assert name == "loss"
        return None


def test_quantile_balancing_optimizer_hook() -> None:
    config = build_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
    assert isinstance(config.layers[0].attention, KDA.Config)
    model = config.build()
    model.init_states()
    assert isinstance(model, KDAMoEModel)
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert isinstance(model.layers["0"], Qwen35TransformerBlock)
    moes = [layer.feed_forward for layer in model.layers.values()]
    for moe in moes:
        assert isinstance(moe, QuantileBalancedMoE)
        assert isinstance(moe.router, QuantileBalancedTopKRouter)
        histogram = moe.router.quantile_balancer.required_bias_histogram_EB
        for expert_id in range(moe.num_experts):
            histogram[expert_id, expert_id % histogram.shape[-1]] = 1
        moe.router.tokens_per_expert_E.fill_(3)
    optimizers = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(
                pattern=r".*",
                lr=0.0,
                weight_decay=0.0,
                fused=False,
            )
        ]
    ).build(model_parts=[model])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)

    model._register_optimizer_hooks(optimizers, [model], _ParallelismContext())
    optimizers.step()

    for moe in moes:
        assert isinstance(moe, QuantileBalancedMoE)
        assert isinstance(moe.router, QuantileBalancedTopKRouter)
        assert moe.expert_bias_E is not None
        assert torch.count_nonzero(moe.expert_bias_E) > 0
        assert (
            torch.count_nonzero(moe.router.quantile_balancer.required_bias_histogram_EB)
            == 0
        )
        assert torch.count_nonzero(moe.router.tokens_per_expert_E) == 0


@pytest.mark.skipif(
    importlib.util.find_spec("cudnn") is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="cuDNN KDA requires the frontend and an SM100/SM103 GPU",
)
def test_ragged_packed_forward_backward_has_finite_gradients() -> None:
    torch.manual_seed(42)
    config = build_model_config(
        "debugmodel", seq_len=257, parallelism=ParallelismConfig()
    )
    with torch.device("meta"):
        model = config.build()
    model.to(torch.bfloat16)
    model.to_empty(device="cuda")
    model.init_states()
    model.train()
    lengths = (1, 3, 5, 17, 31, 63, 257)
    positions = torch.cat(
        [torch.arange(length, device="cuda", dtype=torch.int32) for length in lengths]
    )
    metadata = model._get_attention_metadata(positions, max_num_documents=11)
    assert len(metadata) == 1
    linear_metadata = next(iter(metadata.values()))
    assert isinstance(linear_metadata, LinearAttentionMetadata)
    assert linear_metadata.varlen is not None
    assert linear_metadata.varlen.cu_seq_q.tolist() == [
        0,
        1,
        4,
        9,
        26,
        57,
        120,
        377,
        377,
        377,
        377,
        377,
    ]
    tokens = torch.randint(
        config.vocab_size, (sum(lengths),), device="cuda", dtype=torch.long
    )

    logits = model(tokens, positions, metadata)
    torch.nn.functional.cross_entropy(
        logits.float(), torch.randint_like(tokens, config.vocab_size)
    ).backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().max() > 0, name
    for layer in model.layers.values():
        histogram = (
            layer.feed_forward.router.quantile_balancer.required_bias_histogram_EB
        )
        assert torch.count_nonzero(histogram) > 0
