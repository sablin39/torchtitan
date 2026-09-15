# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch

from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.rae.decoder import (
    create_rae_static_varlen_metadata,
    create_rae_varlen_metadata,
    RAEDecoder,
)


def _cuda_varlen_decoder() -> RAEDecoder:
    config = RAEDecoder.Config(
        latent_dim=8,
        image_size=32,
        patch_size=8,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        attention_backend="varlen",
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cuda")
    model.init_states()
    return model.to(dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rae_varlen_gqa_cuda_forward_backward() -> None:
    model = _cuda_varlen_decoder().train()
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 3]])
    sequence_lengths = grid_thw.prod(dim=-1)
    metadata = create_rae_varlen_metadata(
        sequence_lengths,
        device=torch.device("cuda"),
    )
    latents_TC = torch.randn(
        int(sequence_lengths.sum().item()),
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output_TP = model(
        latents_TC,
        grid_thw=grid_thw,
        fps=torch.tensor([0.0, 24.0], device="cuda"),
        attention_masks=metadata,
    )
    assert output_TP.shape == (28, 192)
    assert torch.isfinite(output_TP).all()

    output_TP.float().square().mean().backward()
    assert latents_TC.grad is not None
    assert torch.isfinite(latents_TC.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rae_varlen_gqa_cuda_block_fullgraph_compile() -> None:
    model = _cuda_varlen_decoder().eval()
    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 3]])
    sequence_lengths = grid_thw.prod(dim=-1)
    metadata = create_rae_varlen_metadata(
        sequence_lengths,
        device=torch.device("cuda"),
    )
    latents_TC = torch.randn(
        int(sequence_lengths.sum().item()),
        8,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_TC = model.input_projection(latents_TC)
    positions_T3 = model.layers[0].attention.rope.build_packed_positions(
        grid_thw,
        fps=torch.tensor([0.0, 24.0], device="cuda"),
    ).to("cuda", non_blocking=True)

    compiled_block = torch.compile(model.layers[0], fullgraph=True, dynamic=False)
    output_TD = compiled_block(
        hidden_TC,
        positions=positions_T3,
        attention_masks=metadata,
    )
    assert output_TD.shape == hidden_TC.shape
    assert torch.isfinite(output_TD).all()
    torch._dynamo.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rae_static_padded_forward_fullgraph_compile() -> None:
    config = RAEDecoder.Config(
        latent_dim=1024,
        image_size=-1,
        patch_size=16,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=2048,
        attention_backend="varlen",
        static_sequence_length=65536,
    )
    config.update_from_config(config=type("Config", (), {})())
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cuda")
    model.init_states()
    model = model.to(dtype=torch.bfloat16).train()
    FullAC.Config().build(dump_folder="").apply(model)
    for layer in model.layers:
        layer.compile(backend="inductor", fullgraph=True)
    latents_TC = torch.randn(
        65536,
        1024,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positions_T3 = torch.randn(65536, 3, device="cuda", dtype=torch.float32)
    metadata = create_rae_static_varlen_metadata(
        [1024 if index % 2 == 0 else 1000 for index in range(63)],
        65536,
        device=torch.device("cuda"),
    )
    assert metadata.cu_seq_q.shape == (65,)
    output_TP = model(
        latents_TC,
        padded_positions_T3=positions_T3,
        attention_masks=metadata,
        return_padded=True,
    )
    assert output_TP.shape == (65536, 768)
    assert torch.isfinite(output_TP).all()
    output_TP.float().square().mean().backward()
    assert latents_TC.grad is not None
    assert torch.isfinite(latents_TC.grad).all()
    torch._dynamo.reset()
