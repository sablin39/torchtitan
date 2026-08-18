# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end training smoke for the cross-attn projector + extra_merge.

Builds a small RWKV-VL model with three DeepStack levels, sets
``proj.kind = 'cross_attn'`` and ``extra_merge_size = 2`` (i.e. processor
merge size 4 vs vision merge size 2), then runs five optimizer steps
against a synthetic batch and verifies finite/decreasing loss.

Requires CUDA (RWKV7 Triton kernels) and is skipped otherwise.
"""

from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.models.rwkv7.model import rwkv7_backbone_config
from torchtitan.models.rwkv_vl.model import (
    RWKV7VLForConditionalGeneration,
    VisualAdapter,
)


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, std=0.02)}


def _linear_cfg(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )


def _make_vision_encoder_config(
    deepstack_indices: list[int],
) -> Qwen3VLVisionEncoder.Config:
    dim = 128
    ffn_dim = 512
    patch_size = 16
    temporal_patch_size = 2
    in_channels = 3
    out_hidden_size = 256
    spatial_merge_size = 2
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    return Qwen3VLVisionEncoder.Config(
        dim=dim,
        ffn_dim=ffn_dim,
        n_layers=4,
        n_heads=4,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=1024,
        deepstack_visual_indices=deepstack_indices,
        patch_embed_proj=_linear_cfg(patch_dim, dim),
        attn_qkv=_linear_cfg(dim, dim * 3),
        attn_proj=_linear_cfg(dim, dim),
        mlp_fc1=_linear_cfg(dim, ffn_dim),
        mlp_fc2=_linear_cfg(ffn_dim, dim),
        merger_fc1=_linear_cfg(merged_hidden_size, merged_hidden_size),
        merger_fc2=_linear_cfg(merged_hidden_size, out_hidden_size),
        param_init=_POS_EMBED_INIT,
    )


def _make_model_config() -> RWKV7VLForConditionalGeneration.Config:
    vocab_size = 2048
    hidden_size = 256
    deepstack_indices = [0, 1, 2]
    return RWKV7VLForConditionalGeneration.Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        llm=rwkv7_backbone_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=4,
            num_heads=4,
            head_dim=64,
            intermediate_size=512,
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
        vision_encoder=_make_vision_encoder_config(deepstack_indices),
        proj=VisualAdapter.Config(
            encoder_dim=256,
            vision_dim=128,
            hidden_dim=512,
            project_dim=hidden_size,
            num_deepstack=len(deepstack_indices),
            norm_eps=1e-5,
            kind="cross_attn",
            language_layer_indices=(0, 1, 2),
            num_query_heads=4,
            num_key_value_heads=2,
            spatial_merge_size=2,
            extra_merge_size=2,
            # head_dim=64 here is below what the default TMA kernels handle
            # cleanly. Production cross_attn flavors (1.5B-v400M+) use
            # head_dim>=128 and keep TMA on. Pin the BLOCK / stages / warps
            # explicitly to skip autotune (which trips at small head_dim).
            kernel_options={
                "USE_TMA": False,
                "fwd_BLOCK_M": 64,
                "fwd_BLOCK_N": 64,
                "fwd_num_stages": 3,
                "fwd_num_warps": 4,
            },
        ),
        lm_head=Linear.Config(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ),
        image_token_id=2007,
        vision_start_token_id=2005,
        vision_end_token_id=2006,
        # decoupled merge: vision encoder merges by 2, processor by 4 → ratio 2
        processor_spatial_merge_size=4,
    )


def _make_batch(model, device, *, seq_len: int = 256):
    """Build a packed token + pixel batch for one forward step.

    Two images of (8, 8) patches (T=1, H=8, W=8) → 64 patches each → 128
    total patches (satisfies the vision FlexAttention TMA constraint).
    With processor_spatial_merge_size=8, each image contributes
    64 // 64 = 1 image_pad token to the text stream.
    """
    image_token_id = model.config.image_token_id
    gen = torch.Generator(device=device).manual_seed(7)
    tokens = torch.randint(
        2, 2000, (1, seq_len), dtype=torch.long, device=device, generator=gen
    )
    # processor_merge=4 → 4 image_pad tokens per (8,8) image.
    tokens[0, 16:20] = image_token_id
    tokens[0, 32:36] = image_token_id
    grid_thw = torch.tensor([[1, 8, 8], [1, 8, 8]], dtype=torch.long, device=device)
    total_patches = int(grid_thw.prod(-1).sum().item())
    pixel_values = torch.randn(
        total_patches, 3 * 2 * 16 * 16, device=device, generator=gen
    )
    # Labels: shift by 1 for next-token; ignore-index where we don't care.
    labels = tokens.clone()
    return tokens, pixel_values, grid_thw, labels


class TestCrossAttnTrainingLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for RWKV7 Triton kernels")
        cls.device = torch.device("cuda")

    def test_cross_attn_extra_merge_training_steps(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        cfg = _make_model_config()
        model = cfg.build().to(self.device).to(torch.bfloat16)
        model.train()
        # Trainable: projector + lm_head. Vision encoder and LLM frozen to
        # isolate projector training behavior.
        for p in model.vision_encoder.parameters():
            p.requires_grad_(False)
        for p in model.llm.parameters():
            p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=1e-3)

        tokens, pixel_values, grid_thw, labels = _make_batch(model, self.device)

        losses = []
        for step in range(5):
            opt.zero_grad()
            logits = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
            # Simple next-token CE on a few positions.
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
            )
            loss.backward()
            if step == 0:
                gradient_parameters = {
                    "seed_q_proj": model.proj.seed_q_proj.weight,
                    "rwkv_q_proj": model.proj.rwkv_q_proj.weight,
                    "shared_k_proj": model.proj.k_proj.weight,
                    "shared_v_proj": model.proj.v_proj.weight,
                    "shared_o_proj": model.proj.o_proj.weight,
                    "seed_query_norm": model.proj.seed_query_norm.weight,
                    "seed_output_norm": model.proj.seed_output_norm.weight,
                    "query_norm": model.proj.query_norms[0].weight,
                    "query_gate_proj": model.proj.query_gate_projs[0].weight,
                    "memory_norm": model.proj.memory_norms[0].weight,
                }
                for name, parameter in gradient_parameters.items():
                    self.assertIsNotNone(parameter.grad, f"{name} has no gradient")
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all(),
                        f"{name} gradient is not finite",
                    )
            opt.step()
            losses.append(float(loss.detach()))
            self.assertTrue(torch.isfinite(loss).item(), f"step {step} loss not finite")
        print(f"cross_attn+extra_merge=2 losses: {losses}")

        # Sanity: at least one of the later steps should have a lower loss
        # than step 0 (random init may bounce, but trend should be down with
        # this LR over five steps).
        self.assertLess(min(losses[1:]), losses[0], f"loss did not decrease: {losses}")
        self.assertEqual(len(model.vision_encoder.deepstack_merger_list), 0)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.vision_encoder.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
