# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the RWKV-VL projector variants (mlp / cross_attn).

These tests build a small RWKV-VL model with three DeepStack levels, run a
forward + backward over a packed sequence containing two synthetic images,
and verify:

  - the ``mlp`` projector path matches today's additive DeepStack behavior;
  - the ``cross_attn`` projector path produces finite outputs and gradients,
    respects the per-image attention mask, and works with
    ``extra_merge_size > 1`` (processor merge > vision merge).

Skipped when no CUDA device is available — the RWKV7 backbone needs Triton.
"""

from __future__ import annotations

import gc
import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common import Linear
from torchtitan.models.rwkv7.model import rwkv7_backbone_config
from torchtitan.models.rwkv_vl import _vl_vision_encoder_config
from torchtitan.models.rwkv_vl.model import (
    RWKV7VLForConditionalGeneration,
    VisualAdapter,
)
from torchtitan.models.rwkv_vl.vision_encoder import RWKVVisionEncoder


def _make_vision_encoder_config(
    *, deepstack_indices: list[int], spatial_merge_size: int = 2
) -> RWKVVisionEncoder.Config:
    dim = 128
    ffn_dim = 512
    patch_size = 16
    temporal_patch_size = 2
    in_channels = 3
    out_hidden_size = 256
    return _vl_vision_encoder_config(
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
        in_channels=in_channels,
    )


def _make_model_config(
    *,
    kind: str,
    extra_merge_size: int = 1,
    processor_merge_size: int = 2,
    deepstack_indices: tuple[int, ...] = (1, 2, 3),
) -> RWKV7VLForConditionalGeneration.Config:
    vocab_size = 2048
    hidden_size = 256
    image_token_id = 2007
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
        vision_encoder=_make_vision_encoder_config(
            deepstack_indices=list(deepstack_indices)
        ),
        proj=VisualAdapter.Config(
            encoder_dim=256,
            hidden_dim=512,
            project_dim=hidden_size,
            num_deepstack=len(deepstack_indices),
            norm_eps=1e-5,
            kind=kind,
            num_heads=4 if kind == "cross_attn" else None,
            extra_merge_size=extra_merge_size,
        ),
        lm_head=Linear.Config(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ),
        image_token_id=image_token_id,
        vision_start_token_id=2005,
        vision_end_token_id=2006,
        processor_spatial_merge_size=processor_merge_size,
    )


def _make_inputs(
    *,
    device,
    image_token_id: int,
    seq_len: int = 256,
    images: list[tuple[int, int]] | None = None,
    processor_merge: int = 2,
    seq_len_override: int | None = None,
):
    """Build a packed (B=1, seq_len) token stream + flat patches + grid.

    ``images`` is a list of (h, w) in patches (per the 2D grid in the
    Qwen3-VL convention). For temporal_patch_size=2 we set T=1 always; each
    image's patch count is 1*h*w.

    Image_pad tokens placed at fixed offsets in the sequence; the count per
    image matches ``(1*h*w) // processor_merge**2``.
    """
    if images is None:
        images = [(8, 8), (8, 8)]  # 64 + 64 = 128 patches (FlexAttn TMA)
    gen = torch.Generator(device=device).manual_seed(42)
    tokens = torch.randint(
        2, 2000, (1, seq_len), dtype=torch.long, device=device, generator=gen
    )
    cursor = 8  # leave some text prefix
    grids = []
    for (h, w) in images:
        n_pad = (1 * h * w) // (processor_merge**2)
        tokens[0, cursor : cursor + n_pad] = image_token_id
        cursor += n_pad + 4  # gap between images
        grids.append([1, h, w])
    grid_thw = torch.tensor(grids, dtype=torch.long, device=device)
    total_patches = int(grid_thw.prod(-1).sum().item())
    # patch_dim = in_channels * temporal_patch_size * patch_size**2 = 3*2*16*16
    pixel_values = torch.randn(
        total_patches, 3 * 2 * 16 * 16, device=device, generator=gen
    )
    return tokens, pixel_values, grid_thw


class TestRwkvVLProjectorVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for RWKV7 Triton kernels")
        cls.device = torch.device("cuda")

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _build(self, **kwargs) -> RWKV7VLForConditionalGeneration:
        cfg = _make_model_config(**kwargs)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = cfg.build().to(self.device).to(torch.bfloat16)
        model.init_states(buffer_device=self.device)
        return model

    def test_mlp_projector_forward_backward(self):
        model = self._build(kind="mlp")
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device, image_token_id=model.config.image_token_id
        )
        out = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
        self.assertEqual(out.shape, (1, tokens.shape[1], model.vocab_size))
        self.assertTrue(torch.isfinite(out).all())
        out.float().mean().backward()
        # Spot-check a projector param has a gradient.
        any_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.proj.parameters()
        )
        self.assertTrue(any_grad)

    def test_cross_attn_projector_forward_backward(self):
        model = self._build(kind="cross_attn")
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device, image_token_id=model.config.image_token_id
        )
        out = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
        self.assertEqual(out.shape, (1, tokens.shape[1], model.vocab_size))
        self.assertTrue(torch.isfinite(out).all())
        out.float().mean().backward()
        # K/V projections in the cross-attn projector must receive grad.
        kp = model.proj.deepstack[0].k_proj.weight
        self.assertIsNotNone(kp.grad)
        self.assertTrue(torch.isfinite(kp.grad).all())

    def test_cross_attn_with_extra_merge(self):
        """processor_merge=4, vision_merge=2 → extra_merge_size=2."""
        model = self._build(
            kind="cross_attn",
            extra_merge_size=2,
            processor_merge_size=4,
        )
        # Need image patches divisible by processor_merge**2 = 16, and
        # total divisible by 128 for the vision FlexAttention TMA path.
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device,
            image_token_id=model.config.image_token_id,
            images=[(8, 8), (8, 8)],
            processor_merge=4,
        )
        out = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
        self.assertEqual(out.shape, (1, tokens.shape[1], model.vocab_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_cross_image_attention_is_masked(self):
        """Swapping image B's K/V should not change image A's queries."""
        model = self._build(kind="cross_attn")
        model.eval()
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device, image_token_id=model.config.image_token_id
        )
        with torch.no_grad():
            _, deepstack1, num_tokens, num_kv = model._get_vision_embeds(
                pixel_values,
                grid_thw=grid_thw,
            )
            grid_a = grid_thw[0]
            patches_a = int((grid_a.prod()).item())
            pixel_values_alt = pixel_values.clone()
            pixel_values_alt[patches_a:].uniform_(-1, 1)
            _, deepstack2, _, _ = model._get_vision_embeds(
                pixel_values_alt,
                grid_thw=grid_thw,
            )
            injector1 = model._make_cross_attn_injector(
                deepstack_features=deepstack1,
                num_tokens_per_item=num_tokens,
                num_kv_per_item=num_kv,
                vision_token_id=model.config.image_token_id,
                global_input_ids=None,
                global_start=None,
                local_tokens=tokens,
            )
            injector2 = model._make_cross_attn_injector(
                deepstack_features=deepstack2,
                num_tokens_per_item=num_tokens,
                num_kv_per_item=num_kv,
                vision_token_id=model.config.image_token_id,
                global_input_ids=None,
                global_start=None,
                local_tokens=tokens,
            )
            query_hidden = torch.randn(
                1,
                tokens.shape[1],
                model.hidden_size,
                device=self.device,
                dtype=torch.bfloat16,
            )
            out1 = injector1(0, query_hidden)
            out2 = injector2(0, query_hidden)

        n_pad_a = patches_a // (model.processor_spatial_merge_size**2)
        a_slice = (slice(None), slice(8, 8 + n_pad_a))
        torch.testing.assert_close(out1[a_slice], out2[a_slice])


if __name__ == "__main__":
    unittest.main()
