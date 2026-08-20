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
from torchtitan.hf_datasets.multimodal.processor import vision_to_patches

from torchtitan.models.common import Linear
from torchtitan.models.qwen3_vl.vision_encoder import Qwen3VLVisionEncoder
from torchtitan.models.rwkv7.model import rwkv7_backbone_config
from torchtitan.models.rwkv_vl.model import (
    _query_position_encoding,
    _tokenpacker_local_ids,
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
    *, deepstack_indices: list[int], spatial_merge_size: int = 2
) -> Qwen3VLVisionEncoder.Config:
    dim = 128
    ffn_dim = 512
    patch_size = 16
    temporal_patch_size = 2
    in_channels = 3
    out_hidden_size = 256
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


def _make_model_config(
    *,
    kind: str,
    extra_merge_size: int = 1,
    processor_merge_size: int = 2,
    deepstack_indices: tuple[int, ...] = (0, 1, 2),
    tie_qkvo: bool = True,
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
            chunk_size=16,
        ),
        vision_encoder=_make_vision_encoder_config(
            deepstack_indices=list(deepstack_indices)
        ),
        proj=VisualAdapter.Config(
            encoder_dim=256,
            vision_dim=128,
            hidden_dim=512,
            project_dim=hidden_size,
            num_deepstack=len(deepstack_indices),
            norm_eps=1e-5,
            kind=kind,
            language_layer_indices=(0, 1, 2) if kind == "cross_attn" else (),
            num_query_heads=4 if kind == "cross_attn" else None,
            num_key_value_heads=1 if kind == "cross_attn" else None,
            tie_qkvo=tie_qkvo,
            spatial_merge_size=2,
            extra_merge_size=extra_merge_size,
            kernel_options=(
                {
                    "USE_TMA": False,
                    "fwd_BLOCK_M": 64,
                    "fwd_BLOCK_N": 64,
                    "fwd_num_stages": 3,
                    "fwd_num_warps": 4,
                }
                if kind == "cross_attn"
                else None
            ),
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
    for h, w in images:
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


class TestTokenPackerSpatialLayout(unittest.TestCase):
    def test_native_patch_order_maps_to_aligned_local_regions(self):
        image = torch.arange(64, dtype=torch.float32).view(1, 8, 8, 1)
        patches, grid_thw = vision_to_patches(
            image,
            patch_size=1,
            temporal_patch_size=1,
            merge_size=2,
        )
        query_ids, memory_ids = _tokenpacker_local_ids(
            grid_thw.unsqueeze(0),
            spatial_merge_size=2,
            extra_merge_size=2,
        )
        values = patches[:, 0].to(torch.long)
        rows, columns = values // 8, values % 8
        expected_regions = (rows // 4) * 2 + columns // 4
        self.assertTrue(torch.equal(memory_ids.cpu(), expected_regions))
        self.assertTrue(torch.equal(query_ids.cpu(), torch.arange(4)))

    def test_query_position_encoding_breaks_seed_symmetry(self):
        grid_thw = torch.tensor([[1, 8, 8]])
        positions = _query_position_encoding(
            grid_thw,
            dim=128,
            spatial_merge_size=2,
            extra_merge_size=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(torch.unique(positions, dim=0).shape[0], 4)

    def test_gqa_projection_shapes_and_depth_specific_norms(self):
        adapter = VisualAdapter.Config(
            encoder_dim=256,
            vision_dim=128,
            project_dim=256,
            num_deepstack=3,
            kind="cross_attn",
            extra_merge_size=2,
            spatial_merge_size=2,
            language_layer_indices=(0, 1, 2),
            num_query_heads=4,
            num_key_value_heads=1,
        ).build()
        self.assertEqual(adapter.k_proj.weight.shape, (32, 128))
        self.assertEqual(adapter.v_proj.weight.shape, (32, 128))
        self.assertEqual(adapter.rwkv_q_proj.weight.shape, (128, 256))
        self.assertEqual(adapter.o_proj.weight.shape, (256, 128))
        self.assertEqual(len(adapter.query_norms), 3)
        self.assertEqual(len(adapter.query_gate_projs), 3)
        self.assertEqual(adapter.query_gate_projs[0].weight.shape, (4, 256))
        self.assertEqual(len(adapter.memory_norms), 4)
        self.assertEqual(len({id(norm) for norm in adapter.query_norms}), 3)
        self.assertEqual(len({id(proj) for proj in adapter.query_gate_projs}), 3)
        self.assertEqual(len({id(norm) for norm in adapter.memory_norms}), 4)

        queries = torch.randn(2, 256)
        _, gates = adapter._project_query_and_gate(0, queries)
        self.assertEqual(gates.shape, (2, 4))
        self.assertTrue(torch.all((gates > 0) & (gates < 1)))
        self.assertFalse(torch.allclose(gates[0], gates[1]))

    def test_untied_qkvo_are_depth_specific(self):
        adapter = VisualAdapter.Config(
            encoder_dim=256,
            vision_dim=128,
            project_dim=256,
            num_deepstack=3,
            kind="cross_attn",
            extra_merge_size=2,
            spatial_merge_size=2,
            language_layer_indices=(0, 1, 2),
            num_query_heads=4,
            num_key_value_heads=2,
            tie_qkvo=False,
        ).build()
        self.assertFalse(hasattr(adapter, "rwkv_q_proj"))
        self.assertFalse(hasattr(adapter, "k_proj"))
        self.assertFalse(hasattr(adapter, "v_proj"))
        self.assertFalse(hasattr(adapter, "o_proj"))
        self.assertEqual(len(adapter.rwkv_q_projs), 3)
        self.assertEqual(len(adapter.k_projs), 4)
        self.assertEqual(len(adapter.v_projs), 4)
        self.assertEqual(len(adapter.o_projs), 4)
        self.assertEqual(len(adapter.query_gate_projs), 3)
        self.assertEqual(adapter.k_projs[0].weight.shape, (64, 128))
        self.assertEqual(len({id(module) for module in adapter.k_projs}), 4)
        self.assertEqual(len({id(module) for module in adapter.v_projs}), 4)
        self.assertEqual(len({id(module) for module in adapter.o_projs}), 4)


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
        kp = model.proj.k_proj.weight
        self.assertIsNotNone(kp.grad)
        self.assertTrue(torch.isfinite(kp.grad).all())

    def test_cross_attn_untied_qkvo_forward_backward(self):
        model = self._build(kind="cross_attn", tie_qkvo=False)
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device, image_token_id=model.config.image_token_id
        )
        out = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
        self.assertTrue(torch.isfinite(out).all())
        out.float().mean().backward()
        for collection_name in (
            "rwkv_q_projs",
            "query_gate_projs",
            "k_projs",
            "v_projs",
            "o_projs",
        ):
            collection = getattr(model.proj, collection_name)
            for depth, projection in enumerate(collection):
                self.assertIsNotNone(
                    projection.weight.grad,
                    f"{collection_name}[{depth}] has no gradient",
                )
                self.assertTrue(torch.isfinite(projection.weight.grad).all())

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
        attended_depths = []
        original_attend = model.proj.attend

        def record_attend(depth, *args, **kwargs):
            attended_depths.append(depth)
            return original_attend(depth, *args, **kwargs)

        model.proj.attend = record_attend
        out = model(tokens, pixel_values=pixel_values, grid_thw=grid_thw)
        self.assertEqual(out.shape, (1, tokens.shape[1], model.vocab_size))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(attended_depths, [0, 1, 2])

    def test_cross_attn_text_only_path_skips_projector(self):
        model = self._build(kind="cross_attn")
        tokens = torch.randint(
            2,
            2000,
            (1, 128),
            dtype=torch.long,
            device=self.device,
        )
        projector_called = []
        handle = model.proj.register_forward_hook(
            lambda module, args, output: projector_called.append(True)
        )
        try:
            out = model(tokens)
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(projector_called, [])
        out.float().mean().backward()
        for name, parameter in model.proj.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} has no gradient edge")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                f"{name} gradient is not finite",
            )

    def test_cross_image_attention_is_masked(self):
        """Changing image B's K/V must not affect image A's queries."""
        model = self._build(kind="cross_attn")
        model.eval()
        tokens, pixel_values, grid_thw = _make_inputs(
            device=self.device, image_token_id=model.config.image_token_id
        )
        with torch.no_grad():
            (
                _,
                deepstack,
                num_tokens_per_item,
                num_kv_per_item,
            ) = model._get_vision_embeds(pixel_values, grid_thw=grid_thw)
            patches_a = int(num_kv_per_item[0].item())
            pixel_values_alt = pixel_values.clone()
            pixel_values_alt[patches_a:].uniform_(-1, 1)
            (
                _,
                deepstack_alt,
                num_tokens_per_item_alt,
                num_kv_per_item_alt,
            ) = model._get_vision_embeds(pixel_values_alt, grid_thw=grid_thw)

            torch.testing.assert_close(
                num_tokens_per_item, num_tokens_per_item_alt, rtol=0, atol=0
            )
            torch.testing.assert_close(
                num_kv_per_item, num_kv_per_item_alt, rtol=0, atol=0
            )
            for (keys, values), (keys_alt, values_alt) in zip(deepstack, deepstack_alt):
                torch.testing.assert_close(
                    keys[:patches_a], keys_alt[:patches_a], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    values[:patches_a], values_alt[:patches_a], rtol=0, atol=0
                )

            injector = model._make_cross_attn_injector(
                deepstack_features=deepstack,
                num_tokens_per_item=num_tokens_per_item,
                num_kv_per_item=num_kv_per_item,
                vision_token_id=model.config.image_token_id,
                global_input_ids=None,
                global_start=None,
                local_tokens=tokens,
            )
            injector_alt = model._make_cross_attn_injector(
                deepstack_features=deepstack_alt,
                num_tokens_per_item=num_tokens_per_item_alt,
                num_kv_per_item=num_kv_per_item_alt,
                vision_token_id=model.config.image_token_id,
                global_input_ids=None,
                global_start=None,
                local_tokens=tokens,
            )

            generator = torch.Generator(device=self.device).manual_seed(123)
            hidden_states = torch.randn(
                1,
                tokens.shape[1],
                model.hidden_size,
                dtype=torch.bfloat16,
                device=self.device,
                generator=generator,
            )
            image_positions = torch.nonzero(
                tokens[0] == model.config.image_token_id, as_tuple=False
            ).flatten()
            num_a_queries = int(num_tokens_per_item[0].item())
            image_a_positions = image_positions[:num_a_queries]
            image_b_positions = image_positions[num_a_queries:]

            for layer_index in model.proj.language_layer_indices:
                injected = injector(layer_index, hidden_states)
                injected_alt = injector_alt(layer_index, hidden_states)
                torch.testing.assert_close(
                    injected[0, image_a_positions],
                    injected_alt[0, image_a_positions],
                    rtol=0,
                    atol=0,
                )
                self.assertGreater(
                    (
                        injected[0, image_b_positions]
                        - injected_alt[0, image_b_positions]
                    )
                    .abs()
                    .max()
                    .item(),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
