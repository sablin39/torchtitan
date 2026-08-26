# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from scripts.rwkv7_exporter.export_hf_model import save_remote_code_assets
from torchtitan.models.qwen3_5.vision_encoder import Qwen35VisionEncoder
from torchtitan.models.rwkv7.model import RWKV7Backbone
from torchtitan.models.rwkv7.tokenizer import CHAT_TEMPLATE
from torchtitan.models.rwkv_vl import _vl_vision_encoder_config
from torchtitan.models.rwkv_vl.model import RWKV7VLForConditionalGeneration
from transformers import BaseImageProcessor


assert torch.cuda.is_available(), "CUDA is required for VL simplification tests"
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0")


def _write_tiny_rwkv_vocab(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for byte in range(256):
            token = bytes([byte])
            token_id = byte + 1
            f.write(f"{token_id} {repr(token)} {len(token)}\n")
        for token_id, token in (
            (65530, b"<|vision_start|>"),
            (65531, b"<|vision_end|>"),
            (65532, b"<|image_pad|>"),
        ):
            f.write(f"{token_id} {repr(token)} {len(token)}\n")


def _load_exported_remote_code(tmpdir: str):
    export_dir = Path(tmpdir) / "remote"
    export_dir.mkdir()
    (export_dir / "__init__.py").write_text("", encoding="utf-8")
    save_remote_code_assets(str(export_dir), include_processor=True)
    package_name = export_dir.name
    module_names = (
        package_name,
        f"{package_name}.tokenizer",
        f"{package_name}.processor",
    )
    sys.path.insert(0, str(export_dir.parent))
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        tokenizer_module = importlib.import_module(f"{package_name}.tokenizer")
        processor_module = importlib.import_module(f"{package_name}.processor")
        return tokenizer_module.RwkvTokenizer, processor_module.ModRWKVProcessor
    finally:
        sys.path.remove(str(export_dir.parent))


class TinyImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.merge_size = 2
        self.size = {"shortest_edge": 1024, "longest_edge": 4096}
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)


class FakeRWKVLayer(nn.Module):
    def __init__(self, dim: int, delta: float):
        super().__init__()
        self.delta = delta
        self.attn = SimpleNamespace(
            value_dim=dim,
            v_proj=nn.Linear(dim, dim, bias=False, device=DEVICE),
        )

    def forward(self, hidden_states, *, v_first, cp_context=None, cu_seqlens=None):
        del cp_context, cu_seqlens
        return hidden_states + self.delta, v_first + 1


class FakeRWKVBackbone(nn.Module):
    def __init__(self, vocab_size: int, dim: int, deltas: list[float]):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim, device=DEVICE)
        with torch.no_grad():
            self.embeddings.weight.copy_(
                torch.arange(vocab_size * dim, device=DEVICE, dtype=torch.float32).view(
                    vocab_size, dim
                )
                / 100.0
            )
        self.pre_norm = nn.Identity()
        self.layers = nn.ModuleDict(
            {str(idx): FakeRWKVLayer(dim, delta) for idx, delta in enumerate(deltas)}
        )
        self.norm = nn.Identity()


def _feature_tensor(values: list[float], dim: int) -> torch.Tensor:
    return (
        torch.tensor(values, device=DEVICE, dtype=torch.float32)
        .view(-1, 1)
        .expand(-1, dim)
    )


class TestModelVisionInsertion(unittest.TestCase):
    def _make_rwkv_vl_fixture(self, features):
        dim = 4
        model = object.__new__(RWKV7VLForConditionalGeneration)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(image_token_id=100)
        fake_llm = FakeRWKVBackbone(vocab_size=256, dim=dim, deltas=[1.0, 10.0, 100.0])
        llm = object.__new__(RWKV7Backbone)
        nn.Module.__init__(llm)
        llm.embeddings = fake_llm.embeddings
        llm.pre_norm = fake_llm.pre_norm
        llm.layers = fake_llm.layers
        llm.norm = fake_llm.norm
        model.llm = llm
        model.lm_head = nn.Identity()
        model.proj = SimpleNamespace(kind="mlp")
        model._skip_lm_head = True
        model._cp_group = None
        model._trainable_roots = ("vision_encoder", "proj", "llm", "lm_head")

        def _get_vision_embeds(self, pixel_values, *, grid_thw):
            del pixel_values, grid_thw
            embeds, deepstack, num_tokens_per_item = features
            return embeds, deepstack, num_tokens_per_item, num_tokens_per_item

        model._get_vision_embeds = types.MethodType(_get_vision_embeds, model)
        return model

    def _run_rwkv_vl_case(self, *, flat: bool):
        dim = 4
        image_id = 100
        local_tokens = torch.tensor(
            [[image_id, image_id, 8, image_id, image_id]], device=DEVICE
        )
        global_tokens = torch.tensor(
            [[9, image_id, image_id, image_id, 8, image_id, image_id, 7, 6]],
            device=DEVICE,
        )
        num_tokens = torch.tensor([3, 2], device=DEVICE)
        main_flat = _feature_tensor([100, 101, 102, 200, 201], dim)
        deep0_flat = _feature_tensor([1000, 1001, 1002, 2000, 2001], dim)
        deep1_flat = _feature_tensor([3000, 3001, 3002, 4000, 4001], dim)
        if flat:
            features = (main_flat, [deep0_flat, deep1_flat], num_tokens)
        else:
            features = (
                torch.stack(
                    [
                        torch.stack([main_flat[0], main_flat[1], main_flat[2]]),
                        torch.stack(
                            [
                                main_flat[3],
                                main_flat[4],
                                torch.zeros(dim, device=DEVICE),
                            ]
                        ),
                    ]
                ),
                [
                    torch.stack(
                        [
                            torch.stack([deep0_flat[0], deep0_flat[1], deep0_flat[2]]),
                            torch.stack(
                                [
                                    deep0_flat[3],
                                    deep0_flat[4],
                                    torch.zeros(dim, device=DEVICE),
                                ]
                            ),
                        ]
                    ),
                    torch.stack(
                        [
                            torch.stack([deep1_flat[0], deep1_flat[1], deep1_flat[2]]),
                            torch.stack(
                                [
                                    deep1_flat[3],
                                    deep1_flat[4],
                                    torch.zeros(dim, device=DEVICE),
                                ]
                            ),
                        ]
                    ),
                ],
                num_tokens,
            )
        model = self._make_rwkv_vl_fixture(features)

        out = model.forward(
            local_tokens,
            pixel_values=torch.empty(5, 1, device=DEVICE),
            grid_thw=torch.ones(2, 3, device=DEVICE, dtype=torch.long),
            fla_cp_global_input_ids=global_tokens,
            fla_cp_global_start=torch.tensor(2, device=DEVICE),
        )

        expected = model.llm.embeddings(local_tokens)
        expected.view(-1, dim)[0:2] = main_flat[1:3]
        expected.view(-1, dim)[3:5] = main_flat[3:5]

        expected = expected + 1.0
        expected.view(-1, dim)[0:2] += deep0_flat[1:3]
        expected.view(-1, dim)[3:5] += deep0_flat[3:5]

        expected = expected + 10.0
        expected.view(-1, dim)[0:2] += deep1_flat[1:3]
        expected.view(-1, dim)[3:5] += deep1_flat[3:5]

        expected = expected + 100.0
        torch.testing.assert_close(out, expected)

    def test_rwkv_vl_padded_cp_overlap_insertion_matches_reference(self):
        self._run_rwkv_vl_case(flat=False)

    def test_rwkv_vl_flat_cp_overlap_insertion_matches_reference(self):
        self._run_rwkv_vl_case(flat=True)

    def test_rwkv7_backbone_forward_embeddings_matches_explicit_embed_then_layers(self):
        backbone = object.__new__(RWKV7Backbone)
        nn.Module.__init__(backbone)
        fake = FakeRWKVBackbone(vocab_size=32, dim=4, deltas=[1.0, 10.0])
        backbone.embeddings = fake.embeddings
        backbone.pre_norm = fake.pre_norm
        backbone.layers = fake.layers
        backbone.norm = fake.norm

        tokens = torch.tensor([[1, 2, 3]], device=DEVICE)
        # Two equivalent ways to invoke the backbone: explicit embedding then
        # forward_embeddings, vs. embedding inlined via forward_embeddings's
        # caller (which is what RWKV7ForCausalLM and the VL wrapper do).
        explicit = RWKV7Backbone.forward_embeddings(
            backbone,
            backbone.embeddings(tokens),
        )
        replayed = RWKV7Backbone.forward_embeddings(
            backbone,
            backbone.embeddings(tokens.clone()),
        )
        torch.testing.assert_close(explicit, replayed)


class TestCudaVisionEncoderParity(unittest.TestCase):
    def test_main_output_matches_upstream_qwen35_encoder(self):
        torch.manual_seed(2026)
        config = _vl_vision_encoder_config(
            dim=64,
            ffn_dim=128,
            n_layers=2,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=1024,
            deepstack_visual_indices=[0],
        )
        encoder = config.build()
        upstream_encoder = Qwen35VisionEncoder(config)
        encoder.to(DEVICE)
        upstream_encoder.to(DEVICE)
        encoder.init_states(buffer_device=DEVICE)
        upstream_encoder.load_state_dict(
            {
                name: value
                for name, value in encoder.state_dict().items()
                if name in upstream_encoder.state_dict()
            }
        )
        encoder.eval()
        upstream_encoder.eval()

        grid_thw = torch.tensor([[1, 2, 2], [2, 2, 4]], device=DEVICE)
        patch_dim = 3 * 2 * 16 * 16
        pixel_values = torch.randn(
            int(grid_thw.prod(-1).sum().item()),
            patch_dim,
            device=DEVICE,
        )
        with torch.no_grad():
            merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
            upstream_merged = upstream_encoder(pixel_values, grid_thw=grid_thw)

        torch.testing.assert_close(merged, upstream_merged)
        self.assertEqual(len(deepstack), 1)

    def test_exact_packed_path_supports_time_and_shape_groups(self):
        torch.manual_seed(2026)
        encoder = _vl_vision_encoder_config(
            dim=64,
            ffn_dim=128,
            n_layers=1,
            n_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=1024,
            deepstack_visual_indices=[0],
        ).build()
        encoder.to(DEVICE)
        encoder.init_states(buffer_device=DEVICE)
        encoder.train()

        grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2], [1, 2, 4]], device=DEVICE)
        patch_dim = 3 * 2 * 16 * 16
        num_patches = int(grid_thw.prod(-1).sum().item())
        pixel_values = torch.randn(
            num_patches,
            patch_dim,
            device=DEVICE,
            requires_grad=True,
        )
        param_names = [
            "pos_embed",
            "patch_embed.weight",
            "layers.0.attn.wq.weight",
            "merger.linear_fc1.weight",
        ]

        merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
        num_merged_patches = num_patches // encoder.spatial_merge_unit
        self.assertEqual(merged.shape, (num_merged_patches, 32))
        self.assertEqual(len(deepstack), 1)
        self.assertEqual(deepstack[0].shape, (num_merged_patches, 32))

        loss = merged.float().square().mean()
        for feature in deepstack:
            loss = loss + feature.float().square().mean()
        loss.backward()

        self.assertIsNotNone(pixel_values.grad)
        self.assertTrue(torch.isfinite(pixel_values.grad).all())
        params = dict(encoder.named_parameters())
        for name in param_names:
            self.assertIsNotNone(params[name].grad, name)
            self.assertTrue(torch.isfinite(params[name].grad).all(), name)

        padded_pixels = torch.cat(
            [pixel_values.detach(), pixel_values.new_zeros((1, patch_dim))],
            dim=0,
        )
        with self.assertRaisesRegex(ValueError, "grid_thw describes"):
            encoder(
                padded_pixels,
                grid_thw=grid_thw,
            )


class TestRemoteProcessor(unittest.TestCase):
    def test_remote_processor_text_policy_and_cuda_vision_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            HFRwkvTokenizer, ModRWKVProcessor = _load_exported_remote_code(tmpdir)
            vocab_file = os.path.join(tmpdir, "wr_vocab_v20230424.txt")
            _write_tiny_rwkv_vocab(vocab_file)
            hf_tok = HFRwkvTokenizer(
                vocab_file=vocab_file,
                chat_template=CHAT_TEMPLATE,
            )
            processor = ModRWKVProcessor(
                tokenizer=hf_tok,
                image_processor=TinyImageProcessor(),
            )
            output = processor(
                images=[
                    Image.new("RGB", (32, 32), color="red"),
                    Image.new("RGB", (32, 32), color="blue"),
                ],
                text="\x16User:<image><image><image>Describe.\x17",
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.vision_start_token_id), 2
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.vision_end_token_id), 2
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.image_token_id),
                int((output["image_grid_thw"].prod(-1) // 4).sum().item()),
            )

            no_insert = ModRWKVProcessor(
                tokenizer=hf_tok,
                image_processor=TinyImageProcessor(),
                auto_insert_image_tags=False,
            )
            no_insert_output = no_insert(
                images=[Image.new("RGB", (32, 32), color="green")],
                text=(
                    f"\x16User:{hf_tok.vision_start_token}"
                    f"{hf_tok.image_token}{hf_tok.vision_end_token}\x17"
                ),
            )
            self.assertEqual(
                no_insert_output["input_ids"][0].count(hf_tok.image_token_id),
                int((no_insert_output["image_grid_thw"].prod(-1) // 4).sum().item()),
            )

            with self.assertRaisesRegex(ValueError, "exceeds provided images"):
                processor(
                    images=[Image.new("RGB", (32, 32), color="purple")],
                    text=["\x16User:<image>\x17", "\x16User:<image>\x17"],
                )

            encoder = _vl_vision_encoder_config(
                dim=64,
                ffn_dim=128,
                n_layers=1,
                n_heads=4,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                out_hidden_size=32,
                num_position_embeddings=1024,
                deepstack_visual_indices=[],
            ).build()
            encoder.to(DEVICE)
            encoder.init_states(buffer_device=DEVICE)
            pixel_values = output["pixel_values"].to(DEVICE)
            grid_thw = output["image_grid_thw"].to(DEVICE)
            merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
            self.assertEqual(merged.device.type, "cuda")
            self.assertEqual(deepstack, [])


if __name__ == "__main__":
    unittest.main()
