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
from torchtitan.models.qwen3_vl.model import Qwen3VLModel
from torchtitan.models.rwkv7.model import RWKV7Backbone
from torchtitan.models.rwkv7.tokenizer_core import CHAT_TEMPLATE
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
        f"{package_name}.tokenizer_core",
        f"{package_name}.processor",
        f"{package_name}.processor_core",
    )
    sys.path.insert(0, str(export_dir.parent))
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        tokenizer_module = importlib.import_module(f"{package_name}.tokenizer")
        processor_module = importlib.import_module(f"{package_name}.processor")
        tokenizer_core_module = importlib.import_module(
            f"{package_name}.tokenizer_core"
        )
        processor_core_module = importlib.import_module(
            f"{package_name}.processor_core"
        )
        assert (
            tokenizer_module.RWKVTokenizerCore
            is tokenizer_core_module.RWKVTokenizerCore
        )
        assert processor_module.process_images is processor_core_module.process_images
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


class AddLayer(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta

    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + self.delta


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
    def _make_qwen_fixture(self, features_by_kind):
        dim = 4
        model = object.__new__(Qwen3VLModel)
        nn.Module.__init__(model)
        model.tok_embeddings = nn.Embedding(256, dim, device=DEVICE)
        with torch.no_grad():
            model.tok_embeddings.weight.copy_(
                torch.arange(256 * dim, device=DEVICE, dtype=torch.float32).view(
                    256, dim
                )
                / 100.0
            )
        model.layers = nn.ModuleDict(
            {"0": AddLayer(1.0), "1": AddLayer(10.0), "2": AddLayer(100.0)}
        )
        model.num_deepstack_layers = 2
        model.norm = nn.Identity()
        model.lm_head = None
        model._skip_lm_head = True
        model.freqs_cis = torch.empty(1, device=DEVICE)
        model.spatial_merge_size = 2

        def _get_vision_embeds(self, pixel_values, *, grid_thw):
            del grid_thw
            key = str(pixel_values)
            return features_by_kind[key]

        def _compute_mrope_freqs(self, *args, **kwargs):
            del args, kwargs
            return self.freqs_cis

        model._get_vision_embeds = types.MethodType(_get_vision_embeds, model)
        model._compute_mrope_freqs = types.MethodType(_compute_mrope_freqs, model)
        return model

    def _run_qwen_case(self, *, flat: bool):
        dim = 4
        image_id, video_id = 100, 101
        tokens = torch.tensor(
            [
                [1, image_id, image_id, 2, video_id, video_id, video_id, 3],
                [4, image_id, 5, video_id, video_id, 6, 7, 8],
            ],
            device=DEVICE,
        )
        image_counts = torch.tensor([2, 1], device=DEVICE)
        video_counts = torch.tensor([3, 2], device=DEVICE)

        image_main_flat = _feature_tensor([100, 101, 102], dim)
        video_main_flat = _feature_tensor([200, 201, 202, 203, 204], dim)
        image_deep0_flat = _feature_tensor([1000, 1001, 1002], dim)
        video_deep0_flat = _feature_tensor([2000, 2001, 2002, 2003, 2004], dim)
        image_deep1_flat = _feature_tensor([3000, 3001, 3002], dim)
        video_deep1_flat = _feature_tensor([4000, 4001, 4002, 4003, 4004], dim)

        if flat:
            image_features = (
                image_main_flat,
                [image_deep0_flat, image_deep1_flat],
                image_counts,
            )
            video_features = (
                video_main_flat,
                [video_deep0_flat, video_deep1_flat],
                video_counts,
            )
        else:
            image_features = (
                torch.stack(
                    [
                        torch.stack([image_main_flat[0], image_main_flat[1]]),
                        torch.stack(
                            [image_main_flat[2], torch.zeros(dim, device=DEVICE)]
                        ),
                    ]
                ),
                [
                    torch.stack(
                        [
                            torch.stack([image_deep0_flat[0], image_deep0_flat[1]]),
                            torch.stack(
                                [image_deep0_flat[2], torch.zeros(dim, device=DEVICE)]
                            ),
                        ]
                    ),
                    torch.stack(
                        [
                            torch.stack([image_deep1_flat[0], image_deep1_flat[1]]),
                            torch.stack(
                                [image_deep1_flat[2], torch.zeros(dim, device=DEVICE)]
                            ),
                        ]
                    ),
                ],
                image_counts,
            )
            video_features = (
                torch.stack(
                    [
                        torch.stack(
                            [video_main_flat[0], video_main_flat[1], video_main_flat[2]]
                        ),
                        torch.stack(
                            [
                                video_main_flat[3],
                                video_main_flat[4],
                                torch.zeros(dim, device=DEVICE),
                            ]
                        ),
                    ]
                ),
                [
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    video_deep0_flat[0],
                                    video_deep0_flat[1],
                                    video_deep0_flat[2],
                                ]
                            ),
                            torch.stack(
                                [
                                    video_deep0_flat[3],
                                    video_deep0_flat[4],
                                    torch.zeros(dim, device=DEVICE),
                                ]
                            ),
                        ]
                    ),
                    torch.stack(
                        [
                            torch.stack(
                                [
                                    video_deep1_flat[0],
                                    video_deep1_flat[1],
                                    video_deep1_flat[2],
                                ]
                            ),
                            torch.stack(
                                [
                                    video_deep1_flat[3],
                                    video_deep1_flat[4],
                                    torch.zeros(dim, device=DEVICE),
                                ]
                            ),
                        ]
                    ),
                ],
                video_counts,
            )

        model = self._make_qwen_fixture(
            {
                "image_pixels": image_features,
                "video_pixels": video_features,
            }
        )

        out = model.forward(
            tokens,
            pixel_values="image_pixels",
            pixel_values_videos="video_pixels",
            grid_thw=torch.ones(2, 3, device=DEVICE, dtype=torch.long),
            grid_thw_videos=torch.ones(2, 3, device=DEVICE, dtype=torch.long),
            special_tokens={"image_id": image_id, "video_id": video_id},
        )

        expected = model.tok_embeddings(tokens)
        expected[0, 1:3] = image_main_flat[0:2]
        expected[1, 1:2] = image_main_flat[2:3]
        expected[0, 4:7] = video_main_flat[0:3]
        expected[1, 3:5] = video_main_flat[3:5]

        expected = expected + 1.0
        expected[0, 1:3] += image_deep0_flat[0:2]
        expected[1, 1:2] += image_deep0_flat[2:3]
        expected[0, 4:7] += video_deep0_flat[0:3]
        expected[1, 3:5] += video_deep0_flat[3:5]

        expected = expected + 10.0
        expected[0, 1:3] += image_deep1_flat[0:2]
        expected[1, 1:2] += image_deep1_flat[2:3]
        expected[0, 4:7] += video_deep1_flat[0:3]
        expected[1, 3:5] += video_deep1_flat[3:5]

        expected = expected + 100.0
        torch.testing.assert_close(out, expected)

    def test_qwen_vl_padded_vision_insertion_matches_reference(self):
        self._run_qwen_case(flat=False)

    def test_qwen_vl_flat_vision_insertion_matches_reference(self):
        self._run_qwen_case(flat=True)

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
        model._vision_patch_sync_group = None
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
    def test_flat_bucketed_path_matches_padded_path_with_time_and_shape_groups(self):
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
        num_patch = grid_thw.prod(-1).to(torch.long)
        max_num_patch = 128
        real_patches = int(num_patch.sum().item())
        flat_pixels = torch.randn(real_patches, patch_dim, device=DEVICE)

        padded_pixels = flat_pixels.new_zeros(
            (grid_thw.shape[0], max_num_patch, patch_dim)
        )
        offset = 0
        for item_idx, count in enumerate(num_patch.tolist()):
            padded_pixels[item_idx, :count] = flat_pixels[offset : offset + count]
            offset += count
        bucketed_pixels = torch.cat(
            [
                flat_pixels,
                flat_pixels.new_zeros((max_num_patch - real_patches, patch_dim)),
            ],
            dim=0,
        )
        param_names = [
            "pos_embed",
            "patch_embed.proj.weight",
            "layers.0.attn.qkv.weight",
            "merger.linear_fc1.weight",
        ]

        def run_flat():
            encoder.zero_grad(set_to_none=True)
            pixel_values = bucketed_pixels.detach().clone().requires_grad_(True)
            merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
            loss = merged.float().square().mean()
            for feature in deepstack:
                loss = loss + feature.float().square().mean()
            loss.backward()
            params = dict(encoder.named_parameters())
            return (
                merged.detach(),
                [feature.detach() for feature in deepstack],
                pixel_values.grad.detach().clone(),
                {name: params[name].grad.detach().clone() for name in param_names},
            )

        def run_padded():
            encoder.zero_grad(set_to_none=True)
            pixel_values = padded_pixels.detach().clone().requires_grad_(True)
            merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
            valid_merged = []
            valid_deepstack = [[] for _ in deepstack]
            for item_idx, patches in enumerate(num_patch.tolist()):
                valid_tokens = patches // encoder.spatial_merge_unit
                valid_merged.append(merged[item_idx, :valid_tokens])
                for layer_idx, feature in enumerate(deepstack):
                    valid_deepstack[layer_idx].append(feature[item_idx, :valid_tokens])
            loss = torch.cat(valid_merged, dim=0).float().square().mean()
            for chunks in valid_deepstack:
                loss = loss + torch.cat(chunks, dim=0).float().square().mean()
            loss.backward()

            valid_grad = []
            for item_idx, patches in enumerate(num_patch.tolist()):
                valid_grad.append(pixel_values.grad[item_idx, :patches])
            params = dict(encoder.named_parameters())
            return (
                torch.cat(valid_merged, dim=0).detach(),
                [torch.cat(chunks, dim=0).detach() for chunks in valid_deepstack],
                torch.cat(valid_grad, dim=0).detach(),
                {name: params[name].grad.detach().clone() for name in param_names},
            )

        flat = run_flat()
        padded = run_padded()
        torch.testing.assert_close(flat[0], padded[0], rtol=5e-4, atol=5e-4)
        for flat_feature, padded_feature in zip(flat[1], padded[1]):
            torch.testing.assert_close(
                flat_feature, padded_feature, rtol=5e-4, atol=5e-4
            )
        torch.testing.assert_close(
            flat[2][:real_patches], padded[2], rtol=1e-3, atol=1e-3
        )
        for name in param_names:
            torch.testing.assert_close(
                flat[3][name],
                padded[3][name],
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg, name=name: f"{name} gradient mismatch:\n{msg}",
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
            pad_len = 128 - pixel_values.shape[0]
            self.assertGreaterEqual(pad_len, 0)
            pixel_values = torch.cat(
                [
                    pixel_values,
                    pixel_values.new_zeros((pad_len, pixel_values.shape[1])),
                ],
                dim=0,
            )
            merged, deepstack = encoder(pixel_values, grid_thw=grid_thw)
            self.assertEqual(merged.device.type, "cuda")
            self.assertEqual(deepstack, [])


if __name__ == "__main__":
    unittest.main()
