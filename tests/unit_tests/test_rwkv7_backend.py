# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch

from torchtitan.distributed.context_parallel import _build_flattened_cu_seqlens
from torchtitan.config.manager import ConfigManager
from torchtitan.models.rwkv7 import model_registry as rwkv7_model_registry
from torchtitan.models.rwkv7.model import _token_shift_varlen_eager
from torchtitan.models.rwkv7.state_dict_adapter import RWKV7StateDictAdapter
from torchtitan.models.rwkv7.tokenizer import RwkvTokenizer
import torchtitan.models.rwkv_vl.config_registry as rwkv_vl_config_registry
from torchtitan.models.rwkv_vl import (
    model_registry as rwkv_vl_model_registry,
    rwkv_vl_configs,
)
from torchtitan.models.rwkv_vl.model import VisualAdapter
import torchtitan.models.rwkv_vl.parallelize as rwkv_vl_parallelize
from torchtitan.models.rwkv_vl.state_dict_adapter import RWKVVLStateDictAdapter
from torchtitan.models.rwkv_vl.tokenizer import RwkvVLMultiModalTokenizer


def _write_tiny_rwkv_vocab(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        token_id = 1
        for byte in range(256):
            token = bytes([byte])
            f.write(f"{token_id} {repr(token)} {len(token)}\n")
            token_id += 1
        for token_id, token in (
            (65530, b"<|vision_start|>"),
            (65531, b"<|vision_end|>"),
            (65532, b"<|image_pad|>"),
        ):
            f.write(f"{token_id} {repr(token)} {len(token)}\n")


class TestRWKV7Backend(unittest.TestCase):
    def test_rwkv7_debugmodel_builds_and_satisfies_module_protocol(self):
        spec = rwkv7_model_registry("debugmodel")
        with torch.device("meta"):
            model = spec.model.build()
        model.verify_module_protocol()
        self.assertEqual(model.vocab_size, 2048)

    def test_rwkv7_first_layer_state_lives_on_backbone(self):
        spec = rwkv7_model_registry("debugmodel")
        with torch.device("meta"):
            model = spec.model.build()

        keys = set(model.state_dict())
        self.assertIn("llm.pre_norm.weight", keys)
        self.assertIn("llm.pre_norm.bias", keys)
        self.assertNotIn("llm.layers.0.pre_norm.weight", keys)
        self.assertNotIn("llm.layers.1.pre_norm.weight", keys)
        self.assertFalse(
            any(key.startswith("llm.layers.0.attn.v_lora.") for key in keys)
        )
        self.assertIn("llm.layers.1.attn.v_lora.lora.0.weight", keys)

        hidden_states = torch.empty(2, 3, 256, device="meta")
        v_first = hidden_states.new_zeros(
            *hidden_states.shape[:-1],
            model.llm.layers["0"].attn.value_dim,
        )
        self.assertEqual(tuple(v_first.shape), (2, 3, 256))

    def test_rwkv7_backbone_loads_legacy_layer0_pre_norm_key(self):
        spec = rwkv7_model_registry("debugmodel")
        model = spec.model.build()
        old_weight = torch.full_like(model.llm.pre_norm.weight, 2.0)
        old_bias = torch.full_like(model.llm.pre_norm.bias, 3.0)

        missing, unexpected = model.load_state_dict(
            {
                "llm.layers.0.pre_norm.weight": old_weight,
                "llm.layers.0.pre_norm.bias": old_bias,
            },
            strict=False,
        )

        self.assertNotIn("llm.pre_norm.weight", missing)
        self.assertNotIn("llm.pre_norm.bias", missing)
        self.assertNotIn("llm.layers.0.pre_norm.weight", unexpected)
        self.assertTrue(torch.equal(model.llm.pre_norm.weight, old_weight))
        self.assertTrue(torch.equal(model.llm.pre_norm.bias, old_bias))

    def test_rwkv7_varlen_token_shift_matches_reference_and_grad(self):
        def ref_token_shift(
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> torch.Tensor:
            shifted = torch.empty_like(x)
            shifted[:, 0, :].zero_()
            shifted[:, 1:, :].copy_(x[:, :-1, :])
            shifted.index_fill_(1, cu_seqlens[:-1], 0)
            return shifted - x

        for cu_seqlens in (
            torch.tensor([0, 2, 5, 8]),
            torch.arange(0, 9),
        ):
            with self.subTest(cu_seqlens=cu_seqlens.tolist()):
                x = torch.randn(1, 8, 3, dtype=torch.float32, requires_grad=True)
                ref_x = x.detach().clone().requires_grad_()
                grad = torch.randn_like(x)

                y = _token_shift_varlen_eager(x, cu_seqlens)
                ref_y = ref_token_shift(ref_x, cu_seqlens)
                self.assertTrue(torch.allclose(y, ref_y))

                y.backward(grad)
                ref_y.backward(grad)
                self.assertTrue(torch.allclose(x.grad, ref_x.grad))

    def test_rwkv_vl_config_builds_and_satisfies_module_protocol(self):
        spec = rwkv_vl_model_registry("debugmodel")
        with torch.device("meta"):
            model = spec.model.build()
        model.verify_module_protocol()
        self.assertEqual(model.config.image_token_id, 2007)
        self.assertEqual(model.vision_encoder.config.dim, 128)

    def test_rwkv_vl_production_flavors_are_explicit(self):
        self.assertIn("0.4B-v100M", rwkv_vl_configs)
        self.assertIn("1.5B-v100M", rwkv_vl_configs)
        self.assertIn("1.5B-v400M", rwkv_vl_configs)
        self.assertNotIn("0.4B", rwkv_vl_configs)
        self.assertNotIn("0.4B-v400M", rwkv_vl_configs)
        self.assertFalse(hasattr(rwkv_vl_config_registry, "rwkv_vl_0_4b_chat"))

    def test_rwkv_vl_production_projector_shapes(self):
        cases = {
            "0.4B-v100M": (1024, 1024, 0),
            "1.5B-v100M": (1024, 2048, 0),
            "1.5B-v400M": (2048, 2048, 3),
        }
        for flavor, (encoder_dim, project_dim, num_deepstack) in cases.items():
            with self.subTest(flavor=flavor):
                spec = rwkv_vl_model_registry(flavor)
                self.assertEqual(spec.model.proj.encoder_dim, encoder_dim)
                self.assertEqual(spec.model.proj.project_dim, project_dim)
                self.assertEqual(spec.model.proj.num_deepstack, num_deepstack)
                with torch.device("meta"):
                    model = spec.model.build()
                self.assertEqual(len(model.proj.deepstack), num_deepstack)

    def test_visual_adapter_projects_each_stream_without_identity_layout(self):
        with torch.device("meta"):
            adapter = VisualAdapter.Config(
                encoder_dim=2048,
                project_dim=2048,
                hidden_dim=8192,
                num_deepstack=3,
            ).build()
            main, deepstack = adapter(
                torch.empty(2, 2048, device="meta"),
                [torch.empty(2, 2048, device="meta") for _ in range(3)],
            )
        self.assertEqual(tuple(main.shape), (2, 2048))
        self.assertEqual([tuple(t.shape) for t in deepstack], [(2, 2048)] * 3)
        keys = set(adapter.state_dict())
        self.assertIn("main.pre_norm.weight", keys)
        self.assertIn("deepstack.0.pre_norm.weight", keys)
        self.assertNotIn("pre_norm.weight", keys)

    def test_rwkv_vl_module_lrs_freeze_zero_lr_roots(self):
        spec = rwkv_vl_model_registry("debugmodel")
        spec.model.root_lrs = {
            "vision_encoder": 0.0,
            "proj": 1e-4,
            "llm": 0.0,
            "lm_head": 0.0,
        }
        with torch.device("meta"):
            model = spec.model.build()

        self.assertTrue(all(p.requires_grad for p in model.proj.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in model.vision_encoder.parameters())
        )
        self.assertTrue(all(not p.requires_grad for p in model.llm.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.lm_head.parameters()))

    def test_rwkv_vl_module_lrs_lm_head_follows_llm_by_default(self):
        cfg = ConfigManager().parse_args(
            [
                "--module",
                "rwkv_vl",
                "--config",
                "rwkv_vl_debugmodel_chat",
                "--optimizer.lr",
                "1e-5",
                "--module-lrs.vision-encoder",
                "0",
                "--module-lrs.proj",
                "0",
                "--module-lrs.llm",
                "2e-5",
            ]
        )
        spec = cfg.model_spec
        spec.model.update_from_config(trainer_config=cfg)
        self.assertEqual(
            spec.model.root_lrs,
            {
                "vision_encoder": 0.0,
                "proj": 0.0,
                "llm": 2e-5,
                "lm_head": 2e-5,
            },
        )
        with torch.device("meta"):
            model = spec.model.build()

        self.assertTrue(all(p.requires_grad for p in model.llm.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.lm_head.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in model.vision_encoder.parameters())
        )
        self.assertTrue(all(not p.requires_grad for p in model.proj.parameters()))

    def test_rwkv_vl_module_lrs_cli_configures_optimizer_groups(self):
        cfg = ConfigManager().parse_args(
            [
                "--module",
                "rwkv_vl",
                "--config",
                "rwkv_vl_debugmodel_chat",
                "--optimizer.lr",
                "1e-5",
                "--module-lrs.vision-encoder",
                "0",
                "--module-lrs.proj",
                "1e-4",
                "--module-lrs.llm",
                "1e-5",
            ]
        )
        self.assertEqual(cfg.module_lrs.vision_encoder, 0.0)
        self.assertEqual(cfg.module_lrs.proj, 1e-4)
        self.assertEqual(cfg.module_lrs.llm, 1e-5)
        self.assertIsNone(cfg.module_lrs.lm_head)

        cfg.model_spec.model.update_from_config(trainer_config=cfg)
        groups = {
            group.pattern: group.lr_multiplier
            for group in cfg.optimizer.param_groups
        }
        self.assertNotIn(r"^vision_encoder\.", groups)
        self.assertEqual(groups[r"^proj\."], 10.0)
        self.assertEqual(groups[r"^llm\."], 1.0)
        self.assertEqual(groups[r"^lm_head\."], 1.0)

    def test_rwkv_vl_backbone_chunk_size_cli_updates_model_config(self):
        cfg = ConfigManager().parse_args(
            [
                "--module",
                "rwkv_vl",
                "--config",
                "rwkv_vl_debugmodel_chat",
                "--backbone-chunk-size",
                "32",
            ]
        )
        self.assertEqual(cfg.backbone_chunk_size, 32)
        self.assertEqual(cfg.model_spec.model.llm.chunk_size, 64)

        cfg.model_spec.model.update_from_config(trainer_config=cfg)
        self.assertEqual(cfg.model_spec.model.llm.chunk_size, 32)
        with torch.device("meta"):
            model = cfg.model_spec.model.build()
        self.assertTrue(
            all(block.attn.chunk_size == 32 for block in model.llm.layers.values())
        )

    def test_rwkv_vl_backbone_chunk_size_rejects_bad_values(self):
        cfg = ConfigManager().parse_args(
            [
                "--module",
                "rwkv_vl",
                "--config",
                "rwkv_vl_debugmodel_chat",
                "--backbone-chunk-size",
                "8",
            ]
        )
        with self.assertRaisesRegex(ValueError, "at least 16"):
            cfg.model_spec.model.update_from_config(trainer_config=cfg)

    def test_rwkv_vl_fsdp_skips_frozen_roots(self):
        spec = rwkv_vl_model_registry("debugmodel")
        spec.model.root_lrs = {
            "vision_encoder": 0.0,
            "proj": 1e-5,
            "llm": 1e-5,
            "lm_head": 1e-5,
        }
        with torch.device("meta"):
            model = spec.model.build()

        frozen_params = {p for p in model.parameters() if not p.requires_grad}
        self.assertGreater(len(frozen_params), 0)
        self.assertTrue(
            all(param in frozen_params for param in model.vision_encoder.parameters())
        )
        self.assertFalse(
            any(param in frozen_params for param in model.proj.parameters())
        )

        module_names = {
            id(model): "model",
            id(model.vision_encoder): "vision_encoder",
            id(model.proj): "proj",
            id(model.llm.embeddings): "llm.embeddings",
            id(model.llm.pre_norm): "llm.pre_norm",
            id(model.llm.norm): "llm.norm",
            id(model.lm_head): "lm_head",
            id(model.llm): "llm",
        }
        module_names.update(
            {id(block): f"llm.layers.{idx}" for idx, block in model.llm.layers.items()}
        )

        fully_shard_calls = []
        original_fully_shard = rwkv_vl_parallelize.fully_shard
        original_disable = rwkv_vl_parallelize.disable_fsdp_gradient_division

        def fake_fully_shard(module, **kwargs):
            fully_shard_calls.append((module_names[id(module)], kwargs))

        try:
            rwkv_vl_parallelize.fully_shard = fake_fully_shard
            rwkv_vl_parallelize.disable_fsdp_gradient_division = lambda model: None
            rwkv_vl_parallelize.apply_fsdp(
                model,
                dp_mesh=object(),
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        finally:
            rwkv_vl_parallelize.fully_shard = original_fully_shard
            rwkv_vl_parallelize.disable_fsdp_gradient_division = original_disable

        sharded_modules = [name for name, _ in fully_shard_calls]
        self.assertNotIn("vision_encoder", sharded_modules)
        self.assertIn("proj", sharded_modules)
        self.assertIn("llm.embeddings", sharded_modules)
        self.assertIn("llm.pre_norm", sharded_modules)
        self.assertIn("lm_head", sharded_modules)
        self.assertIn("llm", sharded_modules)
        self.assertIn("model", sharded_modules)
        frozen_param_ids = {id(param) for param in frozen_params}
        for _, kwargs in fully_shard_calls:
            self.assertEqual(
                {id(param) for param in kwargs["ignored_params"]},
                frozen_param_ids,
            )

    def test_rwkv7_state_dict_adapter_maps_llm_prefix(self):
        spec = rwkv7_model_registry("debugmodel")
        adapter = RWKV7StateDictAdapter(spec.model, hf_assets_path=None)
        tensor = torch.empty(4, 8)
        out = adapter.from_hf(
            {
                "model.llm.embeddings.weight": tensor,
                "model.llm.layers.0.pre_norm.weight": tensor,
                "model.llm.layers.0.attn.x_r": tensor,
                "lm_head.weight": tensor,
                "model.encoder.patch_embed.proj.weight": tensor,
            }
        )
        self.assertIn("llm.embeddings.weight", out)
        self.assertIn("llm.pre_norm.weight", out)
        self.assertNotIn("llm.layers.0.pre_norm.weight", out)
        self.assertIn("llm.layers.0.attn.x_r", out)
        self.assertIn("lm_head.weight", out)
        self.assertNotIn("vision_encoder.patch_embed.proj.weight", out)

    def test_rwkv_vl_state_dict_adapter_maps_and_reshapes_vision(self):
        spec = rwkv_vl_model_registry("debugmodel")
        adapter = RWKVVLStateDictAdapter(spec.model, hf_assets_path=None)
        vision_dim = spec.model.vision_encoder.dim
        hidden_size = spec.model.hidden_size
        conv = torch.empty(vision_dim, 3, 2, 16, 16)
        out = adapter.from_hf(
            {
                "model.encoder.patch_embed.proj.weight": conv,
                "model.encoder.blocks.0.attn.qkv.weight": torch.empty(
                    vision_dim * 3, vision_dim
                ),
                "model.encoder.deepstack_merger_list.0.norm.weight": torch.empty(
                    vision_dim
                ),
                "model.proj.main.pre_norm.weight": torch.empty(hidden_size),
                "model.llm.layers.0.pre_norm.weight": torch.empty(hidden_size),
                "model.llm.norm.weight": torch.empty(hidden_size),
                "lm_head.weight": torch.empty(spec.model.vocab_size, hidden_size),
            }
        )
        self.assertEqual(
            tuple(out["vision_encoder.patch_embed.proj.weight"].shape),
            (vision_dim, 3 * 2 * 16 * 16),
        )
        self.assertIn("vision_encoder.layers.0.attn.qkv.weight", out)
        self.assertIn("vision_encoder.deepstack_merger_list.0.norm.weight", out)
        self.assertIn("proj.main.pre_norm.weight", out)
        self.assertIn("llm.pre_norm.weight", out)
        self.assertIn("llm.norm.weight", out)

    def test_flattened_cu_seqlens_fixed_rows(self):
        cu = _build_flattened_cu_seqlens(
            batch_size=3,
            seq_len=5,
            positions=None,
            device=torch.device("cpu"),
        )
        self.assertEqual(cu.tolist(), [0, 5, 10, 15])

    def test_flattened_cu_seqlens_from_position_resets(self):
        positions = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]])
        cu = _build_flattened_cu_seqlens(
            batch_size=2,
            seq_len=5,
            positions=positions,
            device=torch.device("cpu"),
        )
        self.assertEqual(cu.tolist(), [0, 3, 5, 7, 10])

    def test_flattened_cu_seqlens_collapses_padded_position_tail(self):
        positions = torch.tensor([[0, 1, 2, 0, 1, 0, 0, 0]])
        cu = _build_flattened_cu_seqlens(
            batch_size=1,
            seq_len=8,
            positions=positions,
            device=torch.device("cpu"),
        )
        self.assertEqual(cu.tolist(), [0, 3, 5, 8])


class TestRWKVTokenizer(unittest.TestCase):
    def test_rwkv_tokenizer_preserves_sparse_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_tiny_rwkv_vocab(os.path.join(tmpdir, "wr_vocab_v20230424.txt"))
            tok = RwkvTokenizer(tokenizer_path=tmpdir)
            self.assertEqual(tok.get_vocab_size(), 65536)
            self.assertEqual(tok.image_id, 65532)
            self.assertEqual(tok.vision_start_id, 65530)
            ids = tok.encode("A<|image_pad|>B")
            self.assertIn(65532, ids)
            self.assertEqual(tok.decode(ids), "A<|image_pad|>B")

    def test_rwkv_multimodal_tokenizer_has_no_video_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_tiny_rwkv_vocab(os.path.join(tmpdir, "wr_vocab_v20230424.txt"))
            tok = RwkvVLMultiModalTokenizer(tokenizer_path=tmpdir)
            self.assertEqual(tok.TOKEN_FIELDS, ("image", "vision_start", "vision_end", "pad"))
            self.assertFalse(hasattr(tok, "video_id"))
            self.assertEqual(tok.pad_id, 24)


if __name__ == "__main__":
    unittest.main()
