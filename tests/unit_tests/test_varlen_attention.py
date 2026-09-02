# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend:
#   T = packed tokens, N = attention heads, H = head dimension, D = model dim

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE


class TestPackedVarlenMetadata(unittest.TestCase):
    def test_document_boundaries(self):
        positions_T = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
        metadata = create_varlen_metadata_for_document(
            positions_T,
            include_host_offsets=True,
        )

        expected_cu_seq = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
        torch.testing.assert_close(metadata.cu_seq_q, expected_cu_seq)
        torch.testing.assert_close(metadata.cu_seq_k, expected_cu_seq)
        self.assertEqual(metadata.max_q, 4)
        self.assertEqual(metadata.max_k, 4)
        self.assertEqual(metadata.cu_seq_q_host, (0, 3, 5, 9))


class TestPackedVarlenAttention(unittest.TestCase):
    def test_sm120_restores_default_fa2(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            patch(
                "torchtitan.models.common.attention.restore_flash_attention_impl"
            ) as restore,
            patch(
                "torchtitan.models.common.attention.activate_flash_attention_impl"
            ) as activate,
        ):
            VarlenAttention.Config().build()

        activate.assert_not_called()
        restore.assert_called_once_with(_raise_warn=False)

    def test_custom_flash_activation_failure_restores_default_fa2(self):
        with (
            patch(
                "torchtitan.tools.utils.get_cuda_flash_attention_impl",
                return_value="FA3",
            ),
            patch(
                "torchtitan.models.common.attention.activate_flash_attention_impl",
                side_effect=RuntimeError("FA3 unavailable"),
            ) as activate,
            patch(
                "torchtitan.models.common.attention.restore_flash_attention_impl"
            ) as restore,
        ):
            VarlenAttention.Config().build()

        activate.assert_called_once_with("FA3")
        restore.assert_called_once_with(_raise_warn=False)

    def test_sm90_selection_is_not_stuck_after_sm120_fallback(self):
        capability = [(12, 0)]
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda: capability[0],
            ),
            patch(
                "torchtitan.models.common.attention.restore_flash_attention_impl"
            ) as restore,
            patch(
                "torchtitan.models.common.attention.activate_flash_attention_impl"
            ) as activate,
        ):
            VarlenAttention.Config().build()
            capability[0] = (9, 0)
            VarlenAttention.Config().build()

        restore.assert_called_once_with(_raise_warn=False)
        activate.assert_called_once_with("FA3")

    def test_gqa_preserves_td_shape(self):
        torch.manual_seed(42)
        num_tokens, dim, num_heads, head_dim = 6, 8, 2, 4
        attention = GQAttention.Config(
            n_heads=num_heads,
            n_kv_heads=num_heads,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_context_length=num_tokens),
        ).build()
        x_TD = torch.randn(num_tokens, dim)
        positions_T = torch.tensor([0, 1, 0, 1, 2, 3])
        metadata = create_varlen_metadata_for_document(positions_T)

        def _identity_varlen(q_TNH, k_TNH, v_TNH, *args, **kwargs):
            self.assertEqual(q_TNH.ndim, 3)
            self.assertEqual(k_TNH.ndim, 3)
            self.assertEqual(v_TNH.ndim, 3)
            return q_TNH

        with patch(
            "torchtitan.models.common.attention.varlen_attn",
            side_effect=_identity_varlen,
        ):
            out_TD = attention(x_TD, metadata, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_tnh_sharding_uses_varlen_argument_names(self):
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        model_config = llama3_configs["debugmodel"]("varlen")
        set_llama3_sharding_config(model_config, enable_sp=False)

        sharding = model_config.layers[0].attention.inner_attention.sharding_config
        assert sharding is not None
        self.assertEqual(
            set(sharding.in_src_shardings or {}),
            {"q_TNH", "k_TNH", "v_TNH"},
        )
        q_layout = (sharding.in_src_shardings or {})["q_TNH"]
        k_dst_layout = (sharding.in_dst_shardings or {})["k_TNH"]
        axis_types = q_layout.per_axis_spmd_types()
        self.assertEqual(axis_types[MeshAxisName.DP], spmd.S(0))
        self.assertEqual(axis_types[MeshAxisName.CP], spmd.S(0))
        self.assertEqual(axis_types[MeshAxisName.TP], spmd.S(1))
        self.assertEqual(k_dst_layout.per_axis_spmd_types()[MeshAxisName.CP], spmd.R)

    def test_out_transform_receives_tn_lse(self):
        num_tokens, num_heads, head_dim = 5, 2, 4
        q_TNH = torch.randn(num_tokens, num_heads, head_dim)
        positions_T = torch.tensor([0, 1, 0, 1, 2])
        metadata = create_varlen_metadata_for_document(positions_T)
        inner_attention = VarlenAttention.Config().build()

        def _varlen_with_lse(q, k, v, *args, **kwargs):
            lse_NT = torch.randn(num_heads, num_tokens)
            return q, lse_NT

        def _check_shapes(out_TNH, lse_TN):
            self.assertEqual(out_TNH.shape, q_TNH.shape)
            self.assertEqual(lse_TN.shape, (num_tokens, num_heads))
            return out_TNH

        with patch(
            "torchtitan.models.common.attention.varlen_attn",
            side_effect=_varlen_with_lse,
        ):
            out_TNH = inner_attention(
                q_TNH,
                q_TNH,
                q_TNH,
                attention_masks=metadata,
                out_transform=_check_shapes,
            )

        self.assertEqual(out_TNH.shape, q_TNH.shape)

    def test_llama_decoder_preserves_td_shape(self):
        from torchtitan.models.llama3 import llama3_configs

        model = llama3_configs["debugmodel"]("varlen").build()
        model.init_states()
        num_tokens = 6
        tokens_T = torch.randint(0, 2048, (num_tokens,))
        positions_T = torch.tensor([0, 1, 0, 1, 2, 3])
        metadata = model.get_attention_masks(positions_T)

        def _identity_varlen(q_TNH, k_TNH, v_TNH, *args, **kwargs):
            return q_TNH

        with patch(
            "torchtitan.models.common.attention.varlen_attn",
            side_effect=_identity_varlen,
        ):
            logits_TV = model(tokens_T, positions_T, metadata)

        self.assertEqual(logits_TV.shape, (num_tokens, 2048))


if __name__ == "__main__":
    unittest.main()
