from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import torchtitan.models.common.attention as attention_module
from torchtitan.models.common.attention import (
    build_varlen_metadata,
    configure_flash_attention_backend,
    flash_attention_varlen,
)


class TestVarlenMetadata(unittest.TestCase):
    def test_builds_asymmetric_cumulative_lengths(self):
        metadata = build_varlen_metadata(
            torch.tensor([4, 2, 7]),
            torch.tensor([64, 32, 112]),
        )
        torch.testing.assert_close(
            metadata.cu_seq_q, torch.tensor([0, 4, 6, 13], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.cu_seq_k, torch.tensor([0, 64, 96, 208], dtype=torch.int32)
        )
        self.assertEqual(metadata.max_q, 7)
        self.assertEqual(metadata.max_k, 112)

    def test_rejects_mismatched_segment_counts(self):
        with self.assertRaisesRegex(ValueError, "same number of segments"):
            build_varlen_metadata(torch.tensor([2, 3]), torch.tensor([5]))


class TestFlashAttentionBackendSelection(unittest.TestCase):
    def setUp(self):
        configure_flash_attention_backend.cache_clear()

    def tearDown(self):
        configure_flash_attention_backend.cache_clear()

    def test_activates_fa3_on_hopper(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                attention_module,
                "current_flash_attention_impl",
                side_effect=[None, "FA3"],
            ),
            patch.object(attention_module, "activate_flash_attention_impl") as activate,
        ):
            self.assertEqual(configure_flash_attention_backend(), "FA3")
        activate.assert_called_once_with("FA3")

    def test_keeps_native_backend_on_blackwell(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch.object(
                attention_module, "current_flash_attention_impl", return_value=None
            ),
            patch.object(attention_module, "activate_flash_attention_impl") as activate,
        ):
            self.assertIsNone(configure_flash_attention_backend())
        activate.assert_not_called()

    def test_falls_back_when_optional_fa3_is_unavailable(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(
                attention_module, "current_flash_attention_impl", return_value=None
            ),
            patch.object(
                attention_module,
                "activate_flash_attention_impl",
                side_effect=ImportError("optional FA3 package is absent"),
            ),
        ):
            self.assertIsNone(configure_flash_attention_backend())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FlashAttention")
class TestFlashAttentionVarlen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        configure_flash_attention_backend()

    def _reference(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_lengths: list[int],
        k_lengths: list[int],
        *,
        enable_gqa: bool,
    ) -> torch.Tensor:
        outputs = []
        q_offset = 0
        k_offset = 0
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True):
            q_segment = query[q_offset : q_offset + q_length].transpose(0, 1)[None]
            k_segment = key[k_offset : k_offset + k_length].transpose(0, 1)[None]
            v_segment = value[k_offset : k_offset + k_length].transpose(0, 1)[None]
            output = F.scaled_dot_product_attention(
                q_segment,
                k_segment,
                v_segment,
                enable_gqa=enable_gqa,
            )
            outputs.append(output[0].transpose(0, 1))
            q_offset += q_length
            k_offset += k_length
        return torch.cat(outputs)

    def _assert_forward_backward_parity(
        self,
        q_lengths: list[int],
        k_lengths: list[int],
        *,
        query_heads: int,
        kv_heads: int,
    ) -> None:
        generator = torch.Generator(device=self.device).manual_seed(2026)
        query = torch.randn(
            sum(q_lengths),
            query_heads,
            64,
            dtype=torch.bfloat16,
            device=self.device,
            generator=generator,
            requires_grad=True,
        )
        key = torch.randn(
            sum(k_lengths),
            kv_heads,
            64,
            dtype=torch.bfloat16,
            device=self.device,
            generator=generator,
            requires_grad=True,
        )
        value = torch.randn(
            sum(k_lengths),
            kv_heads,
            64,
            dtype=torch.bfloat16,
            device=self.device,
            generator=generator,
            requires_grad=True,
        )
        metadata = build_varlen_metadata(
            torch.tensor(q_lengths, device=self.device),
            torch.tensor(k_lengths, device=self.device),
        )
        enable_gqa = query_heads != kv_heads
        actual = flash_attention_varlen(
            query,
            key,
            value,
            metadata,
            causal=False,
            enable_gqa=enable_gqa,
        )
        expected = self._reference(
            query,
            key,
            value,
            q_lengths,
            k_lengths,
            enable_gqa=enable_gqa,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

        output_gradient = torch.randn_like(actual, generator=generator)
        actual_gradients = torch.autograd.grad(
            actual, (query, key, value), output_gradient, retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            expected, (query, key, value), output_gradient
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=2e-2, atol=2e-2
            )

    def test_bidirectional_self_attention(self):
        self._assert_forward_backward_parity(
            [17, 9, 23],
            [17, 9, 23],
            query_heads=8,
            kv_heads=8,
        )

    def test_asymmetric_cross_attention_with_gqa(self):
        self._assert_forward_backward_parity(
            [4, 2, 7],
            [64, 32, 112],
            query_heads=8,
            kv_heads=4,
        )


if __name__ == "__main__":
    unittest.main()
