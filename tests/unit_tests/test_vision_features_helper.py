# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

import torch

from torchtitan.models.common.vision_features import (
    _find_vision_spans,
    _VisionSpan,
    apply_vision_slices,
)


def _build_spans(
    num_tokens_per_item: list[int], starts: list[int]
) -> list[_VisionSpan]:
    return [
        _VisionSpan(item_idx=i, start=starts[i], length=num_tokens_per_item[i])
        for i in range(len(num_tokens_per_item))
    ]


class TestFindVisionSpans(unittest.TestCase):
    def test_finds_consecutive_runs(self):
        tokens = torch.tensor([[0, 9, 9, 9, 0, 9, 9, 0]])
        num_tokens = torch.tensor([3, 2])
        spans = _find_vision_spans(tokens, num_tokens, vision_token_id=9)
        self.assertEqual(
            [(s.item_idx, s.start, s.length) for s in spans], [(0, 1, 3), (1, 5, 2)]
        )

    def test_no_matches_returns_empty_list(self):
        tokens = torch.tensor([[0, 1, 2, 3]])
        num_tokens = torch.tensor([], dtype=torch.long)
        self.assertEqual(_find_vision_spans(tokens, num_tokens, vision_token_id=9), [])

    def test_rejects_placeholder_count_mismatch(self):
        tokens = torch.tensor([[0, 9, 9, 0]])
        num_tokens = torch.tensor([2, 1])
        with self.assertRaisesRegex(ValueError, "1 contiguous run"):
            _find_vision_spans(tokens, num_tokens, vision_token_id=9)

    def test_rejects_placeholder_length_mismatch(self):
        tokens = torch.tensor([[0, 9, 9, 0]])
        num_tokens = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "spans 2 token"):
            _find_vision_spans(tokens, num_tokens, vision_token_id=9)


class TestApplyVisionSlicesPadded(unittest.TestCase):
    def _padded_features(
        self, values_per_item: list[list[float]], dim: int
    ) -> torch.Tensor:
        max_len = max(len(values) for values in values_per_item)
        out = torch.zeros(len(values_per_item), max_len, dim, dtype=torch.float32)
        for i, values in enumerate(values_per_item):
            for j, v in enumerate(values):
                out[i, j, :] = v
        return out

    def test_set_overwrites_in_single_shard(self):
        target = torch.full((10, 4), 0.5, dtype=torch.float32)
        features = self._padded_features([[1.0, 2.0, 3.0], [4.0, 5.0]], dim=4)
        spans = _build_spans([3, 2], starts=[1, 6])
        num_tokens = torch.tensor([3, 2])
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=10,
            reduce="set",
        )
        expected = torch.full((10, 4), 0.5, dtype=torch.float32)
        expected[1] = 1.0
        expected[2] = 2.0
        expected[3] = 3.0
        expected[6] = 4.0
        expected[7] = 5.0
        torch.testing.assert_close(target, expected)

    def test_add_accumulates(self):
        target = torch.full((6, 4), 0.5, dtype=torch.float32)
        features = self._padded_features([[1.0, 2.0]], dim=4)
        spans = _build_spans([2], starts=[2])
        num_tokens = torch.tensor([2])
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=6,
            reduce="add",
        )
        expected = torch.full((6, 4), 0.5, dtype=torch.float32)
        expected[2] = 1.5
        expected[3] = 2.5
        torch.testing.assert_close(target, expected)

    def test_shard_clips_partial_spans(self):
        target = torch.zeros(4, 4, dtype=torch.float32)
        features = self._padded_features([[1.0, 2.0, 3.0, 4.0]], dim=4)
        spans = _build_spans([4], starts=[2])
        num_tokens = torch.tensor([4])
        # Shard covers tokens [2, 6) globally, target is local indices [0, 4)
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=2,
            shard_length=4,
            reduce="set",
        )
        expected = torch.zeros(4, 4)
        expected[0] = 1.0
        expected[1] = 2.0
        expected[2] = 3.0
        expected[3] = 4.0
        torch.testing.assert_close(target, expected)

    def test_shard_clips_span_partial_overlap(self):
        target = torch.zeros(3, 4, dtype=torch.float32)
        features = self._padded_features([[1.0, 2.0, 3.0, 4.0, 5.0]], dim=4)
        spans = _build_spans([5], starts=[1])
        num_tokens = torch.tensor([5])
        # Span global [1, 6), shard global [2, 5); local indices [0, 3)
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=2,
            shard_length=3,
            reduce="set",
        )
        expected = torch.zeros(3, 4)
        expected[0] = 2.0
        expected[1] = 3.0
        expected[2] = 4.0
        torch.testing.assert_close(target, expected)

    def test_span_outside_shard_is_no_op(self):
        target = torch.zeros(4, 4, dtype=torch.float32)
        features = self._padded_features([[7.0, 7.0]], dim=4)
        spans = _build_spans([2], starts=[10])
        num_tokens = torch.tensor([2])
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=4,
            reduce="set",
        )
        torch.testing.assert_close(target, torch.zeros(4, 4))


class TestApplyVisionSlicesFlat(unittest.TestCase):
    def test_set_flat_features(self):
        target = torch.zeros(8, 4, dtype=torch.float32)
        features = torch.tensor(
            [[1.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4, [5.0] * 4],
            dtype=torch.float32,
        )
        spans = _build_spans([3, 2], starts=[1, 5])
        num_tokens = torch.tensor([3, 2])
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=8,
            reduce="set",
        )
        expected = torch.zeros(8, 4)
        expected[1] = 1.0
        expected[2] = 2.0
        expected[3] = 3.0
        expected[5] = 4.0
        expected[6] = 5.0
        torch.testing.assert_close(target, expected)

    def test_add_flat_features_with_cast(self):
        target = torch.zeros(4, 2, dtype=torch.float32)
        features = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.bfloat16)
        spans = _build_spans([2], starts=[1])
        num_tokens = torch.tensor([2])
        apply_vision_slices(
            target,
            features,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=4,
            reduce="add",
            cast_to_target=True,
        )
        self.assertEqual(target.dtype, torch.float32)
        expected = torch.zeros(4, 2)
        expected[1] = 1.0
        expected[2] = 2.0
        torch.testing.assert_close(target, expected)

    def test_padded_and_flat_inputs_produce_same_output(self):
        spans = _build_spans([2, 3], starts=[0, 4])
        num_tokens = torch.tensor([2, 3])
        flat = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
        padded = torch.zeros(2, 3, 4, dtype=torch.float32)
        padded[0, :2] = flat[:2]
        padded[1, :3] = flat[2:]

        out_flat = torch.zeros(8, 4)
        out_padded = torch.zeros(8, 4)
        apply_vision_slices(
            out_flat,
            flat,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=8,
            reduce="set",
        )
        apply_vision_slices(
            out_padded,
            padded,
            spans,
            num_tokens,
            shard_start=0,
            shard_length=8,
            reduce="set",
        )
        torch.testing.assert_close(out_flat, out_padded)


if __name__ == "__main__":
    unittest.main()
