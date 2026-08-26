# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchtitan.models.common.multimodal import get_vision_positions


ReduceMode = Literal["set", "add"]


@dataclass(frozen=True, slots=True)
class _VisionSpan:
    """A contiguous run of vision placeholders in flattened token order."""

    item_idx: int
    start: int
    length: int


def _find_vision_spans(
    tokens: torch.Tensor,
    num_tokens_per_item: torch.Tensor,
    vision_token_id: int,
) -> list[_VisionSpan]:
    return [
        _VisionSpan(
            item_idx=item_idx,
            start=start,
            length=length,
        )
        for item_idx, start, length in get_vision_positions(
            tokens.reshape(-1),
            num_tokens_per_item,
            vision_token_id,
        )
    ]


def _select_span_features(
    features: torch.Tensor,
    feature_offsets: torch.Tensor | None,
    span: _VisionSpan,
    feature_start: int,
    feature_len: int,
) -> torch.Tensor:
    if feature_offsets is None:
        return features[span.item_idx, feature_start : feature_start + feature_len]
    base = int(feature_offsets[span.item_idx].item())
    return features[base + feature_start : base + feature_start + feature_len]


def _compute_feature_offsets(
    features: torch.Tensor,
    num_tokens_per_item: torch.Tensor,
) -> torch.Tensor | None:
    if features.dim() != 2:
        return None
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=num_tokens_per_item.device),
            num_tokens_per_item.to(torch.long).cumsum(0),
        ]
    )


def apply_vision_slices(
    target_flat: torch.Tensor,
    features: torch.Tensor,
    spans: list[_VisionSpan],
    num_tokens_per_item: torch.Tensor,
    *,
    shard_start: int,
    shard_length: int,
    reduce: ReduceMode = "set",
    cast_to_target: bool = False,
) -> None:
    """Copy or accumulate vision features into ``target_flat`` in-place.

    ``features`` may be padded ``(num_items, T_max, D)`` or flat ``(total, D)``.
    Spans outside ``[shard_start, shard_start + shard_length)`` are skipped.
    """
    feature_offsets = _compute_feature_offsets(features, num_tokens_per_item)
    shard_end = shard_start + shard_length
    for span in spans:
        span_end = span.start + span.length
        overlap_start = max(span.start, shard_start)
        overlap_end = min(span_end, shard_end)
        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - shard_start
        feature_start = overlap_start - span.start
        feature_len = overlap_end - overlap_start
        vision_slice = _select_span_features(
            features, feature_offsets, span, feature_start, feature_len
        )
        if cast_to_target:
            vision_slice = vision_slice.to(target_flat.dtype)
        local_end = local_start + feature_len
        if reduce == "set":
            target_flat[local_start:local_end] = vision_slice
        else:
            target_flat[local_start:local_end] += vision_slice
