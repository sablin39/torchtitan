# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text processing utilities for multimodal datasets."""

import torch


def pad_seq_len(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_len: int,
    *,
    padding_idx: int,
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad input_ids and labels to desired sequence length."""
    B, L = input_ids.shape

    if L < target_len:
        padding_length = target_len - L
        padding_input = torch.full(
            (B, padding_length), padding_idx, dtype=torch.long, device=input_ids.device
        )
        padding_labels = torch.full(
            (B, padding_length), ignore_idx, dtype=torch.long, device=labels.device
        )

        input_ids = torch.cat([input_ids, padding_input], dim=1)
        labels = torch.cat([labels, padding_labels], dim=1)

    elif L > target_len:
        input_ids = input_ids[:, :target_len]
        labels = labels[:, :target_len]

    return input_ids, labels


def pad_batch_dim(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    target_batch_size: int,
    *,
    padding_idx: int,
    ignore_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad batch dimension to target size."""
    B, L = input_ids.shape
    assert B <= target_batch_size, f"Batch size {B} exceeds target {target_batch_size}"
    if B == target_batch_size:
        return input_ids, labels

    padding_needed = target_batch_size - B
    padding_input = torch.full(
        (padding_needed, L), padding_idx, dtype=torch.long, device=input_ids.device
    )
    padding_labels = torch.full(
        (padding_needed, L), ignore_idx, dtype=torch.long, device=labels.device
    )

    input_ids = torch.cat([input_ids, padding_input], dim=0)
    labels = torch.cat([labels, padding_labels], dim=0)

    return input_ids, labels
