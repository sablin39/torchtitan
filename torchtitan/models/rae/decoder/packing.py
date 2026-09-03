from __future__ import annotations

from collections.abc import Sequence

import torch

from torchtitan.models.common.attention import VarlenMetadata


def _lengths_tensor(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if isinstance(sequence_lengths, torch.Tensor):
        lengths = sequence_lengths.to(device=device, dtype=torch.long)
        if lengths.ndim != 1:
            raise ValueError("RAE sequence_lengths must be one-dimensional")
    else:
        lengths = torch.tensor(sequence_lengths, dtype=torch.long, device=device)
    if lengths.numel() == 0 or torch.any(lengths <= 0):
        raise ValueError("RAE sequence_lengths must contain positive values")
    return lengths


def create_rae_varlen_metadata(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None = None,
    include_host_offsets: bool = True,
) -> VarlenMetadata:
    """Build FA2 cumulative offsets for packed RAE latent sequences."""
    lengths = _lengths_tensor(sequence_lengths, device=device).to(dtype=torch.int32)
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            torch.cumsum(lengths, dim=0).to(dtype=torch.int32),
        ]
    )
    host_offsets = (
        tuple(int(value) for value in offsets.tolist())
        if include_host_offsets
        else None
    )
    max_length = int(lengths.max().item())
    return VarlenMetadata(
        cu_seq_q=offsets,
        cu_seq_k=offsets,
        max_q=max_length,
        max_k=max_length,
        cu_seq_q_host=host_offsets,
    )


def create_rae_padding_mask(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    max_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a bidirectional padding mask for a padded RAE batch."""
    lengths = _lengths_tensor(sequence_lengths, device=device)
    target_length = max_length if max_length is not None else int(lengths.max().item())
    if target_length < int(lengths.max().item()):
        raise ValueError("max_length is smaller than a sequence length")
    token_index = torch.arange(target_length, device=lengths.device)
    valid = token_index.unsqueeze(0) < lengths.unsqueeze(1)
    return valid.unsqueeze(1) & valid.unsqueeze(2)


def create_rae_packed_attention_mask(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a block-diagonal mask for packed SDPA fallback attention."""
    lengths = _lengths_tensor(sequence_lengths, device=device)
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device), lengths
    )
    return sequence_ids.unsqueeze(0) == sequence_ids.unsqueeze(1)


__all__ = [
    "create_rae_varlen_metadata",
    "create_rae_padding_mask",
    "create_rae_packed_attention_mask",
]
