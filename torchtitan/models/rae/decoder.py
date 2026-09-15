# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RAE decoder architecture and variable-media geometry helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Literal

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList

# Tensor suffixes: B=batch, L=tokens, D=hidden, N=heads, H=head width,
# C=channels, Y/X=patch-grid axes, and P/Q=within-patch axes.


class Cosmos3DRotaryPositionEmbedding(Module):
    """Cosmos 3D RoPE for post-merger image and video tokens.

    This follows the public ``CosmosRotaryPosEmbed`` implementation in
    Hugging Face Diffusers' Cosmos transformer (derived from NVIDIA Cosmos):
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos.py
    Cosmos splits each head into temporal, height, and width frequency bands,
    applies FPS scaling only on the temporal band, and uses the contiguous-half
    real rotation from ``apply_rotary_emb(..., use_real_unbind_dim=-2)``.

    RAE latents are already Qwen post-merger tokens. Their spatial coordinates
    therefore use merger-cell centers in pre-merger patch units; this is the
    same coordinate convention as applying Cosmos RoPE after a spatial patch
    projection. ``temporal_start`` enables phase-continuous streaming clips.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.head_dim <= 0 or config.head_dim % 2:
            raise ValueError("Cosmos 3D RoPE head_dim must be positive and even")
        if config.theta <= 1.0:
            raise ValueError("Cosmos 3D RoPE theta must be greater than one")
        if len(config.rope_scale) != 3 or any(
            scale <= 0 for scale in config.rope_scale
        ):
            raise ValueError("Cosmos 3D RoPE rope_scale must contain three positives")
        if config.spatial_merge_size <= 0 or config.temporal_patch_size <= 0:
            raise ValueError("Cosmos 3D RoPE patch factors must be positive")
        if config.reference_fps <= 0:
            raise ValueError("Cosmos 3D RoPE reference_fps must be positive")
        self.head_dim = config.head_dim
        self.theta = config.theta
        self.rope_scale = tuple(float(scale) for scale in config.rope_scale)
        self.spatial_merge_size = config.spatial_merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.reference_fps = config.reference_fps

        # This is Cosmos' allocation: height and width each receive one third
        # of the rotary pairs, and temporal receives the remainder.
        self._axis_dimensions = (
            self.head_dim - 2 * (self.head_dim // 6 * 2),
            self.head_dim // 6 * 2,
            self.head_dim // 6 * 2,
        )
        self.register_buffer(
            "inv_freq_t",
            self._compute_inv_freq(self._axis_dimensions[0], self.rope_scale[0]),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            self._compute_inv_freq(self._axis_dimensions[1], self.rope_scale[1]),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            self._compute_inv_freq(self._axis_dimensions[2], self.rope_scale[2]),
            persistent=False,
        )

    def _compute_inv_freq(
        self, axis_dim: int, scale: float, *, device=None
    ) -> torch.Tensor:
        if axis_dim == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        # Cosmos applies NTK-aware scaling per axis. For a two-dimensional
        # band, the scaling exponent is undefined, so the neutral factor is
        # the continuous limit used by the implementation in practice.
        ntk_factor = 1.0 if axis_dim <= 2 else scale ** (axis_dim / (axis_dim - 2))
        theta = self.theta * ntk_factor
        return 1.0 / (
            theta
            ** (
                torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
                / axis_dim
            )
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.inv_freq_t.device
        if device.type == "meta":
            device = torch.device("cpu")
        self.inv_freq_t = self._compute_inv_freq(
            self._axis_dimensions[0], self.rope_scale[0], device=device
        )
        self.inv_freq_h = self._compute_inv_freq(
            self._axis_dimensions[1], self.rope_scale[1], device=device
        )
        self.inv_freq_w = self._compute_inv_freq(
            self._axis_dimensions[2], self.rope_scale[2], device=device
        )

    @staticmethod
    def _scalar_per_batch(
        value: torch.Tensor | float,
        batch_size: int,
        *,
        device: torch.device,
        name: str,
    ) -> list[float]:
        if isinstance(value, torch.Tensor):
            values = value.detach().to(device=device, dtype=torch.float32).flatten()
            if values.numel() == 1:
                values = values.expand(batch_size)
            elif values.numel() != batch_size:
                raise ValueError(
                    f"RAE {name} must be scalar or have one value per batch"
                )
            result = [float(item) for item in values.tolist()]
        else:
            result = [float(value)] * batch_size
        if name == "fps" and any(item < 0 for item in result):
            raise ValueError("RAE fps values must be non-negative")
        return result

    def _single_grid_positions(
        self,
        grid_thw: tuple[int, int, int],
        *,
        fps: float | None,
        temporal_start: float,
        device: torch.device,
    ) -> torch.Tensor:
        num_frames, height, width = grid_thw
        if min(num_frames, height, width) <= 0:
            raise ValueError("RAE grid_thw entries must be positive")
        if fps is None:
            temporal = (
                torch.arange(num_frames, device=device, dtype=torch.float32)
                + temporal_start
            )
        elif fps == 0.0:
            if num_frames != 1:
                raise ValueError("fps=0 is only valid for one-frame image inputs")
            temporal = torch.full(
                (1,), temporal_start, device=device, dtype=torch.float32
            )
        else:
            temporal = (
                torch.arange(num_frames, device=device, dtype=torch.float32) + 0.5
            ) * self.temporal_patch_size * (self.reference_fps / fps) + temporal_start
        height_axis = (
            torch.arange(height, device=device, dtype=torch.float32) + 0.5
        ) * self.spatial_merge_size
        width_axis = (
            torch.arange(width, device=device, dtype=torch.float32) + 0.5
        ) * self.spatial_merge_size
        temporal_grid, height_grid, width_grid = torch.meshgrid(
            temporal, height_axis, width_axis, indexing="ij"
        )
        return torch.stack(
            [temporal_grid.flatten(), height_grid.flatten(), width_grid.flatten()],
            dim=-1,
        )

    def build_positions(
        self,
        grid_thw: torch.Tensor,
        *,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Build Cosmos coordinates for one grid or equal-length batched grids."""
        if grid_thw.ndim not in (1, 2) or grid_thw.shape[-1] != 3:
            raise ValueError("RAE grid_thw must have shape (3,) or (B, 3)")
        grids = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        if grid_thw.ndim == 1:
            grids = [grids]
        fps_values = (
            None
            if fps is None
            else self._scalar_per_batch(
                fps,
                len(grids),
                device=grid_thw.device,
                name="fps",
            )
        )
        start_values = self._scalar_per_batch(
            temporal_start,
            len(grids),
            device=grid_thw.device,
            name="temporal_start",
        )
        positions = [
            self._single_grid_positions(
                (int(grid[0]), int(grid[1]), int(grid[2])),
                fps=None if fps_values is None else fps_values[index],
                temporal_start=start_values[index],
                device=grid_thw.device,
            )
            for index, grid in enumerate(grids)
        ]
        if len({position.shape[0] for position in positions}) != 1:
            raise ValueError(
                "RAE batched grid_thw entries must have equal token counts; "
                "use packed latents with build_packed_positions for variable grids"
            )
        output = torch.stack(positions, dim=0)
        return output[0] if grid_thw.ndim == 1 else output

    def build_packed_positions(
        self,
        grid_thw: torch.Tensor,
        *,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Build concatenated Cosmos coordinates for variable-length grids."""
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError("packed RAE grid_thw must have shape (num_sequences, 3)")
        grids = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        fps_values = (
            None
            if fps is None
            else self._scalar_per_batch(
                fps,
                len(grids),
                device=grid_thw.device,
                name="fps",
            )
        )
        start_values = self._scalar_per_batch(
            temporal_start,
            len(grids),
            device=grid_thw.device,
            name="temporal_start",
        )
        return torch.cat(
            [
                self._single_grid_positions(
                    (int(grid[0]), int(grid[1]), int(grid[2])),
                    fps=None if fps_values is None else fps_values[index],
                    temporal_start=start_values[index],
                    device=grid_thw.device,
                )
                for index, grid in enumerate(grids)
            ],
            dim=0,
        )

    def _frequencies(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_t = positions[..., 0:1] * self.inv_freq_t
        phase_h = positions[..., 1:2] * self.inv_freq_h
        phase_w = positions[..., 2:3] * self.inv_freq_w
        phase = torch.cat(
            [phase_t, phase_h, phase_w, phase_t, phase_h, phase_w], dim=-1
        )
        return phase.cos(), phase.sin()

    def _rotate(self, values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.head_dim:
            raise ValueError(
                f"Cosmos 3D RoPE expected head width {self.head_dim}, got {values.shape[-1]}"
            )
        cos, sin = self._frequencies(positions)
        if values.ndim == 3 and positions.ndim == 2:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        elif values.ndim == 4 and positions.ndim == 3:
            cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        else:
            raise ValueError(
                "Cosmos 3D RoPE expects packed (T, N, H) or batched "
                "(B, L, N, H) values with matching coordinates"
            )
        values_float = values.float()
        real, imaginary = values_float.reshape(
            *values.shape[:-1], 2, self.head_dim // 2
        ).unbind(-2)
        rotated = torch.cat([-imaginary, real], dim=-1)
        return (values_float * cos + rotated * sin).to(values.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query.ndim != key.ndim
            or query.shape[:-2] != key.shape[:-2]
            or query.shape[-1] != key.shape[-1]
        ):
            raise ValueError(
                "Cosmos 3D RoPE query and key must share token dimensions and head width"
            )
        return self._rotate(query, positions), self._rotate(key, positions)


def _lengths_tensor(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if isinstance(sequence_lengths, torch.Tensor):
        if sequence_lengths.ndim != 1:
            raise ValueError("RAE sequence_lengths must be one-dimensional")
        if sequence_lengths.device.type != "cpu":
            raise ValueError(
                "RAE sequence_lengths must be host metadata on the CPU; a "
                "device tensor would stall the training loop on a D2H readback"
            )
        lengths = sequence_lengths.to(dtype=torch.long)
    else:
        lengths = torch.tensor(sequence_lengths, dtype=torch.long)
    if lengths.numel() == 0 or torch.any(lengths <= 0):
        raise ValueError("RAE sequence_lengths must contain positive values")
    return lengths.to(device=device)


def create_rae_varlen_metadata(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None = None,
    include_host_offsets: bool = True,
) -> VarlenMetadata:
    """Build FA2 cumulative offsets for packed RAE latent sequences."""
    # All integer math stays on the host; the offsets pay a single H2D copy,
    # so a deep GPU queue cannot stall the caller.
    lengths = _lengths_tensor(sequence_lengths, device=None).to(dtype=torch.int32)
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
    offsets = offsets.to(device=device, non_blocking=True)
    return VarlenMetadata(
        cu_seq_q=offsets,
        cu_seq_k=offsets,
        max_q=max_length,
        max_k=max_length,
        cu_seq_q_host=host_offsets,
    )


def create_rae_static_varlen_metadata(
    sequence_lengths: torch.Tensor | Sequence[int],
    static_sequence_length: int,
    *,
    device: torch.device | None = None,
) -> VarlenMetadata:
    """Build fixed-shape FA2 metadata with one isolated padding sequence.

    The real samples remain separate attention documents. Any unused token
    slots are appended as a final document, so padding cannot affect valid
    queries and the cumulative-offset tensor keeps a stable shape for
    compilation and CUDA graph replay.
    """
    # All integer math stays on the host; the offsets pay a single H2D copy,
    # so a deep GPU queue cannot stall the caller.
    lengths = _lengths_tensor(sequence_lengths, device=None).to(dtype=torch.int32)
    valid_length = int(lengths.sum().item())
    if static_sequence_length < valid_length:
        raise ValueError(
            "static_sequence_length must be at least the packed token count"
        )
    if static_sequence_length > valid_length:
        padding_length = static_sequence_length - valid_length
        lengths = torch.cat(
            [
                lengths,
                torch.tensor(
                    [padding_length], dtype=torch.int32, device=lengths.device
                ),
            ]
        )
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            torch.cumsum(lengths, dim=0).to(dtype=torch.int32),
        ]
    )
    offsets = offsets.to(device=device, non_blocking=True)
    return VarlenMetadata(
        cu_seq_q=offsets,
        cu_seq_k=offsets,
        max_q=static_sequence_length,
        max_k=static_sequence_length,
        cu_seq_q_host=None,
    )


def create_rae_packed_attention_mask(
    sequence_lengths: torch.Tensor | Sequence[int],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a bidirectional block-diagonal mask for packed SDPA fallback."""
    lengths = _lengths_tensor(sequence_lengths, device=device)
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device), lengths
    )
    return sequence_ids.unsqueeze(0) == sequence_ids.unsqueeze(1)


def flatten_latents(
    latents: torch.Tensor,
    grid_thw: torch.Tensor | None,
    *,
    latent_dim: int,
    num_patches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Normalize RAE latent layouts and return post-merger grid metadata."""
    packed = latents.ndim == 2
    if latents.ndim == 4:
        _, channels, _, _ = latents.shape
        if channels != latent_dim:
            raise ValueError(f"Expected latent channels {latent_dim}, got {channels}")
        tokens = latents.flatten(2).transpose(1, 2)
    elif latents.ndim == 3:
        if latents.shape[-1] != latent_dim:
            raise ValueError(
                f"Expected latent width {latent_dim}, got {latents.shape[-1]}"
            )
        tokens = latents
    elif latents.ndim == 2:
        if latents.shape[-1] != latent_dim:
            raise ValueError(
                f"Expected latent width {latent_dim}, got {latents.shape[-1]}"
            )
        if grid_thw is None:
            raise ValueError("Packed RAE latents require grid_thw metadata")
        tokens = latents
    else:
        raise ValueError(
            "RAE latents must have shape (B, C, H, W), (B, L, C), or (T, C)"
        )

    if grid_thw is None:
        if packed:
            raise ValueError("Packed RAE latents require grid_thw metadata")
        if latents.ndim == 4 and num_patches is None:
            target_height, target_width = latents.shape[-2:]
        else:
            token_count = tokens.shape[1]
            side = int(math.sqrt(token_count))
            if side * side != token_count:
                raise ValueError(
                    "RAE latent token count must be square when grid_thw is omitted"
                )
            if num_patches is None:
                target_height = target_width = side
            else:
                target_side = int(math.sqrt(num_patches))
                if token_count != num_patches:
                    tokens = (
                        F.interpolate(
                            tokens.transpose(1, 2).reshape(
                                tokens.shape[0], latent_dim, side, side
                            ),
                            size=(target_side, target_side),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .flatten(2)
                        .transpose(1, 2)
                    )
                target_height = target_width = target_side
        grid = torch.tensor(
            [1, target_height, target_width],
            dtype=torch.long,
            device=tokens.device,
        ).expand(tokens.shape[0], -1)
    else:
        if grid_thw.ndim not in (1, 2) or grid_thw.shape[-1] != 3:
            raise ValueError("RAE grid_thw must have shape (3,) or (B, 3)")
        if grid_thw.device.type != "cpu":
            raise ValueError(
                "RAE grid_thw must be host metadata on the CPU; a device "
                "tensor would stall the training loop on a D2H readback"
            )
        if packed:
            expected_tokens = int(
                grid_thw.to(dtype=torch.long).prod(dim=-1).sum().item()
            )
            grid = grid_thw.to(
                device=tokens.device, dtype=torch.long, non_blocking=True
            )
            if expected_tokens != tokens.shape[0]:
                raise ValueError(
                    "Packed RAE latent count does not match grid_thw: "
                    f"{tokens.shape[0]} != {expected_tokens}"
                )
        else:
            batch_size = tokens.shape[0]
            token_counts = grid_thw.to(dtype=torch.long).view(-1, 3).prod(dim=-1)
            grid_thw = grid_thw.to(
                device=tokens.device, dtype=torch.long, non_blocking=True
            )
            grid = (
                grid_thw.view(1, 3).expand(batch_size, -1)
                if grid_thw.ndim == 1
                else grid_thw
            )
            if grid.shape[0] != batch_size:
                raise ValueError("RAE grid_thw batch does not match latents")
            if torch.any(token_counts != tokens.shape[1]):
                raise ValueError(
                    "Every batched RAE grid_thw entry must match the latent token count"
                )
    return tokens, grid, packed


def prepend_cls_positions(positions: torch.Tensor) -> torch.Tensor:
    cls_shape = (*positions.shape[:-2], 1, 3)
    cls_positions = torch.zeros(
        cls_shape, dtype=positions.dtype, device=positions.device
    )
    return torch.cat([cls_positions, positions], dim=-2)


def unpatchify_batched(
    patch_logits: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    if grid_thw.ndim != 2 or grid_thw.shape[0] != patch_logits.shape[0]:
        raise ValueError("Batched RAE output requires one grid_thw entry per sample")
    grids = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
    if len({tuple(grid) for grid in grids}) != 1:
        raise ValueError(
            "Batched RAE outputs require equal grids; use packed latents and "
            "unpatchify_packed for variable resolution"
        )
    num_frames, height, width = (int(value) for value in grids[0])
    patch_logits = patch_logits.view(
        patch_logits.shape[0],
        num_frames,
        height,
        width,
        patch_size,
        patch_size,
        3,
    )
    output = patch_logits.permute(0, 6, 1, 2, 4, 3, 5).reshape(
        patch_logits.shape[0],
        3,
        num_frames,
        height * patch_size,
        width * patch_size,
    )
    return output[:, :, 0] if num_frames == 1 else output


def unpatchify_packed(
    patch_logits: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    patch_size: int,
) -> list[torch.Tensor]:
    """Unpatchify packed logits into one image or video tensor per grid."""
    if patch_logits.ndim != 2 or grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
        raise ValueError("packed logits and grid_thw must be two-dimensional")
    outputs = []
    offset = 0
    for num_frames, height, width in grid_thw.detach().to("cpu").tolist():
        num_frames, height, width = (
            int(num_frames),
            int(height),
            int(width),
        )
        length = num_frames * height * width
        sequence = patch_logits[offset : offset + length]
        if sequence.shape[0] != length:
            raise ValueError("packed logits do not match grid_thw")
        sequence = sequence.view(
            num_frames,
            height,
            width,
            patch_size,
            patch_size,
            3,
        )
        output = sequence.permute(5, 0, 1, 3, 2, 4).reshape(
            3,
            num_frames,
            height * patch_size,
            width * patch_size,
        )
        outputs.append(output[:, 0] if num_frames == 1 else output)
        offset += length
    if offset != patch_logits.shape[0]:
        raise ValueError("packed logits contain tokens not described by grid_thw")
    return outputs


class RAEAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        num_kv_heads: int
        attention_backend: Literal["sdpa", "varlen"] = "sdpa"
        rope_theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("RAE hidden_size must be divisible by num_heads")
        if config.num_kv_heads <= 0:
            raise ValueError("RAE num_kv_heads must be positive")
        if config.num_kv_heads > config.num_heads:
            raise ValueError("RAE num_kv_heads cannot exceed num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("RAE num_heads must be divisible by num_kv_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.enable_gqa = self.num_heads != self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        if self.head_dim % 2:
            raise ValueError("RAE attention head_dim must be even for rotary embedding")
        self.attention_backend = config.attention_backend
        if self.attention_backend not in {"sdpa", "varlen"}:
            raise ValueError(
                f"Unsupported RAE attention backend: {self.attention_backend}"
            )
        self.qkv = Linear.Config(
            in_features=config.hidden_size,
            out_features=(self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
        ).build()
        self.proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
        ).build()
        self.rope = Cosmos3DRotaryPositionEmbedding.Config(
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scale=config.rope_scale,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            reference_fps=config.reference_fps,
        ).build()
        self.varlen_attention = (
            VarlenAttention.Config(window_size=(-1, -1)).build()
            if self.attention_backend == "varlen"
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | VarlenMetadata | None = None,
    ) -> torch.Tensor:
        if x.ndim == 2:
            if self.attention_backend == "varlen":
                if not isinstance(attention_masks, VarlenMetadata):
                    raise ValueError(
                        "RAE varlen attention requires VarlenMetadata for packed "
                        "latents"
                    )
            elif isinstance(attention_masks, VarlenMetadata):
                raise ValueError(
                    "VarlenMetadata requires RAE attention_backend='varlen'"
                )
            token_count, hidden = x.shape
            qkv_TD = self.qkv(x)
            q_TNqH = qkv_TD[..., : self.num_heads * self.head_dim].view(
                token_count, self.num_heads, self.head_dim
            )
            kv_start = self.num_heads * self.head_dim
            kv_width = self.num_kv_heads * self.head_dim
            k_TNkvH = qkv_TD[..., kv_start : kv_start + kv_width].view(
                token_count, self.num_kv_heads, self.head_dim
            )
            v_TNkvH = qkv_TD[..., kv_start + kv_width :].view(
                token_count, self.num_kv_heads, self.head_dim
            )
            if positions is not None:
                q_TNqH, k_TNkvH = self.rope(q_TNqH, k_TNkvH, positions)
            if self.attention_backend == "varlen":
                assert self.varlen_attention is not None
                out_TNqH = self.varlen_attention(
                    q_TNqH,
                    k_TNkvH,
                    v_TNkvH,
                    attention_masks=attention_masks,
                    scale=self.head_dim**-0.5,
                    enable_gqa=self.enable_gqa,
                )
            else:
                if attention_masks is not None and not isinstance(
                    attention_masks, torch.Tensor
                ):
                    raise ValueError(
                        "Packed SDPA attention requires a tensor attention mask"
                    )
                out_NqTH = F.scaled_dot_product_attention(
                    q_TNqH.transpose(0, 1),
                    k_TNkvH.transpose(0, 1),
                    v_TNkvH.transpose(0, 1),
                    attn_mask=attention_masks,
                    scale=self.head_dim**-0.5,
                    enable_gqa=self.enable_gqa,
                )
                out_TNqH = out_NqTH.transpose(0, 1)
            return self.proj(out_TNqH.reshape(token_count, hidden))

        if x.ndim != 3:
            raise ValueError("RAE attention input must have shape (B, L, D) or (T, D)")
        if self.attention_backend == "varlen":
            raise ValueError(
                "RAE varlen attention consumes packed (T, D) latents; flatten the "
                "batch and provide VarlenMetadata"
            )
        if attention_masks is not None and not isinstance(
            attention_masks, torch.Tensor
        ):
            raise ValueError("Batched SDPA attention requires a tensor attention mask")
        batch, length, hidden = x.shape
        qkv_BLD = self.qkv(x)
        q_BLNqH = qkv_BLD[..., : self.num_heads * self.head_dim].view(
            batch, length, self.num_heads, self.head_dim
        )
        kv_start = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        k_BLNkvH = qkv_BLD[..., kv_start : kv_start + kv_width].view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        v_BLNkvH = qkv_BLD[..., kv_start + kv_width :].view(
            batch, length, self.num_kv_heads, self.head_dim
        )
        if positions is not None:
            q_BLNqH, k_BLNkvH = self.rope(q_BLNqH, k_BLNkvH, positions)
        out_BNqLH = F.scaled_dot_product_attention(
            q_BLNqH.transpose(1, 2),
            k_BLNkvH.transpose(1, 2),
            v_BLNkvH.transpose(1, 2),
            attn_mask=attention_masks,
            scale=self.head_dim**-0.5,
            enable_gqa=self.enable_gqa,
        )
        out_BLD = out_BNqLH.transpose(1, 2).reshape(batch, length, hidden)
        return self.proj(out_BLD)


class RAEFeedForward(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
        ).build()
        self.up = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
        ).build()
        self.down = Linear.Config(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
        ).build()

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x_BLD)) * self.up(x_BLD))


class RAEBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        num_kv_heads: int
        intermediate_size: int
        norm_eps: float = 1e-6
        attention_backend: Literal["sdpa", "varlen"] = "sdpa"
        rope_theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0
        residual_dropout: float = 0.1

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.attention = RAEAttention.Config(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            attention_backend=config.attention_backend,
            rope_theta=config.rope_theta,
            rope_scale=config.rope_scale,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            reference_fps=config.reference_fps,
        ).build()
        self.norm2 = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.feed_forward = RAEFeedForward.Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        ).build()
        if not 0.0 <= config.residual_dropout < 1.0:
            raise ValueError("RAE residual_dropout must be in [0, 1)")
        self.residual_dropout = config.residual_dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | VarlenMetadata | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.norm1(x),
            positions=positions,
            attention_masks=attention_masks,
        )
        x = x + F.dropout(
            attention_output,
            p=self.residual_dropout,
            training=self.training,
        )
        return x + F.dropout(
            self.feed_forward(self.norm2(x)),
            p=self.residual_dropout,
            training=self.training,
        )


class RAEDecoder(BaseModel):
    """TorchTitan-native RAEv2 Stage 1 decoder.

    The model consumes legacy ``(B, C, H, W)`` latents, batched post-merger
    ``(B, L, C)`` latents, or packed ``(T, C)`` latents. Batched inputs return
    ``(B, 3, H, W)`` for images and ``(B, 3, T, H, W)`` for videos. Packed
    inputs return patch logits; call :meth:`unpatchify_packed` to recover a
    list of variable-size clips. The encoder, GAN discriminator, and EMA copy
    intentionally live in the Stage 1 trainer so this model remains compatible
    with TorchTitan meta construction. Set ``Config.image_size=-1`` when
    runtime grid metadata should determine the output resolution.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 0
        vocab_size: int = 0
        lm_head: Linear.Config | None = None
        tok_embeddings: Any = None
        norm: RMSNorm.Config | None = None
        layers: list[RAEBlock.Config] = field(default_factory=list)
        latent_dim: int = 1024
        image_size: int = -1
        patch_size: int = 16
        hidden_size: int = 1024
        num_layers: int = 8
        num_heads: int = 16
        num_kv_heads: int = 4
        intermediate_size: int = 3072
        norm_eps: float = 1e-6
        attention_backend: Literal["sdpa", "varlen"] = "sdpa"
        rope_theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0
        residual_dropout: float = 0.1
        static_sequence_length: int = 0
        long_skip_connections: tuple[tuple[int, int], ...] = ()
        """U-ViT-style long skips as explicit ``(source, target)`` block pairs:
        the output of block ``source`` is concatenated into the input of block
        ``target`` through a dedicated linear projection. Empty keeps the plain
        ViT stack."""
        use_dmuon: bool = False
        flops_attention_context: int = 0
        """Document length assumed for the attention term of the FLOPs estimate.

        Packed varlen batches give each token only its own document as
        attention context, so using the packed sequence length would overstate
        attention FLOPs by orders of magnitude. 0 falls back to seq_len."""

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            if self.image_size == 0 or self.image_size < -1:
                raise ValueError("RAE image_size must be -1 or positive")
            if self.image_size != -1 and self.image_size % self.patch_size != 0:
                raise ValueError("RAE image_size must be divisible by patch_size")
            if self.attention_backend not in {"sdpa", "varlen"}:
                raise ValueError(
                    f"Unsupported RAE attention backend: {self.attention_backend}"
                )
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("RAE hidden_size must be divisible by num_heads")
            if self.num_kv_heads <= 0 or self.num_kv_heads > self.num_heads:
                raise ValueError("RAE num_kv_heads must be between one and num_heads")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("RAE num_heads must be divisible by num_kv_heads")
            if (self.hidden_size // self.num_heads) % 2:
                raise ValueError("RAE attention head_dim must be even")
            if self.spatial_merge_size <= 0 or self.temporal_patch_size <= 0:
                raise ValueError("RAE patch factors must be positive")
            if self.reference_fps <= 0:
                raise ValueError("RAE reference_fps must be positive")
            if not 0.0 <= self.residual_dropout < 1.0:
                raise ValueError("RAE residual_dropout must be in [0, 1)")
            if self.static_sequence_length < 0:
                raise ValueError("RAE static_sequence_length must be non-negative")
            skip_targets: set[int] = set()
            for source, target in self.long_skip_connections:
                if not 0 <= source < target < self.num_layers:
                    raise ValueError(
                        "RAE long_skip_connections pairs must satisfy "
                        f"0 <= source < target < num_layers, got ({source}, {target})"
                    )
                if target in skip_targets:
                    raise ValueError(
                        "RAE long_skip_connections target block repeated: " f"{target}"
                    )
                skip_targets.add(target)
            if not self.layers:
                self.layers = [
                    RAEBlock.Config(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        num_kv_heads=self.num_kv_heads,
                        intermediate_size=self.intermediate_size,
                        norm_eps=self.norm_eps,
                        attention_backend=self.attention_backend,
                        rope_theta=self.rope_theta,
                        rope_scale=self.rope_scale,
                        spatial_merge_size=self.spatial_merge_size,
                        temporal_patch_size=self.temporal_patch_size,
                        reference_fps=self.reference_fps,
                        residual_dropout=self.residual_dropout,
                    )
                    for _ in range(self.num_layers)
                ]

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            parameter_count = sum(p.numel() for p in model.parameters())
            head_dim = self.hidden_size // self.num_heads
            context = self.flops_attention_context or seq_len
            attention_flops = (
                6
                * self.num_layers
                * self.num_heads
                * 2
                * head_dim
                * min(context, max(seq_len, 1))
            )
            return parameter_count, 6 * parameter_count + attention_flops

    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.layers:
            raise ValueError("RAEDecoder.Config.layers must be populated")
        self.config = config
        self.latent_dim = config.latent_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (
            None
            if config.image_size == -1
            else (config.image_size // config.patch_size) ** 2
        )
        self.spatial_merge_size = config.spatial_merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.reference_fps = config.reference_fps
        self.static_sequence_length = config.static_sequence_length
        self.input_projection = Linear.Config(
            in_features=config.latent_dim,
            out_features=config.hidden_size,
        ).build()
        self.trainable_cls_token = torch.nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )
        self.layers = ModuleList([layer.build() for layer in config.layers])
        self.skip_projections = ModuleList(
            [
                Linear.Config(
                    in_features=2 * config.hidden_size,
                    out_features=config.hidden_size,
                ).build()
                for _ in range(len(config.long_skip_connections))
            ]
        )
        # Skip routing tables keyed by block index; one projection per pair,
        # ordered as the pairs are declared.
        self._skip_by_target = {
            target: (index, source)
            for index, (source, target) in enumerate(config.long_skip_connections)
        }
        self._skip_sources = {source for source, _ in config.long_skip_connections}
        self.decoder_norm = RMSNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.decoder_pred = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.patch_size * config.patch_size * 3,
        ).build()
        self._dmuon_enabled = config.use_dmuon

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.trainable_cls_token, std=0.02)

    @staticmethod
    def unpatchify_packed(
        patch_logits: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        patch_size: int,
    ) -> list[torch.Tensor]:
        return unpatchify_packed(
            patch_logits,
            grid_thw,
            patch_size=patch_size,
        )

    def _apply_blocks(
        self,
        hidden: torch.Tensor,
        *,
        positions: torch.Tensor,
        attention_masks: torch.Tensor | VarlenMetadata | None,
    ) -> torch.Tensor:
        num_skips = len(self.skip_projections)
        if not num_skips:
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    positions=positions,
                    attention_masks=attention_masks,
                )
            return hidden
        skips: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            if index in self._skip_by_target:
                projection, source = self._skip_by_target[index]
                hidden = self.skip_projections[projection](
                    torch.cat([hidden, skips.pop(source)], dim=-1)
                )
            hidden = layer(
                hidden,
                positions=positions,
                attention_masks=attention_masks,
            )
            if index in self._skip_sources:
                skips[index] = hidden
        return hidden

    def _forward_padded_impl(
        self,
        latents_TD: torch.Tensor,
        positions_T3: torch.Tensor,
        attention_masks: torch.Tensor | VarlenMetadata,
    ) -> torch.Tensor:
        if latents_TD.ndim != 2 or latents_TD.shape[-1] != self.latent_dim:
            raise ValueError("RAE padded latents must have shape (T, latent_dim)")
        if positions_T3.shape != (latents_TD.shape[0], 3):
            raise ValueError("RAE padded positions must match the latent sequence")
        hidden_TD = self.input_projection(latents_TD)
        hidden_TD = self._apply_blocks(
            hidden_TD,
            positions=positions_T3,
            attention_masks=attention_masks,
        )
        return self.decoder_pred(self.decoder_norm(hidden_TD))

    def forward(
        self,
        latents: torch.Tensor,
        *,
        grid_thw: torch.Tensor | None = None,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
        attention_masks: torch.Tensor | VarlenMetadata | None = None,
        return_padded: bool = False,
        padded_positions_T3: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padded_positions_T3 is not None:
            if attention_masks is None:
                raise ValueError(
                    "RAE padded decoding requires fixed varlen attention metadata"
                )
            return self._forward_padded_impl(
                latents,
                padded_positions_T3,
                attention_masks,
            )
        host_grid = grid_thw.to(dtype=torch.long) if grid_thw is not None else None
        tokens, grid, packed = flatten_latents(
            latents,
            grid_thw,
            latent_dim=self.latent_dim,
            num_patches=self.num_patches,
        )
        packed_attention = packed or self.config.attention_backend == "varlen"
        if packed_attention:
            batch_size = tokens.shape[0]
            if not packed:
                tokens = tokens.reshape(-1, tokens.shape[-1])
            sequence_lengths = grid.prod(dim=-1)
            static_length = self.static_sequence_length
            if host_grid is not None:
                # Host grids give token counts and RoPE coordinates without
                # stalling the CPU on a deep GPU queue.
                host_lengths = host_grid.reshape(-1, 3).prod(dim=-1).tolist()
                if host_grid.ndim == 1 and not packed:
                    host_lengths = host_lengths * batch_size
                valid_length = sum(host_lengths)
            else:
                host_lengths = None
                valid_length = int(sequence_lengths.sum().item())
            if static_length:
                if self.config.attention_backend != "varlen":
                    raise ValueError(
                        "RAE static_sequence_length requires attention_backend='varlen'"
                    )
                if static_length <= valid_length:
                    raise ValueError(
                        "RAE static_sequence_length must exceed the packed token count"
                    )
                tokens = F.pad(tokens, (0, 0, 0, static_length - valid_length))
            if attention_masks is None:
                if static_length:
                    attention_masks = create_rae_static_varlen_metadata(
                        host_lengths if host_lengths is not None else sequence_lengths,
                        static_length,
                        device=tokens.device,
                    )
                elif self.config.attention_backend == "sdpa":
                    attention_masks = create_rae_packed_attention_mask(
                        host_lengths if host_lengths is not None else sequence_lengths,
                        device=tokens.device,
                    )
                else:
                    attention_masks = create_rae_varlen_metadata(
                        host_lengths if host_lengths is not None else sequence_lengths,
                        device=tokens.device,
                    )
            elif static_length and (
                not isinstance(attention_masks, VarlenMetadata)
                or attention_masks.cu_seq_q.shape[0]
                not in (grid.shape[0] + 1, grid.shape[0] + 2)
            ):
                raise ValueError(
                    "RAE static_sequence_length requires matching fixed varlen metadata"
                )
            first_layer = cast(RAEBlock, self.layers[0])
            if host_grid is not None:
                positions_grid = (
                    host_grid.view(1, 3).expand(batch_size, -1)
                    if host_grid.ndim == 1 and not packed
                    else host_grid
                )
                positions = first_layer.attention.rope.build_packed_positions(
                    positions_grid,
                    fps=fps,
                    temporal_start=temporal_start,
                ).to(tokens.device, non_blocking=True)
            else:
                positions = first_layer.attention.rope.build_packed_positions(
                    grid,
                    fps=fps,
                    temporal_start=temporal_start,
                )
            if static_length:
                positions = F.pad(
                    positions,
                    (0, 0, 0, static_length - positions.shape[0]),
                )
            patch_logits = self._forward_padded_impl(
                tokens,
                positions,
                attention_masks,
            )
            if static_length and not return_padded:
                patch_logits = patch_logits[:valid_length]
            return patch_logits

        first_layer = cast(RAEBlock, self.layers[0])
        positions = first_layer.attention.rope.build_positions(
            grid,
            fps=fps,
            temporal_start=temporal_start,
        )
        if positions.ndim == 2:
            positions = positions.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        hidden = self.input_projection(tokens)
        cls_B1D = self.trainable_cls_token.expand(hidden.shape[0], -1, -1)
        hidden = torch.cat([cls_B1D, hidden], dim=1)
        positions = prepend_cls_positions(positions)
        hidden = self._apply_blocks(
            hidden,
            positions=positions,
            attention_masks=attention_masks,
        )
        patch_logits = self.decoder_pred(self.decoder_norm(hidden[:, 1:]))
        return unpatchify_batched(
            patch_logits,
            grid.view(-1, 3),
            patch_size=self.patch_size,
        )


__all__ = [
    "RAEDecoder",
    "RAEBlock",
    "RAEAttention",
    "RAEFeedForward",
    "Cosmos3DRotaryPositionEmbedding",
    "create_rae_packed_attention_mask",
    "create_rae_static_varlen_metadata",
    "create_rae_varlen_metadata",
    "flatten_latents",
    "prepend_cls_positions",
    "unpatchify_batched",
    "unpatchify_packed",
]
