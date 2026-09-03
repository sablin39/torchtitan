from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.protocols.module import Module


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
        if query.shape != key.shape:
            raise ValueError("Cosmos 3D RoPE query and key must have identical shapes")
        return self._rotate(query, positions), self._rotate(key, positions)


__all__ = ["Cosmos3DRotaryPositionEmbedding"]
