from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def flatten_latents(
    latents: torch.Tensor,
    grid_thw: torch.Tensor | None,
    *,
    latent_dim: int,
    num_patches: int,
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
        token_count = tokens.shape[1]
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(
                "RAE latent token count must be square when grid_thw is omitted"
            )
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
        grid = torch.tensor(
            [1, target_side, target_side],
            dtype=torch.long,
            device=tokens.device,
        ).expand(tokens.shape[0], -1)
    else:
        if grid_thw.ndim not in (1, 2) or grid_thw.shape[-1] != 3:
            raise ValueError("RAE grid_thw must have shape (3,) or (B, 3)")
        if packed:
            grid = grid_thw.to(device=tokens.device, dtype=torch.long)
            expected_tokens = int(grid.prod(dim=-1).sum().item())
            if expected_tokens != tokens.shape[0]:
                raise ValueError(
                    "Packed RAE latent count does not match grid_thw: "
                    f"{tokens.shape[0]} != {expected_tokens}"
                )
        else:
            batch_size = tokens.shape[0]
            grid_thw = grid_thw.to(device=tokens.device, dtype=torch.long)
            grid = (
                grid_thw.view(1, 3).expand(batch_size, -1)
                if grid_thw.ndim == 1
                else grid_thw
            )
            if grid.shape[0] != batch_size:
                raise ValueError("RAE grid_thw batch does not match latents")
            token_counts = grid.prod(dim=-1)
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


__all__ = [
    "flatten_latents",
    "prepend_cls_positions",
    "unpatchify_batched",
    "unpatchify_packed",
]
