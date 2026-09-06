# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast, Literal

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList
from .layout import (
    flatten_latents,
    prepend_cls_positions,
    unpatchify_batched,
    unpatchify_packed,
)

from .packing import (
    create_rae_packed_attention_mask,
    create_rae_padding_mask,
    create_rae_static_varlen_metadata,
    create_rae_varlen_metadata,
)
from .position import Cosmos3DRotaryPositionEmbedding

# Tensor suffixes: B=batch, L=tokens, D=hidden, N=heads, H=head width,
# C=channels, Y/X=patch-grid axes, and P/Q=within-patch axes.


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

    def forward_padded(
        self,
        latents_TD: torch.Tensor,
        positions_T3: torch.Tensor,
        attention_masks: torch.Tensor | VarlenMetadata,
    ) -> torch.Tensor:
        """Run the fixed-shape packed decoder used by CUDA graph capture."""
        return self._forward_padded_impl(latents_TD, positions_T3, attention_masks)

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
        tokens, grid, packed = flatten_latents(
            latents,
            grid_thw,
            latent_dim=self.latent_dim,
            num_patches=self.num_patches,
        )
        packed_attention = packed or self.config.attention_backend == "varlen"
        if packed_attention:
            if not packed:
                tokens = tokens.reshape(-1, tokens.shape[-1])
            sequence_lengths = grid.prod(dim=-1)
            static_length = self.static_sequence_length
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
                        sequence_lengths,
                        static_length,
                        device=tokens.device,
                    )
                elif self.config.attention_backend == "sdpa":
                    attention_masks = create_rae_packed_attention_mask(
                        sequence_lengths, device=tokens.device
                    )
                else:
                    attention_masks = create_rae_varlen_metadata(
                        sequence_lengths, device=tokens.device
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
    "create_rae_padding_mask",
    "create_rae_packed_attention_mask",
    "create_rae_static_varlen_metadata",
    "create_rae_varlen_metadata",
]
