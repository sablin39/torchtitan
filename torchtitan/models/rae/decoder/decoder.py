from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
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
        qkv_bias: bool = True
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
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.attention_backend = config.attention_backend
        if self.attention_backend not in {"sdpa", "varlen"}:
            raise ValueError(
                f"Unsupported RAE attention backend: {self.attention_backend}"
            )
        self.qkv = Linear.Config(
            in_features=config.hidden_size,
            out_features=3 * config.hidden_size,
            bias=config.qkv_bias,
        ).build()
        self.proj = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias=True,
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
                        "RAE varlen attention requires VarlenMetadata for packed latents"
                    )
            elif isinstance(attention_masks, VarlenMetadata):
                raise ValueError(
                    "VarlenMetadata requires RAE attention_backend='varlen'"
                )
            token_count, hidden = x.shape
            qkv_T3NH: torch.Tensor = self.qkv(x).view(
                token_count, 3, self.num_heads, self.head_dim
            )
            q_TNH, k_TNH, v_TNH = qkv_T3NH.unbind(dim=1)
            if positions is not None:
                q_TNH, k_TNH = self.rope(q_TNH, k_TNH, positions)
            if self.attention_backend == "varlen":
                assert self.varlen_attention is not None
                out_TNH = self.varlen_attention(
                    q_TNH,
                    k_TNH,
                    v_TNH,
                    attention_masks=attention_masks,
                    scale=self.head_dim**-0.5,
                )
            else:
                out_NTH = F.scaled_dot_product_attention(
                    q_TNH.transpose(0, 1),
                    k_TNH.transpose(0, 1),
                    v_TNH.transpose(0, 1),
                    attn_mask=attention_masks,
                    scale=self.head_dim**-0.5,
                )
                out_TNH = out_NTH.transpose(0, 1)
            return self.proj(out_TNH.reshape(token_count, hidden))

        if x.ndim != 3:
            raise ValueError("RAE attention input must have shape (B, L, D) or (T, D)")
        if self.attention_backend == "varlen":
            raise ValueError(
                "RAE varlen attention consumes packed (T, D) latents; flatten the "
                "batch and provide VarlenMetadata"
            )
        batch, length, hidden = x.shape
        qkv_BL3NH: torch.Tensor = self.qkv(x).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q_BLNH, k_BLNH, v_BLNH = qkv_BL3NH.unbind(dim=2)
        if positions is not None:
            q_BLNH, k_BLNH = self.rope(q_BLNH, k_BLNH, positions)
        out_BNLH = F.scaled_dot_product_attention(
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
            attn_mask=attention_masks,
            scale=self.head_dim**-0.5,
        )
        out_BLD = out_BNLH.transpose(1, 2).reshape(batch, length, hidden)
        return self.proj(out_BLD)


class RAEFeedForward(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.up = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
        ).build()
        self.down = Linear.Config(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
        ).build()

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x_BLD), approximate="tanh"))


class RAEBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        intermediate_size: int
        norm_eps: float = 1e-6
        attention_backend: Literal["sdpa", "varlen"] = "sdpa"
        rope_theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.attention = RAEAttention.Config(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            attention_backend=config.attention_backend,
            rope_theta=config.rope_theta,
            rope_scale=config.rope_scale,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            reference_fps=config.reference_fps,
        ).build()
        self.norm2 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.feed_forward = RAEFeedForward.Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        ).build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attention_masks: torch.Tensor | VarlenMetadata | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x),
            positions=positions,
            attention_masks=attention_masks,
        )
        return x + self.feed_forward(self.norm2(x))


class RAEDecoder(BaseModel):
    """TorchTitan-native RAEv2 Stage 1 decoder.

    The model consumes legacy ``(B, C, H, W)`` latents, batched post-merger
    ``(B, L, C)`` latents, or packed ``(T, C)`` latents. Batched inputs return
    ``(B, 3, H, W)`` for images and ``(B, 3, T, H, W)`` for videos. Packed
    inputs return patch logits; call :meth:`unpatchify_packed` to recover a
    list of variable-size clips. The encoder, GAN discriminator, and EMA copy
    intentionally live in the Stage 1 trainer so this model remains compatible
    with TorchTitan meta construction.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 0
        vocab_size: int = 0
        lm_head: Linear.Config | None = None
        tok_embeddings: Any = None
        norm: LayerNorm.Config | None = None
        layers: list[RAEBlock.Config] = field(default_factory=list)
        latent_dim: int = 768
        image_size: int = 256
        patch_size: int = 16
        hidden_size: int = 512
        num_layers: int = 8
        num_heads: int = 16
        intermediate_size: int = 2048
        norm_eps: float = 1e-6
        qkv_bias: bool = True
        attention_backend: Literal["sdpa", "varlen"] = "sdpa"
        rope_theta: float = 10000.0
        rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
        spatial_merge_size: int = 2
        temporal_patch_size: int = 2
        reference_fps: float = 24.0
        use_dmuon: bool = False

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            if self.image_size % self.patch_size != 0:
                raise ValueError("RAE image_size must be divisible by patch_size")
            if self.attention_backend not in {"sdpa", "varlen"}:
                raise ValueError(
                    f"Unsupported RAE attention backend: {self.attention_backend}"
                )
            if self.spatial_merge_size <= 0 or self.temporal_patch_size <= 0:
                raise ValueError("RAE patch factors must be positive")
            if self.reference_fps <= 0:
                raise ValueError("RAE reference_fps must be positive")
            if not self.layers:
                self.layers = [
                    RAEBlock.Config(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        intermediate_size=self.intermediate_size,
                        norm_eps=self.norm_eps,
                        attention_backend=self.attention_backend,
                        rope_theta=self.rope_theta,
                        rope_scale=self.rope_scale,
                        spatial_merge_size=self.spatial_merge_size,
                        temporal_patch_size=self.temporal_patch_size,
                        reference_fps=self.reference_fps,
                    )
                    for _ in range(self.num_layers)
                ]

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            parameter_count = sum(p.numel() for p in model.parameters())
            attention_flops = 6 * self.num_layers * self.hidden_size * max(seq_len, 1)
            return parameter_count, 6 * parameter_count + attention_flops

    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.layers:
            raise ValueError("RAEDecoder.Config.layers must be populated")
        self.config = config
        self.latent_dim = config.latent_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.spatial_merge_size = config.spatial_merge_size
        self.temporal_patch_size = config.temporal_patch_size
        self.reference_fps = config.reference_fps
        self.input_projection = Linear.Config(
            in_features=config.latent_dim,
            out_features=config.hidden_size,
            bias=True,
        ).build()
        self.trainable_cls_token = torch.nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )
        self.layers = ModuleList([layer.build() for layer in config.layers])
        self.decoder_norm = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.decoder_pred = Linear.Config(
            in_features=config.hidden_size,
            out_features=config.patch_size * config.patch_size * 3,
            bias=True,
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

    def forward(
        self,
        latents: torch.Tensor,
        *,
        grid_thw: torch.Tensor | None = None,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
        attention_masks: torch.Tensor | VarlenMetadata | None = None,
    ) -> torch.Tensor:
        tokens, grid, packed = flatten_latents(
            latents,
            grid_thw,
            latent_dim=self.latent_dim,
            num_patches=self.num_patches,
        )
        if packed:
            if attention_masks is None:
                sequence_lengths = grid.prod(dim=-1)
                if self.config.attention_backend == "sdpa":
                    attention_masks = create_rae_packed_attention_mask(
                        sequence_lengths, device=tokens.device
                    )
                else:
                    attention_masks = create_rae_varlen_metadata(
                        sequence_lengths, device=tokens.device
                    )
            positions = self.layers[0].attention.rope.build_packed_positions(
                grid,
                fps=fps,
                temporal_start=temporal_start,
            )
            hidden = self.input_projection(tokens)
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    positions=positions,
                    attention_masks=attention_masks,
                )
            return self.decoder_pred(self.decoder_norm(hidden))

        positions = self.layers[0].attention.rope.build_positions(
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
        for layer in self.layers:
            hidden = layer(
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
    "create_rae_varlen_metadata",
]
