from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList

# Tensor suffixes: B=batch, L=tokens, D=hidden, N=heads, H=head width,
# C=channels, Y/X=patch-grid axes, and P/Q=within-patch axes.


class RAEAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        qkv_bias: bool = True

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("RAE hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
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

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x_BLD.shape
        qkv_BL3NH: torch.Tensor = self.qkv(x_BLD).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        qkv_3BNLH = qkv_BL3NH.permute(2, 0, 3, 1, 4)
        q_BNLH, k_BNLH, v_BNLH = qkv_3BNLH.unbind(0)
        out_BNLH = F.scaled_dot_product_attention(q_BNLH, k_BNLH, v_BNLH)
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

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.attention = RAEAttention.Config(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
        ).build()
        self.norm2 = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            eps=config.norm_eps,
        ).build()
        self.feed_forward = RAEFeedForward.Config(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        ).build()

    def forward(self, x_BLD: torch.Tensor) -> torch.Tensor:
        x_BLD = x_BLD + self.attention(self.norm1(x_BLD))
        return x_BLD + self.feed_forward(self.norm2(x_BLD))


class RAEDecoder(BaseModel):
    """TorchTitan-native RAEv2 Stage 1 decoder.

    The model consumes encoder latents in ``(B, C, H, W)`` format and returns
    reconstructed images in ``(B, 3, image_size, image_size)`` format. The
    encoder, GAN discriminator, and EMA copy intentionally live in the Stage 1
    trainer so this model remains compatible with TorchTitan meta construction.
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
        use_dmuon: bool = False

        def update_from_config(self, *, config, **kwargs) -> None:
            del kwargs
            if self.image_size % self.patch_size != 0:
                raise ValueError("RAE image_size must be divisible by patch_size")
            if not self.layers:
                self.layers = [
                    RAEBlock.Config(
                        hidden_size=self.hidden_size,
                        num_heads=self.num_heads,
                        intermediate_size=self.intermediate_size,
                        norm_eps=self.norm_eps,
                    )
                    for _ in range(self.num_layers)
                ]

        def get_nparams_and_flops(
            self, model: torch.nn.Module, seq_len: int
        ) -> tuple[int, int]:
            del seq_len
            parameter_count = sum(p.numel() for p in model.parameters())
            return parameter_count, 0

    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.layers:
            raise ValueError("RAEDecoder.Config.layers must be populated")
        self.config = config
        self.latent_dim = config.latent_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.input_projection = Linear.Config(
            in_features=config.latent_dim,
            out_features=config.hidden_size,
            bias=True,
        ).build()
        self.trainable_cls_token = torch.nn.Parameter(
            torch.zeros(1, 1, config.hidden_size)
        )
        self.register_buffer(
            "decoder_pos_embed",
            torch.empty(1, self.num_patches + 1, config.hidden_size),
            persistent=True,
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

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.decoder_pos_embed.device
        if device.type == "meta":
            device = torch.device("cpu")
        grid_size = int(math.sqrt(self.num_patches))
        if grid_size * grid_size != self.num_patches:
            raise ValueError("RAE decoder requires a square number of output patches")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            indexing="ij",
        )
        half = self.decoder_pos_embed.shape[-1] // 4
        if half == 0 or self.decoder_pos_embed.shape[-1] % 4:
            raise ValueError("RAE decoder hidden_size must be divisible by four")
        frequency = torch.arange(half, device=device, dtype=torch.float32)
        frequency = 1.0 / (10000 ** (frequency / half))
        angles_y = grid_y.reshape(-1, 1) * frequency.reshape(1, -1)
        angles_x = grid_x.reshape(-1, 1) * frequency.reshape(1, -1)
        patch_pos = torch.cat(
            [angles_y.sin(), angles_y.cos(), angles_x.sin(), angles_x.cos()], dim=1
        )
        cls_pos = torch.zeros(1, self.decoder_pos_embed.shape[-1], device=device)
        self.decoder_pos_embed.copy_(
            torch.cat([cls_pos, patch_pos], dim=0).unsqueeze(0)
        )

    def _flatten_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            batch, channels, _, _ = latents.shape
            if channels != self.latent_dim:
                raise ValueError(
                    f"Expected latent channels {self.latent_dim}, got {channels}"
                )
            tokens_BLC = latents.flatten(2).transpose(1, 2)
        elif latents.ndim == 3:
            if latents.shape[-1] != self.latent_dim:
                raise ValueError(
                    f"Expected latent width {self.latent_dim}, got {latents.shape[-1]}"
                )
            tokens_BLC = latents
        else:
            raise ValueError("RAE latents must have shape (B, C, H, W) or (B, L, C)")
        if tokens_BLC.shape[1] != self.num_patches:
            side = int(math.sqrt(tokens_BLC.shape[1]))
            if side * side != tokens_BLC.shape[1]:
                raise ValueError("RAE latent token count must be a square")
            target_side = int(math.sqrt(self.num_patches))
            tokens_BLC = (
                F.interpolate(
                    tokens_BLC.transpose(1, 2).reshape(
                        tokens_BLC.shape[0], self.latent_dim, side, side
                    ),
                    size=(target_side, target_side),
                    mode="bilinear",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2)
            )
        return tokens_BLC

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        tokens_BLD = self.input_projection(self._flatten_latents(latents))
        cls_B1D = self.trainable_cls_token.expand(tokens_BLD.shape[0], -1, -1)
        tokens_BLD = torch.cat([cls_B1D, tokens_BLD], dim=1)
        tokens_BLD = tokens_BLD + self.decoder_pos_embed.to(tokens_BLD.dtype)
        for layer in self.layers:
            tokens_BLD = layer(tokens_BLD)
        patch_logits_BLP = self.decoder_pred(self.decoder_norm(tokens_BLD[:, 1:]))
        side = self.image_size // self.patch_size
        patch_logits_BYXPQC = patch_logits_BLP.view(
            patch_logits_BLP.shape[0],
            side,
            side,
            self.patch_size,
            self.patch_size,
            3,
        )
        return patch_logits_BYXPQC.permute(0, 5, 1, 3, 2, 4).reshape(
            patch_logits_BLP.shape[0], 3, self.image_size, self.image_size
        )


__all__ = ["RAEDecoder", "RAEBlock", "RAEAttention", "RAEFeedForward"]
