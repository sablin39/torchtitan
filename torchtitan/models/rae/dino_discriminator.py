from __future__ import annotations

# Tensor dimensions: B=batch, L=token sequence, C=channel, N=head, D=head channel.

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINO_RECIPES = {
    "S_16": {
        "depth": 12,
        "patch_size": 16,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "S_8": {
        "depth": 12,
        "patch_size": 8,
        "embed_dim": 384,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "B_16": {
        "depth": 12,
        "patch_size": 16,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


class _DINOFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x_BLC)))


class _DINOSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("DINO embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = x_BLC.shape
        qkv_BL3ND = self.qkv(x_BLC).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query_BNLD, key_BNLD, value_BNLD = qkv_BL3ND.permute(2, 0, 3, 1, 4).unbind(
            dim=0
        )
        attention_BNLL = query_BNLD.mul(self.scale) @ key_BNLD.transpose(-2, -1)
        output_BNLD = attention_BNLL.softmax(dim=-1) @ value_BNLD
        output_BLC = output_BNLD.transpose(1, 2).reshape(
            batch_size, sequence_length, embed_dim
        )
        return self.proj(output_BLC)


class _DINOBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = _DINOSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = _DINOFeedForward(embed_dim, round(embed_dim * mlp_ratio))

    def forward(self, x_BLC: torch.Tensor) -> torch.Tensor:
        x_BLC = x_BLC + self.attn(self.norm1(x_BLC))
        return x_BLC + self.mlp(self.norm2(x_BLC))


class _DINOPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.Identity()

    def forward(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(images_BCHW).flatten(2).transpose(1, 2))


class _FrozenDINO(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        key_depths: tuple[int, ...],
        norm_eps: float,
        patch_size: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_grid_size = 224 // patch_size
        self.patch_embed = _DINOPatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_grid_size**2 + 1, embed_dim)
        )
        selected_depths = tuple(index for index in key_depths if index < depth)
        self.key_depths = frozenset(selected_depths)
        required_depth = max(depth, 1 + max(selected_depths, default=0))
        self.blocks = nn.Sequential(
            *[
                _DINOBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_eps=norm_eps,
                )
                for _ in range(required_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        mean_C = torch.tensor((0.485, 0.456, 0.406))
        std_C = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std_C).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean_C) / std_C).reshape(1, 3, 1, 1))
        self.eval()
        self.requires_grad_(False)

    def _interpolate_position_embedding(
        self, patch_height: int, patch_width: int
    ) -> torch.Tensor:
        if patch_height == self.patch_grid_size and patch_width == self.patch_grid_size:
            return self.pos_embed
        cls_embedding_B1C = self.pos_embed[:, :1]
        patch_embedding_BLC = self.pos_embed[:, 1:]
        patch_embedding_BCHW = patch_embedding_BLC.reshape(
            1,
            self.patch_grid_size,
            self.patch_grid_size,
            self.embed_dim,
        ).permute(0, 3, 1, 2)
        patch_embedding_BCHW = F.interpolate(
            patch_embedding_BCHW,
            size=(patch_height, patch_width),
            mode="bilinear",
            align_corners=False,
        )
        patch_embedding_BLC = patch_embedding_BCHW.permute(0, 2, 3, 1).reshape(
            1, patch_height * patch_width, self.embed_dim
        )
        return torch.cat((cls_embedding_B1C, patch_embedding_BLC), dim=1)

    def forward(self, images_BCHW: torch.Tensor) -> list[torch.Tensor]:
        images_BCHW = F.interpolate(
            images_BCHW,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        images_BCHW = images_BCHW * self.x_scale + self.x_shift
        patches_BLC = self.patch_embed(images_BCHW)
        batch_size = patches_BLC.shape[0]
        cls_token_B1C = self.cls_token.expand(batch_size, -1, -1)
        hidden_BLC = torch.cat((cls_token_B1C, patches_BLC), dim=1)
        patch_height = images_BCHW.shape[-2] // self.patch_size
        patch_width = images_BCHW.shape[-1] // self.patch_size
        hidden_BLC = hidden_BLC + self._interpolate_position_embedding(
            patch_height, patch_width
        )

        activations_BCL = []
        for block_index, block in enumerate(self.blocks):
            hidden_BLC = block(hidden_BLC)
            if block_index in self.key_depths:
                activations_BCL.append(hidden_BLC[:, 1:].transpose(1, 2))
        activations_BCL.insert(0, hidden_BLC[:, 1:].transpose(1, 2))
        return activations_BCL


class _ResidualBlock(nn.Module):
    def __init__(self, function: nn.Module) -> None:
        super().__init__()
        self.fn = function
        self.ratio = 1.0 / math.sqrt(2.0)

    def forward(self, x_BCL: torch.Tensor) -> torch.Tensor:
        return (self.fn(x_BCL).add(x_BCL)).mul_(self.ratio)


class _BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_BCL: torch.Tensor) -> torch.Tensor:
        shape = x_BCL.shape
        x_BCL = x_BCL.float()
        grouped_B1CL = x_BCL.view(shape[0], 1, shape[1], shape[2])
        mean_B11L = grouped_B1CL.mean((1, 3), keepdim=True)
        variance_B11L = grouped_B1CL.var((1, 3), keepdim=True, unbiased=False)
        grouped_B1CL = (grouped_B1CL - mean_B11L) / torch.sqrt(variance_B11L + self.eps)
        grouped_B1CL = (
            grouped_B1CL * self.weight[None, :, None] + self.bias[None, :, None]
        )
        return grouped_B1CL.view(shape)


def _make_head_block(
    channels: int,
    *,
    kernel_size: int,
    norm_type: str,
    norm_eps: float,
    using_spec_norm: bool,
) -> nn.Module:
    if norm_type == "bn":
        normalization = _BatchNormLocal(channels, norm_eps)
    elif norm_type == "gn":
        normalization = nn.GroupNorm(32, channels, eps=norm_eps, affine=True)
    else:
        raise ValueError(f"Unsupported DINO discriminator norm type: {norm_type}")
    convolution = nn.Conv1d(
        channels,
        channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
    )
    if using_spec_norm:
        convolution = nn.utils.spectral_norm(convolution)
    return nn.Sequential(
        convolution,
        normalization,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DINOFeatureDiscriminator(nn.Module):
    """Checkpoint-compatible RAEv2 DINO discriminator."""

    def __init__(
        self,
        *,
        device: torch.device,
        checkpoint_path: str,
        kernel_size: int,
        key_depths: tuple[int, ...],
        norm_type: str,
        using_spec_norm: bool,
        norm_eps: float,
        recipe: str,
    ) -> None:
        super().__init__()
        if recipe not in _DINO_RECIPES:
            raise ValueError(f"Unsupported DINO discriminator recipe: {recipe}")
        recipe_config = _DINO_RECIPES[recipe]
        selected_depths = tuple(
            index for index in key_depths if index < recipe_config["depth"]
        )
        dino = _FrozenDINO(
            depth=recipe_config["depth"],
            key_depths=selected_depths,
            norm_eps=norm_eps,
            patch_size=recipe_config["patch_size"],
            embed_dim=recipe_config["embed_dim"],
            num_heads=recipe_config["num_heads"],
            mlp_ratio=recipe_config["mlp_ratio"],
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("DINO discriminator checkpoint must contain a state dict")
        state = dict(state)
        for name, value in tuple(state.items()):
            if ".attn.qkv.bias" not in name:
                continue
            value = value.clone()
            channels = value.numel() // 3
            value[channels : 2 * channels].zero_()
            state[name] = value
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [name for name in missing if name not in {"x_scale", "x_shift"}]
        if missing:
            raise RuntimeError(f"DINO checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"DINO checkpoint has unexpected keys: {unexpected}")
        dino.to(device=device).eval().requires_grad_(False)
        self.dino_proxy = (dino,)

        channels = recipe_config["embed_dim"]
        heads = []
        for _ in range(len(selected_depths) + 1):
            output = nn.Conv1d(channels, 1, kernel_size=1)
            if using_spec_norm:
                output = nn.utils.spectral_norm(output)
            heads.append(
                nn.Sequential(
                    _make_head_block(
                        channels,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    _ResidualBlock(
                        _make_head_block(
                            channels,
                            kernel_size=kernel_size,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
                        )
                    ),
                    output,
                )
            )
        self.heads = nn.ModuleList(heads)

    def set_head_requires_grad(self, enabled: bool) -> None:
        self.dino_proxy[0].requires_grad_(False)
        self.heads.requires_grad_(enabled)

    def forward(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        activations_BCL = self.dino_proxy[0](images_BCHW)
        logits_BL = [
            head(activation_BCL).flatten(1)
            for head, activation_BCL in zip(self.heads, activations_BCL)
        ]
        return torch.cat(logits_BL, dim=1)


class HFDINOFeatureDiscriminator(nn.Module):
    """DINOv3 feature discriminator loaded from a local HF model directory."""

    def __init__(
        self,
        *,
        model_path: str,
        device: torch.device,
        key_depths: tuple[int, ...],
        kernel_size: int,
        norm_type: str,
        using_spec_norm: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face DINOv3 requires the optional transformers package"
            ) from error
        model_directory = Path(model_path).expanduser()
        if not model_directory.is_dir():
            raise ValueError(f"DINO model directory does not exist: {model_directory}")
        dino = AutoModel.from_pretrained(
            str(model_directory),
            local_files_only=True,
        )
        dino.to(device=device).eval().requires_grad_(False)
        self.dino_proxy = (dino,)
        dino_config = dino.config
        self.image_size = (
            int(dino_config.image_size) if hasattr(dino_config, "image_size") else 224
        )
        self.num_prefix_tokens = 1 + (
            int(dino_config.num_register_tokens)
            if hasattr(dino_config, "num_register_tokens")
            else 0
        )
        hidden_size = int(dino_config.hidden_size)
        num_layers = int(dino_config.num_hidden_layers)
        selected_depths = tuple(index for index in key_depths if index < num_layers)
        self.key_depths = selected_depths
        heads = []
        for _ in range(len(selected_depths) + 1):
            output = nn.Conv1d(hidden_size, 1, kernel_size=1)
            if using_spec_norm:
                output = nn.utils.spectral_norm(output)
            heads.append(
                nn.Sequential(
                    _make_head_block(
                        hidden_size,
                        kernel_size=1,
                        norm_type=norm_type,
                        norm_eps=norm_eps,
                        using_spec_norm=using_spec_norm,
                    ),
                    _ResidualBlock(
                        _make_head_block(
                            hidden_size,
                            kernel_size=kernel_size,
                            norm_type=norm_type,
                            norm_eps=norm_eps,
                            using_spec_norm=using_spec_norm,
                        )
                    ),
                    output,
                )
            )
        self.heads = nn.ModuleList(heads)

    def set_head_requires_grad(self, enabled: bool) -> None:
        self.dino_proxy[0].requires_grad_(False)
        self.heads.requires_grad_(enabled)

    def forward(self, images_BCHW: torch.Tensor) -> torch.Tensor:
        images_BCHW = F.interpolate(
            images_BCHW,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        mean_1C11 = images_BCHW.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std_1C11 = images_BCHW.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        outputs = self.dino_proxy[0](
            pixel_values=((images_BCHW + 1.0) * 0.5 - mean_1C11) / std_1C11,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("DINOv3 did not return hidden states")
        dino_norm = self.dino_proxy[0].norm
        activations_BCL = [
            outputs.last_hidden_state,
            *[dino_norm(hidden_states[index + 1]) for index in self.key_depths],
        ]
        activations_BCL = [
            activation_BLC[:, self.num_prefix_tokens :].transpose(1, 2)
            for activation_BLC in activations_BCL
        ]
        logits_BL = [
            head(activation_BCL).flatten(1)
            for head, activation_BCL in zip(self.heads, activations_BCL)
        ]
        return torch.cat(logits_BL, dim=1)


__all__ = ["DINOFeatureDiscriminator", "HFDINOFeatureDiscriminator"]
