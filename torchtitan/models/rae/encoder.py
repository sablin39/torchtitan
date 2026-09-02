from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINO_HUB_MODELS = {
    "dinov2-vit-s": ("facebookresearch/dinov2", "dinov2_vits14_reg"),
    "dinov2-vit-b": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
    "dinov2-vit-l": ("facebookresearch/dinov2", "dinov2_vitl14_reg"),
    "dinov2-vit-g": ("facebookresearch/dinov2", "dinov2_vitg14_reg"),
    "dinov3-vit-s16": ("facebookresearch/dinov3", "dinov3_vits16"),
    "dinov3-vit-b16": ("facebookresearch/dinov3", "dinov3_vitb16"),
    "dinov3-vit-l16": ("facebookresearch/dinov3", "dinov3_vitl16"),
}


def _merge_qwen_hidden_states(
    outputs,
    merger: nn.Module,
    layer_indices: tuple[int, ...],
) -> torch.Tensor:
    if not layer_indices:
        return outputs.pooler_output
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Qwen vision model did not return hidden states")
    selected = [hidden_states[index + 1] for index in layer_indices]
    return merger(torch.stack(selected).sum(dim=0))


@dataclass(frozen=True, slots=True)
class RAEEncoderConfig:
    kind: str = "fixed"
    name: str = ""
    latent_dim: int = 768
    image_size: int = 256
    noise_tau: float = 0.8
    normalization_stat_path: str | None = None
    checkpoint_path: str | None = None
    repository_path: str | None = None
    layer_indices: tuple[int, ...] = ()
    merge_size: int = 1

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("encoder.image_size must be positive")
        if self.merge_size <= 0:
            raise ValueError("encoder.merge_size must be positive")
        if self.image_size % self.merge_size:
            raise ValueError("encoder.image_size must be divisible by merge_size")


class FrozenRAEEncoder(nn.Module):
    """Frozen image-to-latent adapter for Stage 1.

    ``kind='dino_hub'`` loads a DINO encoder with PyTorch Hub. ``kind='qwen'``
    loads only the local Qwen vision tower and merger. ``kind='fixed'`` is
    deterministic and dependency-free for smoke tests.
    """

    def __init__(self, config: RAEEncoderConfig, device: torch.device) -> None:
        super().__init__()
        self.image_size = config.image_size
        self.latent_dim = config.latent_dim
        self.noise_tau = config.noise_tau
        self.kind = config.kind
        self.layer_indices = config.layer_indices
        self.merge_size = config.merge_size
        self.supervision_image_size = config.image_size // config.merge_size
        self.latent_mean = None
        self.latent_var = None
        if config.normalization_stat_path is not None:
            stats = torch.load(
                config.normalization_stat_path,
                map_location="cpu",
                weights_only=True,
            )
            self.latent_mean = self._format_stat(stats.get("mean"), "mean")
            self.latent_var = self._format_stat(stats.get("var"), "var")
        self.external = None
        if config.kind == "qwen":
            self._init_qwen(config, device)
        elif config.kind in {"dino_hub", "hf"}:
            if not config.name:
                raise ValueError("DINO encoder name is required")
            if config.kind == "hf":
                try:
                    from transformers import AutoModel
                except ImportError as error:
                    raise RuntimeError(
                        "encoder.kind='hf' requires the optional transformers package"
                    ) from error
                model_path = str(Path(config.name).expanduser())
                self.external = AutoModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
                self.external.to(device=device).eval()
                external_dim = getattr(
                    self.external.config,
                    "hidden_size",
                    getattr(self.external.config, "embed_dim", None),
                )
                if external_dim != config.latent_dim:
                    raise ValueError(
                        "DINO encoder hidden size does not match decoder latent_dim: "
                        f"{external_dim} != {config.latent_dim}"
                    )
                self._hf_num_register_tokens = int(
                    getattr(self.external.config, "num_register_tokens", 0)
                )
                self._hf_processor_size = self.image_size
                self._hf_patch_size = int(
                    getattr(self.external.config, "patch_size", 16)
                )
            elif config.name not in _DINO_HUB_MODELS:
                raise ValueError(f"Unsupported DINO encoder name: {config.name}")
            else:
                default_repository, model_name = _DINO_HUB_MODELS[config.name]
                hub_repository = config.repository_path or default_repository
                load_kwargs = {
                    "source": "local" if config.repository_path else "github",
                    "trust_repo": True,
                    "pretrained": config.checkpoint_path is None,
                }
                self.external = torch.hub.load(
                    hub_repository,
                    model_name,
                    **load_kwargs,
                )
                if config.checkpoint_path is not None:
                    state = torch.load(
                        config.checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    if not isinstance(state, dict):
                        raise ValueError(
                            "DINO encoder checkpoint must contain a state dict"
                        )
                    missing, unexpected = self.external.load_state_dict(
                        state, strict=False
                    )
                    if missing:
                        raise RuntimeError(
                            f"DINO encoder checkpoint missing keys: {missing}"
                        )
                    if unexpected:
                        raise RuntimeError(
                            "DINO encoder checkpoint has unexpected keys: "
                            f"{unexpected}"
                        )
                external_dim = getattr(
                    self.external,
                    "embed_dim",
                    getattr(self.external, "hidden_size", None),
                )
            if external_dim != config.latent_dim:
                raise ValueError(
                    "DINO encoder hidden size does not match decoder latent_dim: "
                    f"{external_dim} != {config.latent_dim}"
                )
        elif config.kind == "fixed":
            self.projection = nn.Conv2d(3, config.latent_dim, kernel_size=1)
            with torch.no_grad():
                generator = torch.Generator(device="cpu").manual_seed(0)
                self.projection.weight.copy_(
                    torch.randn(
                        self.projection.weight.shape,
                        generator=generator,
                    )
                    / math.sqrt(3)
                )
                self.projection.bias.zero_()
        else:
            raise ValueError(f"Unsupported RAE encoder kind: {config.kind}")
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    def _init_qwen(self, config: RAEEncoderConfig, device: torch.device) -> None:
        if not config.name:
            raise ValueError("Qwen encoder name is required")
        if config.checkpoint_path is not None:
            raise ValueError("Qwen encoder does not accept checkpoint_path")
        try:
            from safetensors import safe_open
            from transformers import AutoConfig, AutoImageProcessor
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        except ImportError as error:
            raise RuntimeError(
                "encoder.kind='qwen' requires transformers and safetensors"
            ) from error
        model_directory = Path(config.name).expanduser()
        if not model_directory.is_dir():
            raise ValueError(f"Qwen model directory does not exist: {model_directory}")
        full_config = AutoConfig.from_pretrained(
            str(model_directory), local_files_only=True
        )
        vision_config = getattr(full_config, "vision_config", None)
        if vision_config is None:
            raise ValueError("Qwen checkpoint does not contain a vision_config")
        if vision_config.spatial_merge_size != config.merge_size:
            raise ValueError(
                "encoder.merge_size does not match Qwen vision config: "
                f"{config.merge_size} != {vision_config.spatial_merge_size}"
            )
        if config.image_size % (vision_config.patch_size * config.merge_size):
            raise ValueError(
                "Qwen encoder image_size must be divisible by patch_size * merge_size"
            )
        visual = Qwen3_5VisionModel(vision_config)
        index_path = model_directory / "model.safetensors.index.json"
        if index_path.is_file():
            import json

            with index_path.open() as index_file:
                weight_map = json.load(index_file)["weight_map"]
            shard_names = sorted(
                {
                    name
                    for key, name in weight_map.items()
                    if key.startswith("model.visual.")
                }
            )
        else:
            shard_names = sorted(model_directory.glob("*.safetensors"))
            shard_names = [path.name for path in shard_names]
        visual_state: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            shard_path = model_directory / shard_name
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                for key in shard.keys():
                    if key.startswith("model.visual."):
                        visual_state[key[len("model.visual.") :]] = shard.get_tensor(
                            key
                        )
        missing, unexpected = visual.load_state_dict(visual_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Qwen vision checkpoint mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.external = visual.to(device=device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_directory), local_files_only=True
        )
        self._qwen_patch_size = int(vision_config.patch_size)
        self._qwen_depth = int(vision_config.depth)
        external_dim = int(vision_config.out_hidden_size)
        if external_dim != config.latent_dim:
            raise ValueError(
                "Qwen post-merger width does not match decoder latent_dim: "
                f"{external_dim} != {config.latent_dim}"
            )
        invalid_indices = [
            index
            for index in config.layer_indices
            if index < 0 or index >= self._qwen_depth
        ]
        if invalid_indices:
            raise ValueError(
                f"Qwen encoder layer indices are out of range: {invalid_indices}"
            )

    def _format_stat(
        self, value: torch.Tensor | None, name: str
    ) -> torch.Tensor | None:
        if value is None:
            return None
        value = torch.as_tensor(value)
        if value.ndim == 1:
            value = value.view(1, -1, 1, 1)
        elif value.ndim == 3:
            value = value.unsqueeze(0)
        elif value.ndim != 4:
            raise ValueError(
                f"RAE encoder {name} statistics must have shape (C,), (C, H, W), "
                "or (1, C, H, W)"
            )
        if value.shape[1] != self.latent_dim:
            raise ValueError(
                f"RAE encoder {name} statistics have {value.shape[1]} channels, "
                f"expected {self.latent_dim}"
            )
        return value

    def forward(
        self, images_BCHW: torch.Tensor, *, add_noise: bool = False
    ) -> torch.Tensor:
        images_BCHW = images_BCHW.float()
        if images_BCHW.shape[-2:] != (self.image_size, self.image_size):
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.external is not None and self.kind == "qwen":
            processor_output = self.processor(
                images_BCHW.detach(),
                do_rescale=False,
                return_tensors="pt",
            )
            external_device = next(self.external.parameters()).device
            processor_output = {
                name: value.to(device=external_device)
                for name, value in processor_output.items()
                if torch.is_tensor(value)
            }
            with torch.no_grad():
                outputs = self.external(
                    hidden_states=processor_output["pixel_values"],
                    grid_thw=processor_output["image_grid_thw"],
                    output_hidden_states=bool(self.layer_indices),
                )
            merged_hidden_states = _merge_qwen_hidden_states(
                outputs, self.external.merger, self.layer_indices
            )
            grid_thw = processor_output["image_grid_thw"]
            tokens_per_image = (
                grid_thw[:, 1] * grid_thw[:, 2] // self.merge_size**2
            ).tolist()
            if len(set(tokens_per_image)) != 1:
                raise ValueError("Qwen encoder requires equal image grids in a batch")
            tokens_BLC = merged_hidden_states.view(
                images_BCHW.shape[0], tokens_per_image[0], self.latent_dim
            )
        elif self.external is not None and self.kind == "dino_hub":
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(self.image_size, self.image_size),
                mode="bicubic",
            )
            mean_1C11 = images_BCHW.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std_1C11 = images_BCHW.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            images_BCHW = (images_BCHW - mean_1C11) / std_1C11
            if hasattr(self.external, "forward_features"):
                features = self.external.forward_features(images_BCHW)
                tokens_BLC = features["x_norm_patchtokens"]
            else:
                tokens_BLC = self.external(images_BCHW)
        elif self.kind == "hf":
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(self._hf_processor_size, self._hf_processor_size),
                mode="bicubic",
                align_corners=False,
            )
            mean_1C11 = images_BCHW.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std_1C11 = images_BCHW.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            layer_indices = self.layer_indices
            outputs = self.external(
                pixel_values=(images_BCHW - mean_1C11) / std_1C11,
                output_hidden_states=bool(layer_indices),
            )
            prefix_length = 1 + self._hf_num_register_tokens
            if layer_indices:
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("DINO encoder did not return hidden states")
                num_layers = int(self.external.config.num_hidden_layers)
                invalid_indices = [
                    index for index in layer_indices if index < 0 or index >= num_layers
                ]
                if invalid_indices:
                    raise ValueError(
                        "DINO encoder layer indices are out of range: "
                        f"{invalid_indices}"
                    )
                selected = [
                    self.external.norm(hidden_states[index + 1])[:, prefix_length:]
                    for index in layer_indices
                ]
                tokens_BLC = torch.stack(selected).mean(dim=0)
                tokens_BLC = tokens_BLC + selected[-1].mean(dim=1, keepdim=True)
            else:
                tokens_BLC = outputs.last_hidden_state[:, prefix_length:]
        elif self.kind == "fixed":
            latent_side = self.image_size // 16
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(latent_side, latent_side),
                mode="bilinear",
                align_corners=False,
            )
            latents_BCHW = self.projection(images_BCHW)
        if self.external is not None:
            side = int(math.sqrt(tokens_BLC.shape[1]))
            if side * side != tokens_BLC.shape[1]:
                raise ValueError("RAE encoder returned a non-square token grid")
            latents_BCHW = tokens_BLC.transpose(1, 2).reshape(
                images_BCHW.shape[0], self.latent_dim, side, side
            )
        if self.latent_mean is not None or self.latent_var is not None:
            latent_mean = (
                self.latent_mean.to(latents_BCHW.device, dtype=latents_BCHW.dtype)
                if self.latent_mean is not None
                else 0
            )
            latent_var = (
                self.latent_var.to(latents_BCHW.device, dtype=latents_BCHW.dtype)
                if self.latent_var is not None
                else 1
            )
            latents_BCHW = (latents_BCHW - latent_mean) / torch.sqrt(latent_var + 1e-5)
        if add_noise and self.noise_tau > 0:
            noise_scale = self.noise_tau * torch.rand(
                (latents_BCHW.shape[0], 1, 1, 1),
                device=latents_BCHW.device,
                dtype=latents_BCHW.dtype,
            )
            latents_BCHW = latents_BCHW + noise_scale * torch.randn_like(latents_BCHW)
        return latents_BCHW


__all__ = ["RAEEncoderConfig", "FrozenRAEEncoder"]
