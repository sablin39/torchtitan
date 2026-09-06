# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    tokens_per_item: torch.Tensor | None = None,
) -> torch.Tensor:
    """Merge selected Qwen vision blocks with RAEv2 multi-layer-sum semantics.

    Each selected block output is normalized by the merger's LayerNorm (the
    Qwen vision tower has no separate final norm), the selected layers are
    averaged, and the per-item token mean of the final selected layer is added
    back as a global signal. The merger MLP runs once on the combined tokens.
    ``tokens_per_item`` holds the pre-merger token counts of each packed media
    item so the global mean stays within its own image or video.
    """
    if not layer_indices:
        return outputs.pooler_output
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Qwen vision model did not return hidden states")
    for attribute in ("norm", "linear_fc1", "act_fn", "linear_fc2", "hidden_size"):
        if not hasattr(merger, attribute):
            raise ValueError(
                "Qwen multi-layer merging requires a PatchMerger with norm, "
                "linear_fc1, act_fn, linear_fc2, and hidden_size"
            )
    if getattr(merger, "use_postshuffle_norm", False):
        raise ValueError("Qwen multi-layer merging requires a pre-shuffle merger norm")
    selected = [hidden_states[index + 1] for index in layer_indices]
    normed = [merger.norm(hidden) for hidden in selected]
    merged = torch.stack(normed).mean(dim=0)
    global_signal = normed[-1]
    if tokens_per_item is not None:
        counts = tokens_per_item.reshape(-1).to(
            device=global_signal.device, dtype=torch.long
        )
        if int(counts.sum().item()) != global_signal.shape[0]:
            raise ValueError(
                "Qwen tokens_per_item does not match the packed token count"
            )
        # Deterministic contiguous-segment means: a prefix sum keeps a fixed
        # accumulation order (index_add_ would use non-deterministic atomics).
        prefix = torch.cat(
            [
                global_signal.new_zeros(1, global_signal.shape[-1]),
                global_signal.float().cumsum(dim=0),
            ],
            dim=0,
        )
        ends = counts.cumsum(dim=0)
        starts = ends - counts
        segment_sums = prefix[ends] - prefix[starts]
        means = segment_sums / counts.clamp_min(1).unsqueeze(-1)
        item_ids = torch.repeat_interleave(
            torch.arange(counts.numel(), device=global_signal.device), counts
        )
        global_signal = means[item_ids].to(global_signal.dtype)
    else:
        global_signal = global_signal.mean(dim=0, keepdim=True)
    merged = merged + global_signal
    merged = merger.linear_fc2(
        merger.act_fn(merger.linear_fc1(merged.view(-1, merger.hidden_size)))
    )
    return merged


@dataclass(frozen=True, slots=True)
class RAEEncoderConfig:
    """Frozen encoder settings; Qwen accepts ``image_size=-1`` dynamically."""

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
    dtype: str = "float32"
    attn_implementation: str = "flash_attention_2"
    compile: bool = False

    def __post_init__(self) -> None:
        if self.image_size == 0 or (self.image_size < -1):
            raise ValueError("encoder.image_size must be -1 or positive")
        if self.kind != "qwen" and self.image_size == -1:
            raise ValueError("encoder.image_size=-1 is only supported for Qwen")
        if self.merge_size <= 0:
            raise ValueError("encoder.merge_size must be positive")
        if self.image_size != -1 and self.image_size % self.merge_size:
            raise ValueError("encoder.image_size must be divisible by merge_size")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported encoder dtype: {self.dtype}")


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
        self.supervision_image_size = (
            None if config.image_size == -1 else config.image_size // config.merge_size
        )
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
        self.last_grid_thw: torch.Tensor | None = None
        self.last_fps: torch.Tensor | None = None
        self.last_temporal_start: torch.Tensor | None = None
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
        if config.compile:
            if self.external is None:
                raise ValueError("encoder.compile requires an external backbone")
            self.external = torch.compile(self.external, dynamic=True)

    def _init_qwen(self, config: RAEEncoderConfig, device: torch.device) -> None:
        if not config.name:
            raise ValueError("Qwen encoder name is required")
        if config.checkpoint_path is not None:
            raise ValueError("Qwen encoder does not accept checkpoint_path")
        try:
            from safetensors import safe_open
            from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
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
        # The HF default (sdpa) splits the packed sequence per document and
        # runs one attention call per image per block; flash attention consumes
        # cu_seqlens in a single varlen kernel.
        vision_config._attn_implementation = config.attn_implementation
        if vision_config.spatial_merge_size != config.merge_size:
            raise ValueError(
                "encoder.merge_size does not match Qwen vision config: "
                f"{config.merge_size} != {vision_config.spatial_merge_size}"
            )
        if config.image_size != -1 and config.image_size % (
            vision_config.patch_size * config.merge_size
        ):
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
        encoder_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        self.external = visual.to(device=device, dtype=encoder_dtype).eval()
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(model_directory), local_files_only=True
            )
        except (OSError, ValueError):
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

    @staticmethod
    def _as_btchw(media: Any) -> torch.Tensor:
        if not isinstance(media, torch.Tensor):
            import io

            import numpy as np
            from PIL import Image

            if isinstance(media, (bytes, bytearray)):
                media = Image.open(io.BytesIO(media)).convert("RGB")
            if hasattr(media, "convert"):
                media = np.array(media.convert("RGB"), copy=True)
            if isinstance(media, (list, tuple)):
                frames = [FrozenRAEEncoder._as_btchw(frame)[0] for frame in media]
                return torch.stack(frames, dim=0)
            media = torch.from_numpy(np.asarray(media))
        media = media.float()
        if media.numel() and media.max() > 1:
            media = media / 255.0
        if media.ndim == 3:
            if media.shape[-1] in (1, 3, 4):
                media = media[..., :3].permute(2, 0, 1)
            elif media.shape[0] not in (1, 3, 4):
                raise ValueError("RAE media must be HWC or CHW with three channels")
            else:
                media = media[:3]
            return media.unsqueeze(0)
        if media.ndim == 4:
            if media.shape[1] in (1, 3, 4):
                return media[:, :3]
            if media.shape[-1] in (1, 3, 4):
                return media[..., :3].permute(0, 3, 1, 2)
            raise ValueError("RAE videos must use TCHW or THWC layout")
        if media.ndim == 5:
            if media.shape[0] != 1 or media.shape[2] not in (1, 3, 4):
                raise ValueError("RAE media batches must use one BTCHW item")
            return media[0, :, :3]
        raise ValueError("RAE media must have CHW, TCHW, or BTCHW dimensions")

    def forward(
        self,
        images_BTCHW: torch.Tensor | Sequence[torch.Tensor] | Mapping[str, Any],
        *,
        add_noise: bool = False,
        return_grid_thw: bool = False,
        fps: torch.Tensor | float | None = None,
        temporal_start: torch.Tensor | float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self.last_grid_thw = None
        self.last_fps = None
        self.last_temporal_start = None

        def scalar_values(
            value: torch.Tensor | float | None,
            batch_size: int,
            default: float,
        ) -> list[float]:
            if value is None:
                return [default] * batch_size
            if isinstance(value, torch.Tensor):
                values = value.detach().float().flatten().tolist()
            else:
                values = [float(value)]
            if len(values) == 1:
                values *= batch_size
            if len(values) != batch_size:
                raise ValueError("RAE metadata must be scalar or one value per sample")
            return [float(item) for item in values]

        processor_output: Mapping[str, Any] | None = None
        images_BCHW: torch.Tensor | None = None
        batch_size: int
        media_kind = "image"
        if isinstance(images_BTCHW, Mapping):
            if self.kind != "qwen":
                raise ValueError(
                    "Preprocessed Qwen mappings require encoder.kind='qwen'"
                )
            processor_output = images_BTCHW
            pixel_key = next(
                (
                    key
                    for key in ("pixel_values", "pixel_values_videos", "input")
                    if processor_output.get(key) is not None
                ),
                None,
            )
            grid_key = next(
                (
                    key
                    for key in (
                        "image_grid_thw",
                        "video_grid_thw",
                        "grid_thw",
                        "grid_thw_videos",
                    )
                    if processor_output.get(key) is not None
                ),
                None,
            )
            if pixel_key is None or grid_key is None:
                raise ValueError(
                    "Qwen encoder mappings require pixel_values and grid_thw metadata"
                )
            media_kind = str(
                processor_output.get(
                    "media_kind", "video" if "video" in pixel_key else "image"
                )
            )
            grid_thw = torch.as_tensor(processor_output[grid_key])
            batch_size = int(grid_thw.reshape(-1, 3).shape[0])
        elif isinstance(images_BTCHW, torch.Tensor):
            raw_media = images_BTCHW.float()
            if raw_media.numel() and raw_media.max() > 1:
                raw_media = raw_media / 255.0
            if self.kind == "qwen":
                if raw_media.ndim == 4:
                    images_BCHW = raw_media
                    batch_size = raw_media.shape[0]
                    processor_output = self.processor(
                        images=raw_media.detach(),
                        do_rescale=False,
                        return_tensors="pt",
                    )
                elif raw_media.ndim == 5:
                    media_kind = "video"
                    videos_TCHW = [
                        raw_media[index].detach() for index in range(raw_media.shape[0])
                    ]
                    batch_size = len(videos_TCHW)
                    if not hasattr(self.processor, "video_processor"):
                        raise RuntimeError(
                            "Qwen processor does not provide a video processor"
                        )
                    processor_output = self.processor(
                        videos=videos_TCHW,
                        do_rescale=False,
                        do_sample_frames=False,
                        return_metadata=True,
                        return_tensors="pt",
                    )
                else:
                    raise ValueError("Qwen encoder expects BCHW images or BTCHW videos")
            else:
                if raw_media.ndim == 5:
                    if raw_media.shape[1] != 1:
                        raise ValueError(
                            "Non-Qwen RAE encoders only accept one-frame BTCHW images"
                        )
                    raw_media = raw_media[:, 0]
                if raw_media.ndim != 4:
                    raise ValueError("Non-Qwen RAE encoders expect BCHW images")
                images_BCHW = raw_media
                batch_size = raw_media.shape[0]
        else:
            media_items = list(images_BTCHW)
            if not media_items:
                raise ValueError("RAE encoder requires at least one image or video")
            if self.kind != "qwen":
                if any(item.ndim not in (3, 4) for item in media_items):
                    raise ValueError(
                        "Non-Qwen RAE encoders expect CHW or one-frame TCHW images"
                    )
                if any(item.ndim == 4 and item.shape[0] != 1 for item in media_items):
                    raise ValueError(
                        "Non-Qwen RAE encoders only accept one-frame TCHW images"
                    )
                images_BCHW = torch.stack(
                    [
                        item.float() if item.ndim == 3 else item[0].float()
                        for item in media_items
                    ]
                )
                batch_size = images_BCHW.shape[0]
            else:
                media_items = [self._as_btchw(item) for item in media_items]
                media_kind = (
                    "video"
                    if any(item.shape[0] > 1 for item in media_items)
                    else "image"
                )
                batch_size = len(media_items)
                if media_kind == "video":
                    if not hasattr(self.processor, "video_processor"):
                        raise RuntimeError(
                            "Qwen processor does not provide a video processor"
                        )
                    processor_output = self.processor(
                        videos=[item.detach() for item in media_items],
                        do_rescale=False,
                        do_sample_frames=False,
                        return_metadata=True,
                        return_tensors="pt",
                    )
                else:
                    processor_output = self.processor(
                        images=[item[0].detach() for item in media_items],
                        do_rescale=False,
                        return_tensors="pt",
                    )

        if self.kind != "qwen" and images_BCHW is not None:
            if images_BCHW.shape[1] != 3:
                raise ValueError("RAE images must have three channels")
            if images_BCHW.shape[-2:] != (self.image_size, self.image_size):
                images_BCHW = F.interpolate(
                    images_BCHW,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

        tokens_BLC: torch.Tensor | None = None
        latents_BCHW: torch.Tensor | None = None
        if self.external is not None and self.kind == "qwen":
            assert processor_output is not None
            pixel_key = next(
                key
                for key in ("pixel_values", "pixel_values_videos", "input")
                if processor_output.get(key) is not None
            )
            grid_key = next(
                key
                for key in (
                    "image_grid_thw",
                    "video_grid_thw",
                    "grid_thw",
                    "grid_thw_videos",
                )
                if processor_output.get(key) is not None
            )
            grid_thw = torch.as_tensor(processor_output[grid_key])
            external_device = next(self.external.parameters()).device
            external_dtype = next(self.external.parameters()).dtype
            model_inputs = {
                name: (
                    value.to(device=external_device, dtype=external_dtype)
                    if name == pixel_key
                    else value.to(device=external_device)
                )
                for name, value in processor_output.items()
                if name == pixel_key or name == grid_key
            }
            with torch.no_grad():
                outputs = self.external(
                    hidden_states=model_inputs[pixel_key],
                    grid_thw=model_inputs[grid_key],
                    output_hidden_states=bool(self.layer_indices),
                )
            merged_hidden_states = _merge_qwen_hidden_states(
                outputs,
                self.external.merger,
                self.layer_indices,
                tokens_per_item=grid_thw.reshape(-1, 3).prod(dim=-1),
            )
            grid_thw = grid_thw.to(
                device=merged_hidden_states.device, dtype=torch.long
            ).reshape(-1, 3)
            tokens_per_item = grid_thw.prod(dim=-1) // self.merge_size**2
            post_merge_grid_thw = grid_thw.clone()
            post_merge_grid_thw[:, 1:] //= self.merge_size
            self.last_grid_thw = post_merge_grid_thw.detach().cpu()
            same_grid = torch.all(post_merge_grid_thw == post_merge_grid_thw[0])
            if torch.all(tokens_per_item == tokens_per_item[0]) and same_grid:
                tokens_BLC = merged_hidden_states.view(
                    batch_size, int(tokens_per_item[0].item()), self.latent_dim
                )
            else:
                tokens_BLC = merged_hidden_states
            fps_source = processor_output.get("fps") if fps is None else fps
            if fps_source is None and media_kind == "video":
                metadata = processor_output.get("video_metadata")
                metadata_fps = (
                    [getattr(item, "fps", None) for item in metadata]
                    if metadata
                    else []
                )
                if len(metadata_fps) != batch_size or any(
                    item_fps is None or item_fps <= 0 for item_fps in metadata_fps
                ):
                    raise ValueError(
                        "Video FPS must be provided with the input or processor metadata"
                    )
                fps_values = [float(item_fps) for item_fps in metadata_fps]
            else:
                fps_values = scalar_values(fps_source, batch_size, 0.0)
            if media_kind == "video" and any(value <= 0 for value in fps_values):
                raise ValueError("Video FPS must be positive")
            self.last_fps = torch.tensor(fps_values, device=merged_hidden_states.device)
            start_values = scalar_values(temporal_start, batch_size, 0.0)
            self.last_temporal_start = torch.tensor(
                start_values, device=merged_hidden_states.device
            )
        elif self.external is not None and self.kind == "dino_hub":
            assert images_BCHW is not None
            images_BCHW = F.interpolate(
                images_BCHW, size=(self.image_size, self.image_size), mode="bicubic"
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
            assert images_BCHW is not None
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
            assert images_BCHW is not None
            latent_side = self.image_size // 16
            images_BCHW = F.interpolate(
                images_BCHW,
                size=(latent_side, latent_side),
                mode="bilinear",
                align_corners=False,
            )
            latents_BCHW = self.projection(images_BCHW)

        if self.external is not None:
            if tokens_BLC is None:
                raise RuntimeError("RAE external encoder did not return tokens")
            if tokens_BLC.ndim == 3 and self.kind == "qwen" and return_grid_thw:
                latents = tokens_BLC.reshape(-1, tokens_BLC.shape[-1])
            elif tokens_BLC.ndim == 3:
                side = int(math.sqrt(tokens_BLC.shape[1]))
                if side * side != tokens_BLC.shape[1]:
                    if return_grid_thw:
                        latents = tokens_BLC
                    else:
                        raise ValueError(
                            "Qwen encoder returned a non-square token grid; "
                            "request return_grid_thw=True for variable resolution"
                        )
                else:
                    latents_BCHW = tokens_BLC.transpose(1, 2).reshape(
                        batch_size, self.latent_dim, side, side
                    )
                    latents = latents_BCHW
            else:
                latents = tokens_BLC
        else:
            if latents_BCHW is None:
                raise RuntimeError("RAE encoder did not return image latents")
            latents = latents_BCHW
        if self.latent_mean is not None or self.latent_var is not None:
            if latents.ndim == 2:
                mean_shape = (1, self.latent_dim)
                variance_shape = (1, self.latent_dim)
            elif latents.ndim == 3:
                mean_shape = (1, 1, self.latent_dim)
                variance_shape = (1, 1, self.latent_dim)
            else:
                mean_shape = (
                    self.latent_mean.shape
                    if self.latent_mean is not None
                    else (1, self.latent_dim, 1, 1)
                )
                variance_shape = (
                    self.latent_var.shape
                    if self.latent_var is not None
                    else (1, self.latent_dim, 1, 1)
                )
            latent_mean = self.latent_mean
            if latent_mean is not None:
                latent_mean = latent_mean.to(latents.device, dtype=latents.dtype)
                if latents.ndim < 4 and latent_mean.ndim == 4:
                    latent_mean = latent_mean.mean(dim=(2, 3))
                latent_mean = latent_mean.reshape(mean_shape)
            else:
                latent_mean = 0
            latent_var = self.latent_var
            if latent_var is not None:
                latent_var = latent_var.to(latents.device, dtype=latents.dtype)
                if latents.ndim < 4 and latent_var.ndim == 4:
                    latent_var = latent_var.mean(dim=(2, 3))
                latent_var = latent_var.reshape(variance_shape)
            else:
                latent_var = 1
            latents = (latents - latent_mean) / torch.sqrt(latent_var + 1e-5)
        if add_noise and self.noise_tau > 0:
            if latents.ndim == 4:
                noise_scale = self.noise_tau * torch.rand(
                    (latents.shape[0], 1, 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
            elif latents.ndim == 3:
                noise_scale = self.noise_tau * torch.rand(
                    (latents.shape[0], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
            elif self.last_grid_thw is not None:
                # Packed (T, C) latents: one noise scale per packed media item.
                tokens_per_item = (
                    self.last_grid_thw.reshape(-1, 3)
                    .prod(dim=-1)
                    .to(device=latents.device)
                )
                if int(tokens_per_item.sum().item()) != latents.shape[0]:
                    raise ValueError(
                        "Qwen grid metadata does not match the packed latent count"
                    )
                noise_scale = torch.repeat_interleave(
                    self.noise_tau
                    * torch.rand(
                        tokens_per_item.numel(),
                        device=latents.device,
                        dtype=latents.dtype,
                    ),
                    tokens_per_item,
                ).unsqueeze(-1)
            else:
                noise_scale = self.noise_tau * torch.rand(
                    (1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
            latents = latents + noise_scale * torch.randn_like(latents)
        if return_grid_thw:
            if self.last_grid_thw is None:
                if latents.ndim == 4:
                    self.last_grid_thw = torch.tensor(
                        [[1, latents.shape[-2], latents.shape[-1]]],
                        dtype=torch.long,
                    )
                else:
                    raise ValueError(
                        "return_grid_thw requires Qwen grid metadata for token latents"
                    )
            return latents, self.last_grid_thw.to(latents.device)
        return latents


__all__ = ["RAEEncoderConfig", "FrozenRAEEncoder"]
