# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RWKV-VL HF remote-code model used by the exporter.

This file intentionally lives beside the exporter so the generated checkpoint
is self-contained.
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.models.utils import Cache
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3VLVisionModel,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

try:
    from .configuration_rwkv7 import RWKV7Config
    from .modeling_rwkv7 import RWKV7Model
except ImportError:
    from configuration_rwkv7 import RWKV7Config
    from modeling_rwkv7 import RWKV7Model


_BATCH_INVARIANT_STREAMS: dict[torch.device, list[torch.cuda.Stream]] = {}


class _BatchInvariantLinear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type != "cuda" or inputs.shape[0] == 1:
            return F.linear(inputs, self.weight, self.bias)

        streams = _BATCH_INVARIANT_STREAMS.setdefault(inputs.device, [])
        while len(streams) < inputs.shape[0]:
            streams.append(torch.cuda.Stream(device=inputs.device))

        producer_stream = torch.cuda.current_stream(inputs.device)
        outputs = []
        for sample_index in range(inputs.shape[0]):
            stream = streams[sample_index]
            stream.wait_stream(producer_stream)
            with torch.cuda.stream(stream):
                outputs.append(
                    F.linear(
                        inputs[sample_index : sample_index + 1],
                        self.weight,
                        self.bias,
                    )
                )
        for stream in streams[: inputs.shape[0]]:
            producer_stream.wait_stream(stream)
        return torch.cat(outputs, dim=0)


def _enable_batch_invariant_linears(module: nn.Module) -> int:
    converted = 0
    for child in module.modules():
        if type(child) is nn.Linear:
            child.__class__ = _BatchInvariantLinear
            converted += 1
    return converted


@dataclass
class ModRWKVProjectorConfig:
    projector_type: str = "visual"
    encoder_dim: int = 1024
    vision_dim: int | None = None
    project_dim: int = 1024
    hidden_dim: int | None = None
    num_deepstack: int = 0
    # Variant selectors. Defaults preserve the original ReLU/LayerNorm MLP
    # projector for back-compat with existing exported checkpoints.
    kind: str = "mlp"  # "mlp" | "cross_attn"
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    ffn: str = "relu"  # "relu" | "gelu" | "swiglu"
    visual_layer_indices: tuple[int, ...] = ()
    language_layer_indices: tuple[int, ...] = ()
    num_query_heads: int | None = None
    num_key_value_heads: int | None = None
    tie_qkvo: bool = True
    spatial_merge_size: int = 2
    # TokenPacker query-grid downsampling ratio when the processor's image-token
    # grid is coarser than the vision encoder's native merged-token grid.
    extra_merge_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "projector_type": self.projector_type,
            "encoder_dim": self.encoder_dim,
            "vision_dim": self.vision_dim,
            "project_dim": self.project_dim,
            "hidden_dim": self.hidden_dim,
            "num_deepstack": self.num_deepstack,
            "kind": self.kind,
            "norm": self.norm,
            "ffn": self.ffn,
            "visual_layer_indices": list(self.visual_layer_indices),
            "language_layer_indices": list(self.language_layer_indices),
            "num_query_heads": self.num_query_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "tie_qkvo": self.tie_qkvo,
            "spatial_merge_size": self.spatial_merge_size,
            "extra_merge_size": self.extra_merge_size,
        }


class ModRWKVConfig(PretrainedConfig):
    model_type = "modrwkv"
    is_composition = True

    @staticmethod
    def _to_vision_config(
        vision_config: Qwen3VLVisionConfig | dict[str, Any],
    ) -> Qwen3VLVisionConfig:
        if isinstance(vision_config, Qwen3VLVisionConfig):
            return vision_config
        if not isinstance(vision_config, dict) and hasattr(vision_config, "to_dict"):
            vision_config = vision_config.to_dict()
        if isinstance(vision_config, dict):
            if not vision_config:
                return Qwen3VLVisionConfig()
            model_type = vision_config.get("model_type")
            if model_type not in {None, Qwen3VLVisionConfig.model_type, "qwen3_vl"}:
                raise TypeError(
                    "ModRWKVConfig expects a Qwen3-VL vision config; "
                    f"got model_type={model_type!r}."
                )
            return Qwen3VLVisionConfig(
                depth=vision_config["depth"],
                hidden_size=vision_config["hidden_size"],
                hidden_act=vision_config.get("hidden_act", "gelu_pytorch_tanh"),
                intermediate_size=vision_config["intermediate_size"],
                num_heads=vision_config["num_heads"],
                in_channels=vision_config.get("in_channels", 3),
                patch_size=vision_config.get("patch_size", 16),
                spatial_merge_size=vision_config.get("spatial_merge_size", 2),
                temporal_patch_size=vision_config.get("temporal_patch_size", 2),
                out_hidden_size=vision_config["out_hidden_size"],
                num_position_embeddings=vision_config.get(
                    "num_position_embeddings",
                    2304,
                ),
                deepstack_visual_indexes=list(
                    vision_config.get("deepstack_visual_indexes")
                    or vision_config.get("deepstack_visual_indices")
                    or []
                ),
                initializer_range=vision_config.get("initializer_range", 0.02),
            )
        raise TypeError(f"Unsupported vision config type: {type(vision_config)!r}")

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: RWKV7Config | dict[str, Any],
        vision_config: Qwen3VLVisionConfig | dict[str, Any],
        projector_config: ModRWKVProjectorConfig | dict[str, Any] | None = None,
        **kwargs,
    ) -> "ModRWKVConfig":
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            projector_config=projector_config,
            **kwargs,
        )

    def __init__(
        self,
        text_config: RWKV7Config | dict[str, Any] | None = None,
        vision_config: Qwen3VLVisionConfig | dict[str, Any] | None = None,
        projector_config: ModRWKVProjectorConfig | dict[str, Any] | None = None,
        image_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        use_conv_in_projector: bool = False,
        processor_spatial_merge_size: int | None = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        # Resolved later (after vision_config is parsed) so it can default to
        # the vision encoder's spatial_merge_size when not explicitly set.
        self._processor_spatial_merge_size_override = processor_spatial_merge_size

        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = RWKV7Config(**text_config)
        self.text_config = text_config

        if vision_config is None:
            vision_config = {}
        vision_config = self._to_vision_config(vision_config)
        self.vision_config = vision_config

        if projector_config is None:
            deepstack_indexes = getattr(
                vision_config,
                "deepstack_visual_indexes",
                getattr(vision_config, "deepstack_visual_indices", []),
            )
            projector_config = ModRWKVProjectorConfig(
                encoder_dim=getattr(vision_config, "out_hidden_size", 1024),
                project_dim=getattr(text_config, "hidden_size", 1024),
                num_deepstack=len(deepstack_indexes),
            )
        elif isinstance(projector_config, dict):
            projector_config = ModRWKVProjectorConfig(**projector_config)
        self.projector_config = projector_config

        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.use_conv_in_projector = use_conv_in_projector
        # Default the processor-side spatial merge size to the vision encoder's
        # when not explicitly set. Used by the projector's optional extra
        # merger and by image_pad token counting.
        if self._processor_spatial_merge_size_override is not None:
            self.processor_spatial_merge_size = (
                self._processor_spatial_merge_size_override
            )
        else:
            self.processor_spatial_merge_size = getattr(
                vision_config, "spatial_merge_size", 2
            )

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["text_config"] = (
            self.text_config.to_dict()
            if hasattr(self.text_config, "to_dict")
            else self.text_config
        )
        output["vision_config"] = (
            self.vision_config.to_dict()
            if hasattr(self.vision_config, "to_dict")
            else self.vision_config
        )
        output["projector_config"] = (
            self.projector_config.to_dict()
            if hasattr(self.projector_config, "to_dict")
            else self.projector_config
        )
        output["image_token_id"] = self.image_token_id
        output["vision_start_token_id"] = self.vision_start_token_id
        output["vision_end_token_id"] = self.vision_end_token_id
        output["use_conv_in_projector"] = self.use_conv_in_projector
        output["processor_spatial_merge_size"] = self.processor_spatial_merge_size
        return output


class ModRWKVPreTrainedModel(PreTrainedModel):
    config_class = ModRWKVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Block"]
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]


def _build_norm(kind: str, dim: int, eps: float = 1e-5) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if kind == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm kind: {kind!r}; expected layernorm|rmsnorm")


class _SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (HF mirror of the torchtitan projector FFN)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _build_ffn(
    kind: str, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True
) -> nn.Module:
    if kind == "relu":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "gelu":
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )
    if kind == "swiglu":
        return _SwiGLUFFN(in_dim, hidden_dim, out_dim, bias=bias)
    raise ValueError(f"Unknown ffn kind: {kind!r}; expected relu|gelu|swiglu")


class _VisualStreamProjector(nn.Module):
    """HF mirror of the torchtitan MLP projector.

    When ``merge_size > 1`` the projector also performs the spatial merge:
    ``merge_size**2`` adjacent tokens are concatenated along channels and
    fed through a 2-layer MLP that maps ``encoder_dim * merge_size**2``
    channels to ``project_dim``. This subsumes the previous
    ``_ExtraPatchMerger`` module.
    """

    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: int | None = None,
        norm: str = "layernorm",
        ffn: str = "relu",
        merge_size: int = 1,
    ):
        super().__init__()
        if merge_size < 1:
            raise ValueError(f"merge_size must be >= 1; got {merge_size}")
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4
        self.merge_size = merge_size
        self.merge_unit = merge_size**2

        in_dim = encoder_dim * self.merge_unit
        self.in_norm = _build_norm(norm, encoder_dim) if merge_size > 1 else None
        self.pre_norm = _build_norm(norm, project_dim)
        self.mlp = _build_ffn(ffn, in_dim, self.hidden_dim, project_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_size > 1:
            n = x.shape[0]
            x = self.in_norm(x)
            x = x.reshape(n // self.merge_unit, self.encoder_dim * self.merge_unit)
        x = self.mlp(x)
        return x + self.pre_norm(x)


def _tokenpacker_query_seeds(
    merged_features: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    extra_merge_size: int,
) -> torch.Tensor:
    merged_counts = grid_thw.prod(-1) // (spatial_merge_size**2)
    chunks = merged_features.split([int(count) for count in merged_counts.tolist()])
    queries = []
    for chunk, grid in zip(chunks, grid_thw.tolist(), strict=True):
        temporal, height, width = (int(value) for value in grid)
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        query_height = merged_height // extra_merge_size
        query_width = merged_width // extra_merge_size
        feature_grid = chunk.view(
            temporal, merged_height, merged_width, chunk.shape[-1]
        ).permute(0, 3, 1, 2)
        query_grid = F.interpolate(
            feature_grid.float(),
            size=(query_height, query_width),
            mode="bilinear",
            align_corners=False,
        ).to(chunk.dtype)
        queries.append(query_grid.permute(0, 2, 3, 1).reshape(-1, chunk.shape[-1]))
    return torch.cat(queries, dim=0)


def _query_position_encoding(
    grid_thw: torch.Tensor,
    *,
    dim: int,
    spatial_merge_size: int,
    extra_merge_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"2D query position encoding requires dim % 4 == 0, got {dim}")
    quarter_dim = dim // 4
    frequency = torch.exp(
        -torch.arange(quarter_dim, device=device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0, device=device)) / max(quarter_dim - 1, 1))
    )
    chunks = []
    merge = spatial_merge_size * extra_merge_size
    for temporal, height, width in grid_thw.tolist():
        query_height = int(height) // merge
        query_width = int(width) // merge
        y = torch.linspace(0.0, 1.0, query_height, device=device)
        x = torch.linspace(0.0, 1.0, query_width, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
        y_phase = y_grid.reshape(-1, 1) * (2 * torch.pi) * frequency.reshape(1, -1)
        x_phase = x_grid.reshape(-1, 1) * (2 * torch.pi) * frequency.reshape(1, -1)
        spatial = torch.cat(
            [y_phase.sin(), y_phase.cos(), x_phase.sin(), x_phase.cos()], dim=-1
        )
        chunks.append(spatial.repeat(int(temporal), 1))
    return torch.cat(chunks, dim=0).to(dtype=dtype)


def _tokenpacker_memory_order(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    extra_merge_size: int,
) -> torch.Tensor:
    memory_ids = []
    region_offset = 0
    for temporal, height, width in grid_thw.tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        query_height = height // (spatial_merge_size * extra_merge_size)
        query_width = width // (spatial_merge_size * extra_merge_size)
        ids = torch.arange(
            temporal * query_height * query_width,
            device=grid_thw.device,
            dtype=torch.long,
        ).view(temporal, query_height, query_width)
        full_grid = ids.repeat_interleave(
            spatial_merge_size * extra_merge_size, dim=1
        ).repeat_interleave(spatial_merge_size * extra_merge_size, dim=2)
        memory_ids.append(
            full_grid.view(
                temporal,
                height // spatial_merge_size,
                spatial_merge_size,
                width // spatial_merge_size,
                spatial_merge_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1)
            + region_offset
        )
        region_offset += temporal * query_height * query_width
    return torch.argsort(torch.cat(memory_ids), stable=True)


class VisualAdapter(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        vision_dim: int | None,
        project_dim: int,
        hidden_dim: int | None = None,
        num_deepstack: int = 0,
        use_conv: bool = False,
        kind: str = "mlp",
        norm: str = "layernorm",
        ffn: str = "relu",
        language_layer_indices: tuple[int, ...] = (),
        num_query_heads: int | None = None,
        num_key_value_heads: int | None = None,
        tie_qkvo: bool = True,
        spatial_merge_size: int = 2,
        extra_merge_size: int = 1,
    ):
        super().__init__()
        if use_conv:
            raise ValueError("Convolutional visual projectors are not supported.")
        if extra_merge_size < 1:
            raise ValueError(f"extra_merge_size must be >= 1; got {extra_merge_size}")
        if extra_merge_size > 1 and kind != "cross_attn":
            raise ValueError(
                "extra_merge_size > 1 is only supported with kind='cross_attn'"
            )
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4
        self.num_deepstack = num_deepstack
        self.kind = kind
        self.extra_merge_size = extra_merge_size
        self.language_layer_indices = tuple(language_layer_indices)
        self.tie_qkvo = tie_qkvo
        self.spatial_merge_size = spatial_merge_size
        if kind == "mlp":
            self.main = _VisualStreamProjector(
                encoder_dim=encoder_dim,
                project_dim=project_dim,
                hidden_dim=self.hidden_dim,
                norm=norm,
                ffn=ffn,
                merge_size=extra_merge_size,
            )
            self.deepstack = nn.ModuleList(
                [
                    _VisualStreamProjector(
                        encoder_dim=encoder_dim,
                        project_dim=project_dim,
                        hidden_dim=self.hidden_dim,
                        norm=norm,
                        ffn=ffn,
                    )
                    for _ in range(num_deepstack)
                ]
            )
            return
        if kind != "cross_attn":
            raise ValueError(f"Unknown projector kind: {kind!r}")
        if norm != "layernorm":
            raise ValueError("cross_attn uses separate LayerNorms at every depth")
        self.vision_dim = vision_dim or encoder_dim
        self.num_query_heads = num_query_heads or 1
        self.num_key_value_heads = num_key_value_heads or self.num_query_heads
        if self.vision_dim % self.num_query_heads != 0:
            raise ValueError("vision_dim must be divisible by num_query_heads")
        if self.num_query_heads % self.num_key_value_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_key_value_heads")
        if len(self.language_layer_indices) != num_deepstack:
            raise ValueError("language_layer_indices must match num_deepstack")
        self.head_dim = self.vision_dim // self.num_query_heads
        kv_dim = self.num_key_value_heads * self.head_dim
        self.seed_query_norm = nn.LayerNorm(encoder_dim, eps=1e-5)
        self.seed_output_norm = nn.LayerNorm(self.vision_dim, eps=1e-5)
        self.query_norms = nn.ModuleList(
            [nn.LayerNorm(project_dim, eps=1e-5) for _ in range(num_deepstack)]
        )
        self.query_gate_projs = nn.ModuleList(
            [
                nn.Linear(project_dim, self.num_query_heads, bias=False)
                for _ in range(num_deepstack)
            ]
        )
        self.memory_norms = nn.ModuleList(
            [nn.LayerNorm(self.vision_dim, eps=1e-5) for _ in range(num_deepstack + 1)]
        )
        self.seed_q_proj = nn.Linear(encoder_dim, self.vision_dim, bias=False)
        if self.tie_qkvo:
            # One projection set is reused at every visual/RWKV depth and by
            # the TokenPacker seed retrieval. GQA independently shares each
            # KV-head slice among a group of query heads.
            self.rwkv_q_proj = nn.Linear(project_dim, self.vision_dim, bias=False)
            self.k_proj = nn.Linear(self.vision_dim, kv_dim, bias=False)
            self.v_proj = nn.Linear(self.vision_dim, kv_dim, bias=False)
            self.o_proj = nn.Linear(self.vision_dim, project_dim, bias=False)
        else:
            self.rwkv_q_projs = nn.ModuleList(
                [
                    nn.Linear(project_dim, self.vision_dim, bias=False)
                    for _ in range(num_deepstack)
                ]
            )
            self.k_projs = nn.ModuleList(
                [
                    nn.Linear(self.vision_dim, kv_dim, bias=False)
                    for _ in range(num_deepstack + 1)
                ]
            )
            self.v_projs = nn.ModuleList(
                [
                    nn.Linear(self.vision_dim, kv_dim, bias=False)
                    for _ in range(num_deepstack + 1)
                ]
            )
            self.o_projs = nn.ModuleList(
                [
                    nn.Linear(self.vision_dim, project_dim, bias=False)
                    for _ in range(num_deepstack + 1)
                ]
            )

    def _project_memory(
        self, features: torch.Tensor, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.memory_norms[depth](features)
        k_proj = self.k_proj if self.tie_qkvo else self.k_projs[depth]
        v_proj = self.v_proj if self.tie_qkvo else self.v_projs[depth]
        keys = k_proj(features).reshape(-1, self.num_key_value_heads, self.head_dim)
        values = v_proj(features).reshape(-1, self.num_key_value_heads, self.head_dim)
        return keys, values

    def _project_query_and_gate(
        self, depth: int, query_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_hidden_states = self.query_norms[depth](query_hidden_states)
        q_proj = self.rwkv_q_proj if self.tie_qkvo else self.rwkv_q_projs[depth]
        queries = q_proj(query_hidden_states)
        gates = torch.sigmoid(self.query_gate_projs[depth](query_hidden_states))
        return queries, gates

    def _project_output(self, depth: int, attended: torch.Tensor) -> torch.Tensor:
        o_proj = self.o_proj if self.tie_qkvo else self.o_projs[depth]
        return o_proj(attended)

    def attend(
        self,
        depth: int,
        query_hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        queries, gates = self._project_query_and_gate(depth, query_hidden_states)
        queries = queries.reshape(-1, self.num_query_heads, self.head_dim)
        attended = F.scaled_dot_product_attention(
            queries.transpose(0, 1).unsqueeze(0),
            keys.transpose(0, 1).unsqueeze(0),
            values.transpose(0, 1).unsqueeze(0),
            enable_gqa=self.num_query_heads != self.num_key_value_heads,
        )
        attended = attended[0].transpose(0, 1)
        attended = (attended * gates.unsqueeze(-1)).reshape(-1, self.vision_dim)
        return self._project_output(depth, attended)

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: list[torch.Tensor] | None = None,
        *,
        grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        if deepstack_features is None:
            deepstack_features = []
        expected_features = self.num_deepstack + (self.kind == "cross_attn")
        if len(deepstack_features) != expected_features:
            raise ValueError(
                f"Expected {expected_features} visual feature tensors, "
                f"got {len(deepstack_features)}."
            )
        if self.kind == "mlp":
            projected_deepstack = [
                projector(feature)
                for projector, feature in zip(self.deepstack, deepstack_features)
            ]
            return self.main(x), projected_deepstack
        if grid_thw is None:
            raise ValueError("grid_thw is required by the cross_attn projector")

        seed_features = _tokenpacker_query_seeds(
            x,
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
        )
        seed_queries = self.seed_q_proj(self.seed_query_norm(seed_features))
        seed_queries = seed_queries + _query_position_encoding(
            grid_thw,
            dim=self.vision_dim,
            spatial_merge_size=self.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
            dtype=seed_queries.dtype,
            device=seed_queries.device,
        )
        seed_keys, seed_values = self._project_memory(
            deepstack_features[-1], self.num_deepstack
        )
        memory_order = _tokenpacker_memory_order(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            extra_merge_size=self.extra_merge_size,
        )
        local_length = (self.spatial_merge_size * self.extra_merge_size) ** 2
        seed_keys = seed_keys[memory_order].view(
            seed_queries.shape[0], local_length, self.num_key_value_heads, self.head_dim
        )
        seed_values = seed_values[memory_order].view_as(seed_keys)
        local_attention = F.scaled_dot_product_attention(
            seed_queries.view(-1, self.num_query_heads, 1, self.head_dim),
            seed_keys.permute(0, 2, 1, 3),
            seed_values.permute(0, 2, 1, 3),
            enable_gqa=self.num_query_heads != self.num_key_value_heads,
        ).reshape(-1, self.vision_dim)
        projected = self._project_output(
            self.num_deepstack,
            self.seed_output_norm(seed_queries + local_attention),
        )
        memories = [
            self._project_memory(feature, depth)
            for depth, feature in enumerate(deepstack_features[:-1])
        ]
        return projected, memories


class RWKV7VLModel(ModRWKVPreTrainedModel):
    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        proj_cfg = config.projector_config
        if proj_cfg.kind == "cross_attn":
            visual_layers = tuple(int(index) for index in proj_cfg.visual_layer_indices)
            language_layers = tuple(
                int(index) for index in proj_cfg.language_layer_indices
            )
            if tuple(sorted(set(visual_layers))) != visual_layers:
                raise ValueError("visual_layer_indices must be unique and increasing")
            if tuple(sorted(set(language_layers))) != language_layers:
                raise ValueError("language_layer_indices must be unique and increasing")
            if len(visual_layers) != len(language_layers):
                raise ValueError(
                    "visual and language layer lists must have equal length"
                )
            if any(
                index < 0 or index >= config.vision_config.depth
                for index in visual_layers
            ):
                raise ValueError("visual_layer_indices select a missing ViT layer")
            if any(
                index < 0 or index >= config.text_config.num_hidden_layers
                for index in language_layers
            ):
                raise ValueError("language_layer_indices select a missing RWKV layer")
        vision_config = copy.deepcopy(config.vision_config)
        if proj_cfg.kind == "cross_attn":
            vision_config.deepstack_visual_indexes = []
        self.encoder = Qwen3VLVisionModel(vision_config)
        self.proj = VisualAdapter(
            encoder_dim=proj_cfg.encoder_dim,
            vision_dim=proj_cfg.vision_dim,
            project_dim=proj_cfg.project_dim,
            hidden_dim=proj_cfg.hidden_dim,
            num_deepstack=proj_cfg.num_deepstack,
            use_conv=config.use_conv_in_projector,
            kind=getattr(proj_cfg, "kind", "mlp"),
            norm=getattr(proj_cfg, "norm", "layernorm"),
            ffn=getattr(proj_cfg, "ffn", "relu"),
            language_layer_indices=tuple(proj_cfg.language_layer_indices),
            num_query_heads=proj_cfg.num_query_heads,
            num_key_value_heads=proj_cfg.num_key_value_heads,
            tie_qkvo=proj_cfg.tie_qkvo,
            spatial_merge_size=proj_cfg.spatial_merge_size,
            extra_merge_size=getattr(proj_cfg, "extra_merge_size", 1),
        )
        # processor_spatial_merge_size determines how many ``<image_pad>``
        # tokens the processor inserts per image. TokenPacker may use a coarser
        # query grid while the vision encoder retains its native patch layout.
        self._processor_spatial_merge_size = getattr(
            config,
            "processor_spatial_merge_size",
            getattr(self.encoder.config, "spatial_merge_size", 2),
        )
        self.llm = RWKV7Model(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, list[Any], torch.LongTensor]:
        """Run the vision encoder + projector.

        Returns ``(main_features, deepstack_levels, num_kv_per_item)`` where
        ``main_features`` has shape ``(total_image_pad_tokens, hidden_size)``
        after TokenPacker query resampling and local retrieval, and
        ``deepstack_levels`` is either:
          - ``kind='mlp'``: a list of ``(total_image_pad_tokens, hidden_size)``
            tensors to scatter-add into the LLM hidden state, or
          - ``kind='cross_attn'``: a list of ``(k, v)`` tuples each of shape
            ``(total_kv_tokens, num_heads, head_dim)``.
        ``num_kv_per_item`` is the per-image raw ViT patch count for cross
        attention (or native vision-merged count for MLP), used to build
        per-image attention masks.
        """
        captured_features = {}
        handles = []
        if self.proj.kind == "cross_attn":
            for layer_index in self.config.projector_config.visual_layer_indices:

                def capture(module, args, output, layer_index=layer_index):
                    captured_features[int(layer_index)] = output

                handles.append(
                    self.encoder.blocks[int(layer_index)].register_forward_hook(capture)
                )
        try:
            vision_output = self.encoder(pixel_values, image_grid_thw)
        finally:
            for handle in handles:
                handle.remove()
        if hasattr(vision_output, "pooler_output"):
            vision_embeds = vision_output.pooler_output
        elif hasattr(vision_output, "last_hidden_state"):
            vision_embeds = vision_output.last_hidden_state
        elif isinstance(vision_output, (tuple, list)):
            vision_embeds = vision_output[0]
        else:
            vision_embeds = vision_output

        if self.proj.kind == "cross_attn":
            deepstack_features = [
                captured_features[int(layer_index)]
                for layer_index in self.config.projector_config.visual_layer_indices
            ]
            deepstack_features.append(vision_output.last_hidden_state)
        else:
            deepstack_features = getattr(vision_output, "deepstack_features", None)
            if deepstack_features is None:
                deepstack_features = []
        projected, projected_deepstack = self.proj(
            vision_embeds,
            list(deepstack_features),
            grid_thw=image_grid_thw,
        )
        projected = projected.reshape(-1, self.config.text_config.hidden_size)

        vision_merge = getattr(self.encoder.config, "spatial_merge_size", 2)
        processor_merge = self._processor_spatial_merge_size
        num_kv_per_item = image_grid_thw.prod(-1)
        if self.proj.kind == "mlp":
            num_kv_per_item = num_kv_per_item // (vision_merge**2)
        expected_main_tokens = int(
            (image_grid_thw.prod(-1) // (processor_merge**2)).sum().item()
        )
        if expected_main_tokens != projected.shape[0]:
            raise ValueError(
                "Projected image features and image grid do not match: "
                f"features={projected.shape[0]} grid_tokens={expected_main_tokens}"
            )

        if self.proj.kind == "mlp":
            projected_deepstack = [
                feature.reshape(-1, self.config.text_config.hidden_size)
                for feature in projected_deepstack
            ]
        # For cross_attn, projected_deepstack is a list of (k, v) tuples
        # already shaped (total_kv_tokens, num_heads, head_dim); leave as-is.

        if projected.numel() == 0:
            empty = torch.empty(
                0,
                self.config.text_config.hidden_size,
                device=self.get_input_embeddings().weight.device,
            )
            return empty, [], num_kv_per_item
        return projected, projected_deepstack, num_kv_per_item

    def _inject_image_features(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        image_mask = input_ids == self.config.image_token_id
        if image_mask.sum().item() != image_features.shape[0]:
            raise ValueError(
                "Image features and image placeholder tokens do not match: "
                f"tokens={image_mask.sum().item()} features={image_features.shape[0]}"
            )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[image_mask] = image_features.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        return inputs_embeds

    def _add_image_features(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ) -> torch.FloatTensor:
        image_mask = input_ids == self.config.image_token_id
        if image_mask.sum().item() != image_features.shape[0]:
            raise ValueError(
                "DeepStack features and image placeholder tokens do not match: "
                f"tokens={image_mask.sum().item()} features={image_features.shape[0]}"
            )
        hidden_states = hidden_states.clone()
        hidden_states[image_mask] += image_features.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return hidden_states

    def _cross_attn_image_features(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        kv_pair: tuple[torch.Tensor, torch.Tensor],
        *,
        num_kv_per_item: torch.LongTensor,
        num_tokens_per_item: torch.LongTensor,
        depth: int,
    ) -> torch.FloatTensor:
        """Cross-attention deepstack injection (HF inference path).

        Q is gathered from ``<image_pad>`` positions in the LLM hidden state;
        K/V come from the projector's per-level deepstack output. The
        attention mask is block-diagonal so each query attends only to its
        own image's K/V.
        """
        flat_input = input_ids.view(-1)
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1]).clone()
        q_positions = (
            (flat_input == self.config.image_token_id)
            .nonzero(as_tuple=False)
            .reshape(-1)
        )
        if q_positions.numel() == 0:
            return hidden_states
        if q_positions.shape[0] != int(num_tokens_per_item.sum().item()):
            raise ValueError(
                "image_pad token count does not match expected per-image counts: "
                f"got {q_positions.shape[0]}, expected "
                f"{int(num_tokens_per_item.sum().item())}"
            )

        k_real, v_real = kv_pair
        kv_total = int(num_kv_per_item.sum().item())
        k_real = k_real[:kv_total]
        v_real = v_real[:kv_total]
        query_cursor = 0
        memory_cursor = 0
        for query_count, memory_count in zip(
            num_tokens_per_item.tolist(),
            num_kv_per_item.tolist(),
            strict=True,
        ):
            query_count, memory_count = int(query_count), int(memory_count)
            positions = q_positions[query_cursor : query_cursor + query_count]
            query_states = flat_hidden.index_select(0, positions)
            keys = k_real[memory_cursor : memory_cursor + memory_count]
            values = v_real[memory_cursor : memory_cursor + memory_count]
            delta = self.proj.attend(
                depth,
                query_states.to(keys.dtype),
                keys,
                values,
            )
            flat_hidden.index_add_(0, positions, delta.to(flat_hidden.dtype))
            query_cursor += query_count
            memory_cursor += memory_count
        return flat_hidden.view_as(hidden_states)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.text_config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.text_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.text_config.output_hidden_states
        )
        use_cache = (
            use_cache
            if use_cache is not None
            else (self.config.text_config.use_cache if not self.training else False)
        )
        if output_attentions:
            warnings.warn(
                "`RWKV7Model` does not support `output_attentions`; setting it to `False`."
            )
            output_attentions = False

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError(
                "pixel_values and image_grid_thw must be provided together."
            )
        if pixel_values is not None and input_ids is None:
            raise ValueError("input_ids are required when pixel_values are provided.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        deepstack_features: list[Any] = []
        num_kv_per_item: torch.Tensor | None = None
        num_tokens_per_item: torch.Tensor | None = None
        if pixel_values is not None:
            (
                image_features,
                deepstack_features,
                num_kv_per_item,
            ) = self._get_image_features(pixel_values, image_grid_thw)
            num_tokens_per_item = image_grid_thw.prod(-1) // (
                self._processor_spatial_merge_size**2
            )
            inputs_embeds = self._inject_image_features(
                input_ids,
                inputs_embeds,
                image_features,
            )

        if use_cache and not isinstance(past_key_values, Cache):
            from_legacy_cache = getattr(Cache, "from_legacy_cache", None)
            if callable(from_legacy_cache):
                past_key_values = from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        hidden_states = inputs_embeds
        v_first = torch.zeros_like(hidden_states)
        depth_by_layer = {
            layer_index: depth
            for depth, layer_index in enumerate(self.proj.language_layer_indices)
        }
        for layer_idx, layer in enumerate(self.llm.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states, attentions, past_key_values, v_first = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                v_first=v_first,
                **kwargs,
            )
            if self.proj.kind == "cross_attn":
                depth = depth_by_layer.get(layer_idx)
                if depth is not None:
                    hidden_states = self._cross_attn_image_features(
                        input_ids,
                        hidden_states,
                        deepstack_features[depth],
                        num_kv_per_item=num_kv_per_item,
                        num_tokens_per_item=num_tokens_per_item,
                        depth=depth,
                    )
            elif layer_idx < len(deepstack_features):
                hidden_states = self._add_image_features(
                    input_ids,
                    hidden_states,
                    deepstack_features[layer_idx],
                )
            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.llm.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                item
                for item in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attns,
                ]
                if item is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class RWKV7VLForConditionalGeneration(ModRWKVPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        self.model = RWKV7VLModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def enable_batch_invariant_decode(self) -> int:
        return _enable_batch_invariant_linears(
            self.model.llm
        ) + _enable_batch_invariant_linears(self.lm_head)

    @staticmethod
    def _has_nonempty_past_key_values(
        past_key_values: Any | None,
        cache_position: torch.LongTensor | None,
    ) -> bool:
        if past_key_values is None:
            return False
        if cache_position is not None:
            return cache_position.numel() > 0 and cache_position[0].item() > 0

        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return get_seq_length() > 0
            except (AttributeError, TypeError):
                pass

        try:
            return len(past_key_values) > 0
        except TypeError:
            return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        use_cache: bool | None = True,
        logits_to_keep: int | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        has_past = self._has_nonempty_past_key_values(
            past_key_values,
            cache_position,
        )

        if has_past and input_ids is not None:
            if (
                cache_position is not None
                and input_ids.shape[1] != cache_position.shape[0]
            ):
                input_ids = input_ids[:, cache_position]
            else:
                input_ids = input_ids[:, -1:]

        model_inputs: dict[str, Any] = {
            "input_ids": input_ids.contiguous() if input_ids is not None else None,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

        if inputs_embeds is not None and not has_past:
            model_inputs["inputs_embeds"] = inputs_embeds

        if not has_past:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw

        if cache_position is not None:
            model_inputs["cache_position"] = cache_position
        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        labels: torch.LongTensor | None = None,
        shift_labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.text_config.use_return_dict
        )
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(
            hidden_states
            if logits_to_keep is None
            else hidden_states[:, -logits_to_keep:]
        )

        loss = None
        if labels is not None or shift_labels is not None:
            if shift_labels is None:
                ignore = torch.full_like(labels[:, :1], -100)
                shift_labels = torch.cat((labels[..., 1:], ignore), dim=1)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.shape[-1]),
                shift_labels.to(logits.device).reshape(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoConfig.register(ModRWKVConfig.model_type, ModRWKVConfig, exist_ok=True)
AutoModel.register(ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True)
AutoModelForCausalLM.register(
    ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True
)
AutoModelForImageTextToText.register(
    ModRWKVConfig,
    RWKV7VLForConditionalGeneration,
    exist_ok=True,
)

ModRWKVConfig.register_for_auto_class("AutoConfig")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModel")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
