# -*- coding: utf-8 -*-
"""RWKV-VL HF remote-code model used by the exporter.

This file intentionally lives beside the exporter so the generated checkpoint
is self-contained.
"""

from dataclasses import dataclass
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
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
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

try:
    from .configuration_rwkv7 import RWKV7Config
    from .modeling_rwkv7 import RWKV7Model
except ImportError:
    from configuration_rwkv7 import RWKV7Config
    from modeling_rwkv7 import RWKV7Model


@dataclass
class ModRWKVProjectorConfig:
    projector_type: str = "visual"
    encoder_dim: int = 1024
    project_dim: int = 1024
    hidden_dim: Optional[int] = None
    num_deepstack: int = 0
    # Variant selectors. Defaults preserve the original ReLU/LayerNorm MLP
    # projector for back-compat with existing exported checkpoints.
    kind: str = "mlp"  # "mlp" | "cross_attn"
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    ffn: str = "relu"  # "relu" | "gelu" | "swiglu"
    # cross_attn-only fields.
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    # Extra PatchMerger on the main stream when the processor's spatial merge
    # size is coarser than the vision encoder's. ``1`` (default) means the
    # extra merger is omitted entirely.
    extra_merge_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "projector_type": self.projector_type,
            "encoder_dim": self.encoder_dim,
            "project_dim": self.project_dim,
            "hidden_dim": self.hidden_dim,
            "num_deepstack": self.num_deepstack,
            "kind": self.kind,
            "norm": self.norm,
            "ffn": self.ffn,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "extra_merge_size": self.extra_merge_size,
        }


class ModRWKVConfig(PretrainedConfig):
    model_type = "modrwkv"
    is_composition = True

    @staticmethod
    def _to_vision_config(
        vision_config: Union[Qwen3VLVisionConfig, Dict[str, Any]],
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
        text_config: Union[RWKV7Config, Dict[str, Any]],
        vision_config: Union[Qwen3VLVisionConfig, Dict[str, Any]],
        projector_config: Optional[Union[ModRWKVProjectorConfig, Dict[str, Any]]] = None,
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
        text_config: Optional[Union[RWKV7Config, Dict[str, Any]]] = None,
        vision_config: Optional[Union[Qwen3VLVisionConfig, Dict[str, Any]]] = None,
        projector_config: Optional[Union[ModRWKVProjectorConfig, Dict[str, Any]]] = None,
        image_token_id: int = 65532,
        vision_start_token_id: int = 65530,
        vision_end_token_id: int = 65531,
        tie_word_embeddings: bool = False,
        use_conv_in_projector: bool = False,
        processor_spatial_merge_size: Optional[int] = None,
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
        hidden_dim: Optional[int] = None,
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
        self.in_norm = (
            _build_norm(norm, encoder_dim) if merge_size > 1 else None
        )
        self.pre_norm = _build_norm(norm, project_dim)
        self.mlp = _build_ffn(ffn, in_dim, self.hidden_dim, project_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_size > 1:
            n = x.shape[0]
            x = self.in_norm(x)
            x = x.reshape(n // self.merge_unit, self.encoder_dim * self.merge_unit)
        x = self.mlp(x)
        return x + self.pre_norm(x)


class _VisualStreamCrossAttnProjector(nn.Module):
    """HF mirror of the torchtitan cross-attn projector. Uses plain SDPA at
    inference time (no triton dependency). Minimal block: pre-norm Q,
    cross-attention, output projection — no post-attn FFN.
    """

    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        num_heads: int,
        head_dim: int,
        norm: str = "layernorm",
    ):
        super().__init__()
        if num_heads * head_dim != project_dim:
            raise ValueError(
                f"cross_attn projector requires num_heads*head_dim "
                f"({num_heads}*{head_dim}) == project_dim ({project_dim})"
            )
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_norm = _build_norm(norm, encoder_dim)
        self.k_proj = nn.Linear(encoder_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(encoder_dim, num_heads * head_dim, bias=False)
        self.q_norm = _build_norm(norm, project_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, project_dim, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.kv_norm(x)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        return k, v

    def attend(
        self,
        q_real: torch.Tensor,
        k_real: torch.Tensor,
        v_real: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA cross-attention with a dense (Q_real, KV_real) bool mask.

        Args:
            q_real: ``(Q_real, project_dim)``.
            k_real: ``(KV_real, num_heads, head_dim)``.
            v_real: ``(KV_real, num_heads, head_dim)``.
            attn_mask: ``(Q_real, KV_real)`` bool, ``True`` = attend.

        Returns the per-query delta to ADD into the LLM hidden state.
        """
        q = self.q_norm(q_real).reshape(-1, self.num_heads, self.head_dim)
        # SDPA wants (B, H, S, D); pack Q_real into a single batch.
        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k_real.transpose(0, 1).unsqueeze(0)
        v4 = v_real.transpose(0, 1).unsqueeze(0)
        # attn_mask comes in as (Q, KV); broadcast to (1, 1, Q, KV).
        mask4 = attn_mask[None, None, :, :]
        attn = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask4)
        attn = attn.squeeze(0).transpose(0, 1).reshape(q_real.shape[0], -1)
        return self.o_proj(attn)


class VisualAdapter(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: Optional[int] = None,
        num_deepstack: int = 0,
        use_conv: bool = False,
        kind: str = "mlp",
        norm: str = "layernorm",
        ffn: str = "relu",
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        extra_merge_size: int = 1,
    ):
        super().__init__()
        if use_conv:
            raise ValueError("Convolutional visual projectors are not supported.")
        if extra_merge_size < 1:
            raise ValueError(
                f"extra_merge_size must be >= 1; got {extra_merge_size}"
            )
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

        # The main projector also performs the optional extra spatial merge
        # via its own MLP (PatchMerger pattern). No separate ``extra_merger``
        # module is needed on the main path.
        self.main = _VisualStreamProjector(
            encoder_dim=encoder_dim,
            project_dim=project_dim,
            hidden_dim=self.hidden_dim,
            norm=norm,
            ffn=ffn,
            merge_size=extra_merge_size,
        )
        if kind == "mlp":
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
        elif kind == "cross_attn":
            if num_heads is None:
                raise ValueError(
                    "VisualAdapter num_heads is required when kind='cross_attn'"
                )
            resolved_head_dim = head_dim or (project_dim // num_heads)
            if num_heads * resolved_head_dim != project_dim:
                raise ValueError(
                    f"cross_attn projector requires num_heads*head_dim "
                    f"({num_heads}*{resolved_head_dim}) == project_dim "
                    f"({project_dim})"
                )
            self.deepstack = nn.ModuleList(
                [
                    _VisualStreamCrossAttnProjector(
                        encoder_dim=encoder_dim,
                        project_dim=project_dim,
                        num_heads=num_heads,
                        head_dim=resolved_head_dim,
                        norm=norm,
                    )
                    for _ in range(num_deepstack)
                ]
            )
        else:
            raise ValueError(f"Unknown projector kind: {kind!r}")

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[Any]]:
        if deepstack_features is None:
            deepstack_features = []
        if len(deepstack_features) != self.num_deepstack:
            raise ValueError(
                f"Expected {self.num_deepstack} DeepStack feature tensors, "
                f"got {len(deepstack_features)}."
            )
        projected_deepstack = [
            projector(feature)
            for projector, feature in zip(self.deepstack, deepstack_features)
        ]
        return self.main(x), projected_deepstack


class RWKV7VLModel(ModRWKVPreTrainedModel):
    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        self.encoder = Qwen3VLVisionModel(config.vision_config)

        proj_cfg = config.projector_config
        self.proj = VisualAdapter(
            encoder_dim=proj_cfg.encoder_dim,
            project_dim=proj_cfg.project_dim,
            hidden_dim=proj_cfg.hidden_dim,
            num_deepstack=proj_cfg.num_deepstack,
            use_conv=config.use_conv_in_projector,
            kind=getattr(proj_cfg, "kind", "mlp"),
            norm=getattr(proj_cfg, "norm", "layernorm"),
            ffn=getattr(proj_cfg, "ffn", "relu"),
            num_heads=getattr(proj_cfg, "num_heads", None),
            head_dim=getattr(proj_cfg, "head_dim", None),
            extra_merge_size=getattr(proj_cfg, "extra_merge_size", 1),
        )
        # processor_spatial_merge_size determines how many ``<image_pad>``
        # tokens the processor inserts per image; the vision encoder may use
        # a smaller merge size, in which case the projector's extra_merger
        # bridges the gap on the main stream.
        self._processor_spatial_merge_size = getattr(
            config, "processor_spatial_merge_size",
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
        (already compressed to image_pad length by the projector's extra
        merger if any), and ``deepstack_levels`` is either:
          - ``kind='mlp'``: a list of ``(total_image_pad_tokens, hidden_size)``
            tensors to scatter-add into the LLM hidden state, or
          - ``kind='cross_attn'``: a list of ``(k, v)`` tuples each of shape
            ``(total_kv_tokens, num_heads, head_dim)``.
        ``num_kv_per_item`` is the per-image vision-merged token count, used
        by the cross_attn path to build per-image attention masks.
        """
        vision_output = self.encoder(pixel_values, image_grid_thw)
        if hasattr(vision_output, "pooler_output"):
            vision_embeds = vision_output.pooler_output
        elif hasattr(vision_output, "last_hidden_state"):
            vision_embeds = vision_output.last_hidden_state
        elif isinstance(vision_output, (tuple, list)):
            vision_embeds = vision_output[0]
        else:
            vision_embeds = vision_output

        deepstack_features = getattr(vision_output, "deepstack_features", None)
        if deepstack_features is None:
            deepstack_features = []
        projected, projected_deepstack = self.proj(
            vision_embeds,
            list(deepstack_features),
        )
        projected = projected.reshape(-1, self.config.text_config.hidden_size)

        vision_merge = getattr(self.encoder.config, "spatial_merge_size", 2)
        processor_merge = self._processor_spatial_merge_size
        num_kv_per_item = image_grid_thw.prod(-1) // (vision_merge**2)
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
        projector: "_VisualStreamCrossAttnProjector",
    ) -> torch.FloatTensor:
        """Cross-attention deepstack injection (HF inference path).

        Q is gathered from ``<image_pad>`` positions in the LLM hidden state;
        K/V come from the projector's per-level deepstack output. The
        attention mask is block-diagonal so each query attends only to its
        own image's K/V.
        """
        flat_input = input_ids.view(-1)
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1]).clone()
        q_positions = (flat_input == self.config.image_token_id).nonzero(
            as_tuple=False
        ).reshape(-1)
        if q_positions.numel() == 0:
            return hidden_states
        if q_positions.shape[0] != int(num_tokens_per_item.sum().item()):
            raise ValueError(
                "image_pad token count does not match expected per-image counts: "
                f"got {q_positions.shape[0]}, expected "
                f"{int(num_tokens_per_item.sum().item())}"
            )

        # Build per-row image_id labels for Q and K/V.
        device = hidden_states.device
        q_image_id = torch.empty(
            q_positions.shape[0], dtype=torch.long, device=device
        )
        cursor = 0
        for i, count in enumerate(num_tokens_per_item.tolist()):
            q_image_id[cursor : cursor + count] = i
            cursor += count
        k_real, v_real = kv_pair
        kv_total = int(num_kv_per_item.sum().item())
        k_real = k_real[:kv_total]
        v_real = v_real[:kv_total]
        kv_image_id = torch.empty(kv_total, dtype=torch.long, device=device)
        cursor = 0
        for i, count in enumerate(num_kv_per_item.tolist()):
            kv_image_id[cursor : cursor + count] = i
            cursor += count

        # (Q, KV) bool mask: True where same image.
        attn_mask = q_image_id[:, None] == kv_image_id[None, :]

        q_real = flat_hidden.index_select(0, q_positions)
        delta = projector.attend(
            q_real.to(k_real.dtype), k_real, v_real, attn_mask
        )
        flat_hidden.index_add_(0, q_positions, delta.to(flat_hidden.dtype))
        return flat_hidden.view_as(hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
            raise ValueError("pixel_values and image_grid_thw must be provided together.")
        if pixel_values is not None and input_ids is None:
            raise ValueError("input_ids are required when pixel_values are provided.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        deepstack_features: list[Any] = []
        num_kv_per_item: Optional[torch.Tensor] = None
        num_tokens_per_item: Optional[torch.Tensor] = None
        if pixel_values is not None:
            image_features, deepstack_features, num_kv_per_item = (
                self._get_image_features(pixel_values, image_grid_thw)
            )
            num_tokens_per_item = image_grid_thw.prod(-1) // (
                self._processor_spatial_merge_size**2
            )
            inputs_embeds = self._inject_image_features(
                input_ids,
                inputs_embeds,
                image_features,
            )

        if use_cache and past_key_values is not None and not isinstance(
            past_key_values,
            Cache,
        ):
            from_legacy_cache = getattr(Cache, "from_legacy_cache", None)
            if callable(from_legacy_cache):
                past_key_values = from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        hidden_states = inputs_embeds
        v_first = torch.zeros_like(hidden_states)
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
            if layer_idx < len(deepstack_features):
                if self.proj.kind == "cross_attn":
                    hidden_states = self._cross_attn_image_features(
                        input_ids,
                        hidden_states,
                        deepstack_features[layer_idx],
                        num_kv_per_item=num_kv_per_item,
                        num_tokens_per_item=num_tokens_per_item,
                        projector=self.proj.deepstack[layer_idx],
                    )
                else:
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

    @staticmethod
    def _has_nonempty_past_key_values(
        past_key_values: Optional[Any],
        cache_position: Optional[torch.LongTensor],
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
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        logits_to_keep: Optional[int] = None,
        cache_position: Optional[torch.LongTensor] = None,
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
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
            hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:]
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
AutoModelForCausalLM.register(ModRWKVConfig, RWKV7VLForConditionalGeneration, exist_ok=True)
AutoModelForImageTextToText.register(
    ModRWKVConfig,
    RWKV7VLForConditionalGeneration,
    exist_ok=True,
)

ModRWKVConfig.register_for_auto_class("AutoConfig")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModel")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModelForCausalLM")
RWKV7VLForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
