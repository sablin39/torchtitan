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

    def to_dict(self) -> dict[str, Any]:
        return {
            "projector_type": self.projector_type,
            "encoder_dim": self.encoder_dim,
            "project_dim": self.project_dim,
            "hidden_dim": self.hidden_dim,
            "num_deepstack": self.num_deepstack,
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
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

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
        return output


class ModRWKVPreTrainedModel(PreTrainedModel):
    config_class = ModRWKVConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Block"]
    _supports_cache_class = True
    _skip_keys_device_placement = ["past_key_values"]


class _VisualStreamProjector(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4

        self.pre_norm = nn.LayerNorm(project_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, project_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x + self.pre_norm(x)


class VisualAdapter(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: Optional[int] = None,
        num_deepstack: int = 0,
        use_conv: bool = False,
    ):
        super().__init__()
        if use_conv:
            raise ValueError("Convolutional visual projectors are not supported.")
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4
        self.num_deepstack = num_deepstack
        self.main = _VisualStreamProjector(
            encoder_dim=encoder_dim,
            project_dim=project_dim,
            hidden_dim=self.hidden_dim,
        )
        self.deepstack = nn.ModuleList(
            [
                _VisualStreamProjector(
                    encoder_dim=encoder_dim,
                    project_dim=project_dim,
                    hidden_dim=self.hidden_dim,
                )
                for _ in range(num_deepstack)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        deepstack_features: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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
    ) -> tuple[torch.FloatTensor, list[torch.FloatTensor]]:
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
        projected_deepstack = [
            feature.reshape(-1, self.config.text_config.hidden_size)
            for feature in projected_deepstack
        ]

        spatial_merge_size = getattr(self.encoder.config, "spatial_merge_size", 2)
        expected_tokens = int(
            (image_grid_thw.prod(-1) // (spatial_merge_size**2)).sum().item()
        )
        if expected_tokens != projected.shape[0]:
            raise ValueError(
                "Projected image features and image grid do not match: "
                f"features={projected.shape[0]} grid_tokens={expected_tokens}"
            )
        if projected.numel() == 0:
            empty = torch.empty(
                0,
                self.config.text_config.hidden_size,
                device=self.get_input_embeddings().weight.device,
            )
            return empty, []
        return projected, projected_deepstack

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
        deepstack_features: list[torch.Tensor] = []
        if pixel_values is not None:
            image_features, deepstack_features = self._get_image_features(
                pixel_values,
                image_grid_thw,
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
