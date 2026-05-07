# -*- coding: utf-8 -*-
"""RWKV-VL HF remote-code model used by the exporter.

This file intentionally lives beside the exporter so the generated checkpoint
is self-contained.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3_5VisionModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "projector_type": self.projector_type,
            "encoder_dim": self.encoder_dim,
            "project_dim": self.project_dim,
            "hidden_dim": self.hidden_dim,
        }


class ModRWKVConfig(PretrainedConfig):
    model_type = "modrwkv"
    is_composition = True

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: Union[RWKV7Config, Dict[str, Any]],
        vision_config: Union[Qwen3_5VisionConfig, Dict[str, Any]],
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
        vision_config: Optional[Union[Qwen3_5VisionConfig, Dict[str, Any]]] = None,
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
        if isinstance(vision_config, dict):
            vision_config = Qwen3_5VisionConfig(**vision_config)
        self.vision_config = vision_config

        if projector_config is None:
            projector_config = ModRWKVProjectorConfig(
                encoder_dim=getattr(vision_config, "out_hidden_size", 1024),
                project_dim=getattr(text_config, "hidden_size", 1024),
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


class VisualAdapter(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        project_dim: int,
        hidden_dim: Optional[int] = None,
        use_conv: bool = False,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.project_dim = project_dim
        self.hidden_dim = hidden_dim or project_dim * 4
        self.use_conv = use_conv

        self.pre_norm = nn.LayerNorm(encoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, project_dim),
        )
        if use_conv:
            self.conv = nn.Conv1d(
                in_channels=encoder_dim,
                out_channels=encoder_dim,
                kernel_size=3,
                stride=2,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + self.mlp(self.pre_norm(x))


class RWKV7VLModel(ModRWKVPreTrainedModel):
    def __init__(self, config: ModRWKVConfig):
        super().__init__(config)
        self.encoder = Qwen3_5VisionModel(config.vision_config)

        proj_cfg = config.projector_config
        self.proj = VisualAdapter(
            encoder_dim=proj_cfg.encoder_dim,
            project_dim=proj_cfg.project_dim,
            hidden_dim=proj_cfg.hidden_dim,
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
    ) -> torch.FloatTensor:
        vision_output = self.encoder(pixel_values, image_grid_thw)
        if hasattr(vision_output, "last_hidden_state"):
            vision_embeds = vision_output.last_hidden_state
        elif isinstance(vision_output, (tuple, list)):
            vision_embeds = vision_output[0]
        else:
            vision_embeds = vision_output

        spatial_merge_size = getattr(self.encoder.config, "spatial_merge_size", 2)
        split_sizes = (image_grid_thw.prod(-1) // (spatial_merge_size**2)).tolist()
        image_embeds = torch.split(vision_embeds, split_sizes)
        projected = []
        for embeds in image_embeds:
            projected.append(self.proj(embeds).reshape(-1, self.config.text_config.hidden_size))
        if not projected:
            return torch.empty(
                0,
                self.config.text_config.hidden_size,
                device=self.get_input_embeddings().weight.device,
            )
        return torch.cat(projected, dim=0)

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
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds.")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = self._get_image_features(pixel_values, image_grid_thw)
            inputs_embeds = self._inject_image_features(
                input_ids,
                inputs_embeds,
                image_features,
            )

        return self.llm(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class RWKV7VLForConditionalGeneration(ModRWKVPreTrainedModel):
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
