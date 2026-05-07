# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

from torchtitan.models.rwkv7.tokenizer import RwkvTokenizer


class RwkvVLMultiModalTokenizer(RwkvTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(RwkvTokenizer.Config):
        image_token: str = "<|image_pad|>"
        vision_start_token: str = "<|vision_start|>"
        vision_end_token: str = "<|vision_end|>"
        pad_token: str = "\x17"
        image_placeholder_token: str = "<image>"

    TOKEN_FIELDS = ("image", "vision_start", "vision_end", "pad")

    def __init__(self, config: Config | None = None, *, tokenizer_path: str):
        super().__init__(config, tokenizer_path=tokenizer_path)
        cfg = config or RwkvVLMultiModalTokenizer.Config()
        for name in self.TOKEN_FIELDS:
            token_str = getattr(cfg, f"{name}_token")
            token_id = self.token_to_id(token_str)
            if token_id is None:
                raise ValueError(
                    f"RWKV-VL multimodal token '{token_str}' not found in vocab"
                )
            setattr(self, f"{name}_token", token_str)
            setattr(self, f"{name}_id", token_id)
        self.image_placeholder_token = cfg.image_placeholder_token
        self.image_token_id = self.image_id
        self.vision_start_token_id = self.vision_start_id
        self.vision_end_token_id = self.vision_end_id
        self.vision_image_token = (
            f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
        )

    def _template_kwargs(self) -> dict[str, Any]:
        kwargs = super()._template_kwargs()
        kwargs["image_placeholder_token"] = self.image_placeholder_token
        return kwargs

    def expand_image_placeholders(
        self,
        rendered_text: str,
        image_token_counts: list[int],
    ) -> str:
        return self.core.expand_image_placeholders(rendered_text, image_token_counts)

    def render_mm_chat(
        self,
        messages: list[dict[str, Any]],
        image_token_counts_by_message: list[list[int]],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        return self.core.render_mm_chat(
            messages,
            image_token_counts_by_message,
            add_generation_prompt=add_generation_prompt,
        )

    def assistant_token_spans(
        self,
        messages: list[dict[str, Any]],
        image_token_counts_by_message: list[list[int]],
        *,
        add_bos: bool = True,
    ) -> list[tuple[int, int]]:
        return self.core.assistant_token_spans(
            messages,
            image_token_counts_by_message,
            add_bos=add_bos,
        )
