# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass
from typing import Any

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.rwkv7.tokenizer_core import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_UNK_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    DEFAULT_VISION_START_TOKEN,
    DEFAULT_VOCAB_SIZE,
    RWKVSpecialTokens,
    RWKVTokenizerCore,
)


class RwkvTokenizer(BaseTokenizer):
    """TorchTitan wrapper around the shared RWKV tokenizer core."""

    CHAT_TEMPLATE_FILE = "chat_template.jinja"

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_file: str = "wr_vocab_v20230424.txt"
        vocab_size: int = DEFAULT_VOCAB_SIZE
        bos_token: str = DEFAULT_BOS_TOKEN
        eos_token: str = DEFAULT_EOS_TOKEN
        pad_token: str = DEFAULT_PAD_TOKEN
        unk_token: str = DEFAULT_UNK_TOKEN
        image_token: str = DEFAULT_IMAGE_TOKEN
        vision_start_token: str = DEFAULT_VISION_START_TOKEN
        vision_end_token: str = DEFAULT_VISION_END_TOKEN
        add_bos_token: bool = False
        add_eos_token: bool = False

    def __init__(
        self,
        config: Config | None = None,
        *,
        tokenizer_path: str,
    ):
        super().__init__()
        self.config = config or RwkvTokenizer.Config()
        self.tokenizer_path = tokenizer_path
        self.vocab_file = os.path.join(tokenizer_path, self.config.vocab_file)
        if not os.path.exists(self.vocab_file):
            raise FileNotFoundError(f"RWKV vocab file not found: {self.vocab_file}")

        add_bos = self.config.add_bos_token
        add_eos = self.config.add_eos_token
        hf_config = None
        config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                hf_config = json.load(f)
            add_bos = bool(hf_config.get("add_bos_token", add_bos))
            add_eos = bool(hf_config.get("add_eos_token", add_eos))

        special_tokens = RWKVSpecialTokens(
            bos_token=self.config.bos_token,
            eos_token=self.config.eos_token,
            pad_token=self.config.pad_token,
            unk_token=self.config.unk_token,
            image_token=self.config.image_token,
            vision_start_token=self.config.vision_start_token,
            vision_end_token=self.config.vision_end_token,
            image_placeholder_token=getattr(
                self.config,
                "image_placeholder_token",
                "<image>",
            ),
        )
        self.core = RWKVTokenizerCore(
            self.vocab_file,
            vocab_size=self.config.vocab_size,
            special_tokens=special_tokens,
            add_bos_token=add_bos,
            add_eos_token=add_eos,
        )

        jinja_path = os.path.join(tokenizer_path, self.CHAT_TEMPLATE_FILE)
        if os.path.exists(jinja_path):
            with open(jinja_path) as f:
                self.set_chat_template(f.read())
        elif hf_config is not None and "chat_template" in hf_config:
            self.set_chat_template(hf_config["chat_template"])

        self.idx2token = self.core.idx2token
        self.token2idx = self.core.token2idx
        self.bos_token = self.config.bos_token
        self.eos_token = self.config.eos_token
        self.pad_token = self.config.pad_token
        self.unk_token = self.config.unk_token
        self.image_token = self.core.image_token
        self.vision_start_token = self.core.vision_start_token
        self.vision_end_token = self.core.vision_end_token
        self.bos_id = self.core.bos_id
        self.eos_id = self.core.eos_id
        self.pad_id = self.core.pad_id
        self.unk_id = self.core.unk_id
        self.image_id = self.core.image_id
        self.vision_start_id = self.core.vision_start_id
        self.vision_end_id = self.core.vision_end_id
        self.image_token_id = self.image_id
        self.vision_start_token_id = self.vision_start_id
        self.vision_end_token_id = self.vision_end_id
        self.default_add_bos = self.core.default_add_bos
        self.default_add_eos = self.core.default_add_eos

    def set_chat_template(self, template: str) -> None:
        self.core.set_chat_template(template)

    def _template_kwargs(self) -> dict[str, Any]:
        return self.core.template_kwargs()

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs) -> str:
        return self.core.render_chat_template(messages, **kwargs)

    def encode(self, *args, **kwargs) -> list[int]:
        text = args[0] if args else kwargs.get("text", "")
        return self.core.encode(
            text,
            add_bos=kwargs.get("add_bos", None),
            add_eos=kwargs.get("add_eos", None),
        )

    def decode(self, *args, **kwargs) -> str:
        token_ids = args[0] if args else kwargs.get("token_ids", [])
        return self.core.decode(token_ids)

    def get_vocab_size(self) -> int:
        return self.core.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def get_vocab(self) -> dict[str, int]:
        return self.core.get_vocab()

    def token_to_id(self, token: str | bytes) -> int | None:
        return self.core.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.core.id_to_token(token_id)
