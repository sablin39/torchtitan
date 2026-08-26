# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from torchtitan.config import Configurable

from transformers import AutoTokenizer


class BaseTokenizer(ABC, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    eos_id: int | None

    def __init__(self):
        self.eos_id = None
        self._chat_template = None
        self.chat_template_add_bos = True
        self.chat_template_append_eos = True

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]:
        ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        ...

    def set_chat_template(self, template: str) -> None:
        """Compile and store a Jinja chat template."""
        import json

        import jinja2
        import jinja2.ext
        import jinja2.sandbox

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(
            x, ensure_ascii=False, indent=None, separators=None, sort_keys=False
        ):
            return json.dumps(
                x,
                ensure_ascii=ensure_ascii,
                indent=indent,
                separators=separators,
                sort_keys=sort_keys,
            )

        def strftime_now(fmt):
            from datetime import datetime

            return datetime.now().strftime(fmt)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=[jinja2.ext.loopcontrols],
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = strftime_now
        env.filters["tojson"] = tojson
        self._chat_template = env.from_string(template)

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Render messages through the Jinja chat template."""
        if self._chat_template is None:
            raise ValueError("No chat template set. Call set_chat_template() first.")
        return self._chat_template.render(messages=messages, **kwargs)


class HuggingFaceTokenizer(BaseTokenizer):
    """Thin TorchTitan adapter for a Hugging Face Transformers tokenizer."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        trust_remote_code: bool = False
        chat_template_add_bos: bool | None = None
        chat_template_append_eos: bool | None = None
        image_token: str | None = None
        video_token: str | None = None
        vision_start_token: str | None = None
        vision_end_token: str | None = None
        pad_token: str | None = None

    def __init__(
        self,
        config: Config | None = None,
        *,
        tokenizer_path: str,
    ):
        super().__init__()
        self.config = config or HuggingFaceTokenizer.Config()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.default_add_bos = bool(getattr(self.tokenizer, "add_bos_token", False))
        self.default_add_eos = bool(getattr(self.tokenizer, "add_eos_token", False))
        self.chat_template_add_bos = (
            self.default_add_bos
            if self.config.chat_template_add_bos is None
            else self.config.chat_template_add_bos
        )
        self.chat_template_append_eos = (
            self.default_add_eos
            if self.config.chat_template_append_eos is None
            else self.config.chat_template_append_eos
        )

        for name in (
            "bos",
            "eos",
            "pad",
            "unk",
            "image",
            "video",
            "vision_start",
            "vision_end",
        ):
            self._expose_special_token(name)

        self.TOKEN_FIELDS = tuple(
            name
            for name in ("image", "video", "vision_start", "vision_end", "pad")
            if getattr(self, f"{name}_id", None) is not None
        )

    def __getattr__(self, name: str) -> Any:
        tokenizer = self.__dict__.get("tokenizer")
        if tokenizer is None:
            raise AttributeError(name)
        return getattr(tokenizer, name)

    def _expose_special_token(self, name: str) -> None:
        configured_token = getattr(self.config, f"{name}_token", None)
        if configured_token is not None:
            token = configured_token
            token_id = self.tokenizer.get_vocab().get(token)
            if token_id is None:
                raise ValueError(
                    f"Special token {token!r} configured as {name}_token was not "
                    f"found in the tokenizer at {self.tokenizer_path!r}."
                )
        else:
            token = getattr(self.tokenizer, f"{name}_token", None)
            token_id = getattr(self.tokenizer, f"{name}_token_id", None)
            if token_id is None and token is not None:
                token_id = self.tokenizer.get_vocab().get(str(token))
            if token is None and token_id is not None:
                token = self.tokenizer.convert_ids_to_tokens(token_id)

        if token is not None:
            setattr(self, f"{name}_token", token)
        if token_id is not None:
            setattr(self, f"{name}_id", int(token_id))
            setattr(self, f"{name}_token_id", int(token_id))

    def set_chat_template(self, template: str) -> None:
        self.tokenizer.chat_template = template

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        kwargs["tokenize"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode(self, *args, **kwargs) -> list[int]:
        text = args[0] if args else kwargs.pop("text", "")
        add_bos = kwargs.pop("add_bos", None)
        add_eos = kwargs.pop("add_eos", None)
        if add_bos is None and add_eos is None:
            return self.tokenizer.encode(text, **kwargs)

        add_bos = self.default_add_bos if add_bos is None else add_bos
        add_eos = self.default_add_eos if add_eos is None else add_eos
        kwargs["add_special_tokens"] = False
        token_ids = self.tokenizer.encode(text, **kwargs)
        if add_bos and self.bos_id is not None:
            token_ids.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, *args, **kwargs) -> str:
        token_ids = args[0] if args else kwargs.pop("token_ids", [])
        return self.tokenizer.decode(token_ids, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def get_vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def token_to_id(self, token: str | bytes) -> int | None:
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return self.get_vocab().get(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self.tokenizer.convert_ids_to_tokens(token_id)


class MultiModalTokenizer(HuggingFaceTokenizer):
    """Single source of truth for multimodal special tokens.

    Requires 5 token strings via config, validates them against the vocabulary
    at init, and exposes both string and ID attributes (e.g. ``image_token``,
    ``image_id``). The Qwen multimodal Grain processor requires
    ``MultiModalTokenizer`` (not ``HuggingFaceTokenizer``) and reads these
    attributes directly; the collator packs the IDs into a plain
    ``dict[str, int]`` that travels through the batch to the model forward.
    Adding a new VLM means filling in 5 config strings — no subclassing needed.

    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        image_token: str
        """Token string for image placeholders, e.g. ``"<|image_pad|>"``."""

        video_token: str
        """Token string for video placeholders, e.g. ``"<|video_pad|>"``."""

        vision_start_token: str
        """Token string marking the start of a vision sequence."""

        vision_end_token: str
        """Token string marking the end of a vision sequence."""

        pad_token: str
        """Token string for padding."""

    # Config field prefixes that follow the {name}_token pattern.
    TOKEN_FIELDS = ("image", "video", "vision_start", "vision_end", "pad")

    def __init__(self, config: Config, *, tokenizer_path: str):
        super().__init__(config, tokenizer_path=tokenizer_path)
        for name in self.TOKEN_FIELDS:
            if getattr(self, f"{name}_id", None) is None:
                raise ValueError(
                    f"Special token configured as {name}_token was not found "
                    f"in the tokenizer at {tokenizer_path!r}."
                )
