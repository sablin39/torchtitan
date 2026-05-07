# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_VOCAB_SIZE = 65536
DEFAULT_BOS_TOKEN = "\x16"
DEFAULT_EOS_TOKEN = "\x17"
DEFAULT_PAD_TOKEN = "\x17"
DEFAULT_UNK_TOKEN = "\x16"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VISION_START_TOKEN = "<|vision_start|>"
DEFAULT_VISION_END_TOKEN = "<|vision_end|>"
DEFAULT_IMAGE_PLACEHOLDER_TOKEN = "<image>"

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '\\x16' + ('Assistant' if message['role'] == 'assistant' else 'System' if message['role'] == 'system' else 'User') + ':' }}"
    "{{ message['content'] }}"
    "{{ '\\x17' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\\x16Assistant:' }}"
    "{% if thinking is defined and thinking %}{{ ' <think>' }}{% endif %}"
    "{% endif %}"
)

CHAT_TEMPLATE_FAKE_THINKING = (
    "{% for message in messages %}"
    "{{ '\\x16' + ('Assistant' if message['role'] == 'assistant' else 'System' if message['role'] == 'system' else 'User') + ':' }}"
    "{% if message['role'] == 'assistant' %}{{ ' <think>\\n</think>\\n' }}{% endif %}"
    "{{ message['content'] }}"
    "{{ '\\x17' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\\x16Assistant: <think>\\n</think>\\n' }}{% endif %}"
)

SPECIAL_TOKEN_TEXT_TO_ID = {
    DEFAULT_VISION_START_TOKEN: 65530,
    DEFAULT_VISION_END_TOKEN: 65531,
    DEFAULT_IMAGE_TOKEN: 65532,
}

SPECIAL_TOKEN_ID_TO_TEXT = {
    token_id: text for text, token_id in SPECIAL_TOKEN_TEXT_TO_ID.items()
}


@dataclass(frozen=True)
class RWKVSpecialTokens:
    bos_token: str = DEFAULT_BOS_TOKEN
    eos_token: str = DEFAULT_EOS_TOKEN
    pad_token: str = DEFAULT_PAD_TOKEN
    unk_token: str = DEFAULT_UNK_TOKEN
    image_token: str = DEFAULT_IMAGE_TOKEN
    vision_start_token: str = DEFAULT_VISION_START_TOKEN
    vision_end_token: str = DEFAULT_VISION_END_TOKEN
    image_placeholder_token: str = DEFAULT_IMAGE_PLACEHOLDER_TOKEN


class ByteTrie:
    __slots__ = ("children", "value")

    def __init__(self) -> None:
        self.children: dict[int, "ByteTrie"] = {}
        self.value: int | None = None

    def add(self, token: bytes, token_id: int) -> None:
        node = self
        for byte in token:
            node = node.children.setdefault(byte, ByteTrie())
        node.value = token_id

    def longest(self, data: bytes, start: int) -> tuple[int, int]:
        node = self
        best_id = None
        best_end = start
        idx = start
        while idx < len(data) and data[idx] in node.children:
            node = node.children[data[idx]]
            idx += 1
            if node.value is not None:
                best_id = node.value
                best_end = idx
        if best_id is None:
            raise ValueError(f"RWKV tokenizer could not encode byte at offset {start}")
        return best_end, best_id


def _as_bytes(token: str | bytes) -> bytes:
    return token.encode("utf-8") if isinstance(token, str) else token


def load_rwkv_vocab(vocab_file: str) -> tuple[dict[int, bytes], dict[bytes, int]]:
    idx2token: dict[int, bytes] = {}
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            first_space = line.index(" ")
            last_space = line.rindex(" ")
            token_id = int(line[:first_space])
            token = ast.literal_eval(line[first_space + 1 : last_space])
            if isinstance(token, str):
                token = token.encode("utf-8")
            if not isinstance(token, bytes):
                raise ValueError(f"Invalid RWKV vocab token on line: {line}")
            token_len = int(line[last_space + 1 :])
            if len(token) != token_len:
                raise ValueError(
                    f"Invalid RWKV vocab token length for id {token_id}: "
                    f"expected {token_len}, got {len(token)}"
                )
            idx2token[token_id] = token
    token2idx = {token: idx for idx, token in idx2token.items()}
    return idx2token, token2idx


def build_chat_template(template: str):
    import jinja2
    import jinja2.ext
    import jinja2.sandbox

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
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
    return env.from_string(template)


class RWKVTokenizerCore:
    def __init__(
        self,
        vocab_file: str,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        special_tokens: RWKVSpecialTokens | None = None,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        chat_template: str | None = None,
    ) -> None:
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or RWKVSpecialTokens()
        self.default_add_bos = add_bos_token
        self.default_add_eos = add_eos_token

        self.idx2token, self.token2idx = load_rwkv_vocab(vocab_file)
        self.root = ByteTrie()
        for token, token_id in self.token2idx.items():
            self.root.add(token, token_id)

        self.special_token_text_to_id = dict(SPECIAL_TOKEN_TEXT_TO_ID)
        self.special_token_id_to_text = dict(SPECIAL_TOKEN_ID_TO_TEXT)

        self.bos_id = self.token_to_id(self.special_tokens.bos_token)
        self.eos_id = self.token_to_id(self.special_tokens.eos_token)
        self.pad_id = self.token_to_id(self.special_tokens.pad_token)
        self.unk_id = self.token_to_id(self.special_tokens.unk_token)
        self.image_id = self.token_to_id(self.special_tokens.image_token)
        self.vision_start_id = self.token_to_id(self.special_tokens.vision_start_token)
        self.vision_end_id = self.token_to_id(self.special_tokens.vision_end_token)

        pattern_tokens = {
            *self.special_token_text_to_id,
            self.special_tokens.bos_token,
            self.special_tokens.eos_token,
            self.special_tokens.pad_token,
            self.special_tokens.unk_token,
            self.special_tokens.image_token,
            self.special_tokens.vision_start_token,
            self.special_tokens.vision_end_token,
        }
        pattern = "|".join(
            re.escape(token)
            for token in sorted(pattern_tokens, key=len, reverse=True)
            if token
        )
        self.special_token_pattern = re.compile(f"({pattern})") if pattern else None

        self._chat_template = None
        if chat_template is not None:
            self.set_chat_template(chat_template)

    @property
    def image_token(self) -> str:
        return self.special_tokens.image_token

    @property
    def vision_start_token(self) -> str:
        return self.special_tokens.vision_start_token

    @property
    def vision_end_token(self) -> str:
        return self.special_tokens.vision_end_token

    @property
    def image_placeholder_token(self) -> str:
        return self.special_tokens.image_placeholder_token

    @property
    def vision_image_token(self) -> str:
        return f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"

    def template_kwargs(self) -> dict[str, Any]:
        return {
            "bos_token": self.special_tokens.bos_token,
            "eos_token": self.special_tokens.eos_token,
            "pad_token": self.special_tokens.pad_token,
            "unk_token": self.special_tokens.unk_token,
            "image_token": self.image_token,
            "vision_start_token": self.vision_start_token,
            "vision_end_token": self.vision_end_token,
            "image_placeholder_token": self.image_placeholder_token,
        }

    def set_chat_template(self, template: str) -> None:
        self._chat_template = build_chat_template(template)

    def render_chat_template(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        if self._chat_template is None:
            raise ValueError("No chat template set. Call set_chat_template() first.")
        template_kwargs = self.template_kwargs()
        template_kwargs.update(kwargs)
        return self._chat_template.render(messages=messages, **template_kwargs)

    def _encode_bytes(self, data: bytes) -> list[int]:
        idx = 0
        tokens = []
        while idx < len(data):
            idx, token_id = self.root.longest(data, idx)
            tokens.append(token_id)
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool | None = None,
        add_eos: bool | None = None,
    ) -> list[int]:
        if add_bos is None:
            add_bos = self.default_add_bos
        if add_eos is None:
            add_eos = self.default_add_eos

        tokens: list[int] = []
        chunks = (
            self.special_token_pattern.split(text)
            if self.special_token_pattern is not None
            else [text]
        )
        for chunk in chunks:
            if not chunk:
                continue
            token_id = self.token_to_id(chunk)
            if token_id is not None and token_id != self.unk_id:
                tokens.append(token_id)
            else:
                tokens.extend(self._encode_bytes(chunk.encode("utf-8")))
        if add_bos and self.bos_id is not None:
            tokens.insert(0, self.bos_id)
        if add_eos and self.eos_id is not None:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return b"".join(self.idx2token[int(i)] for i in token_ids).decode(
            "utf-8",
            errors="replace",
        )

    def token_to_id(self, token: str | bytes | int) -> int | None:
        if isinstance(token, int):
            return token
        if isinstance(token, str) and token in self.special_token_text_to_id:
            return self.special_token_text_to_id[token]
        return self.token2idx.get(_as_bytes(token))

    def id_to_token(self, token_id: int) -> str | None:
        token = self.idx2token.get(int(token_id))
        if token is None:
            return None
        return token.decode("utf-8", errors="replace")

    def get_vocab(self) -> dict[str, int]:
        return {
            token.decode("utf-8", errors="replace"): idx
            for idx, token in self.idx2token.items()
        }

    def save_vocabulary(self, vocab_file: str) -> None:
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token_index, token in sorted(self.idx2token.items()):
                writer.write(f"{token_index} {repr(token)} {len(token)}\n")

    def expand_image_placeholders(
        self,
        rendered_text: str,
        image_token_counts: list[int],
    ) -> str:
        patterns = [self.vision_image_token, self.image_placeholder_token]
        deduped_patterns = []
        for pattern in patterns:
            if pattern and pattern not in deduped_patterns:
                deduped_patterns.append(pattern)

        pieces: list[str] = []
        pos = 0
        image_idx = 0
        while pos < len(rendered_text):
            matches = []
            for pattern in deduped_patterns:
                start = rendered_text.find(pattern, pos)
                if start != -1:
                    matches.append((start, -len(pattern), pattern))
            if not matches:
                pieces.append(rendered_text[pos:])
                break

            start, _, pattern = min(matches)
            pieces.append(rendered_text[pos:start])
            if image_idx >= len(image_token_counts):
                raise ValueError(
                    "Rendered chat contains more image placeholders than images: "
                    f"saw at least {image_idx + 1}, got {len(image_token_counts)}"
                )
            n_tokens = image_token_counts[image_idx]
            pieces.append(
                f"{self.vision_start_token}"
                f"{self.image_token * n_tokens}"
                f"{self.vision_end_token}"
            )
            image_idx += 1
            pos = start + len(pattern)

        if image_idx != len(image_token_counts):
            raise ValueError(
                "Rendered chat contains fewer image placeholders than images: "
                f"saw {image_idx}, got {len(image_token_counts)}"
            )
        return "".join(pieces)

    def render_mm_chat(
        self,
        messages: list[dict[str, Any]],
        image_token_counts_by_message: list[list[int]],
        *,
        add_generation_prompt: bool = False,
    ) -> str:
        if len(messages) != len(image_token_counts_by_message):
            raise ValueError(
                "image_token_counts_by_message must have one entry per message: "
                f"got {len(image_token_counts_by_message)} counts for "
                f"{len(messages)} messages"
            )
        rendered = self.render_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
        ).rstrip("\n")
        image_token_counts = [
            count
            for message_counts in image_token_counts_by_message
            for count in message_counts
        ]
        return self.expand_image_placeholders(rendered, image_token_counts)

    def assistant_token_spans(
        self,
        messages: list[dict[str, Any]],
        image_token_counts_by_message: list[list[int]],
        *,
        add_bos: bool = True,
    ) -> list[tuple[int, int]]:
        spans = []
        for idx, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            start_text = self.render_mm_chat(
                messages[:idx],
                image_token_counts_by_message[:idx],
                add_generation_prompt=True,
            )
            end_text = self.render_mm_chat(
                messages[: idx + 1],
                image_token_counts_by_message[: idx + 1],
                add_generation_prompt=False,
            )
            start = len(self.encode(start_text, add_bos=add_bos, add_eos=False))
            end = len(self.encode(end_text, add_bos=add_bos, add_eos=False))
            if start < end:
                spans.append((start, end))
        return spans
