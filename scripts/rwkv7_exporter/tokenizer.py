# coding=utf-8
"""HF remote-code tokenizer wrapper for RWKV/RWKV-VL exports."""

import os
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

try:
    from .tokenizer_core import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_VISION_END_TOKEN,
        DEFAULT_VISION_START_TOKEN,
        RWKVSpecialTokens,
        RWKVTokenizerCore,
    )
except ImportError:
    try:
        from tokenizer_core import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_VISION_END_TOKEN,
            DEFAULT_VISION_START_TOKEN,
            RWKVSpecialTokens,
            RWKVTokenizerCore,
        )
    except ImportError:
        from torchtitan.models.rwkv7.tokenizer_core import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_VISION_END_TOKEN,
            DEFAULT_VISION_START_TOKEN,
            RWKVSpecialTokens,
            RWKVTokenizerCore,
        )


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "wr_vocab_v20230424.txt",
}

CHAT_TEMPLATE = """{% for message in messages -%}
<|im_start|>{{ message['role'][:1] | upper }}{{ message['role'][1:] }}: {{ message['content'] }}
<|im_end|>

{% endfor -%}
{% if add_generation_prompt -%}
<|im_start|>Assistant: {% if thinking is defined and thinking %}<think>{% else %}<think></think>{% endif %}
{% endif -%}"""

DEFAULT_ADDITIONAL_SPECIAL_TOKENS = [
    "<tool_calls_begin>",
    "</tool_calls_end>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    DEFAULT_VISION_START_TOKEN,
    DEFAULT_VISION_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
]


def _token_content(token):
    return getattr(token, "content", token)


class RwkvTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
        unk_token="<|im_start|>",
        chat_template=None,
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        bos_token = _token_content(bos_token)
        eos_token = _token_content(eos_token)
        pad_token = _token_content(pad_token)
        unk_token = _token_content(unk_token)

        self.add_bos_token = bool(kwargs.pop("add_bos_token", False))
        self.core = RWKVTokenizerCore(
            vocab_file,
            special_tokens=RWKVSpecialTokens(
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                unk_token=unk_token,
            ),
            add_bos_token=self.add_bos_token,
            add_eos_token=False,
            chat_template=CHAT_TEMPLATE if chat_template is None else chat_template,
        )
        self.encoder = self.core.token2idx
        self.decoder = self.core.idx2token
        self.chat_template = CHAT_TEMPLATE if chat_template is None else chat_template
        self.special_token_text_to_id = dict(self.core.special_token_text_to_id)

        self._added_tokens_encoder = {}
        self._added_tokens_decoder = {}
        for tok_text, tok_id in self.special_token_text_to_id.items():
            self._added_tokens_encoder[tok_text] = tok_id
            self._added_tokens_decoder[tok_id] = AddedToken(tok_text, special=True)
        for tok in {bos_token, eos_token, pad_token, unk_token}:
            if tok is None:
                continue
            tok_id = self.core.token_to_id(str(tok))
            if tok_id is not None:
                self._added_tokens_encoder[str(tok)] = tok_id
                self._added_tokens_decoder[tok_id] = AddedToken(str(tok), special=True)

        additional_special_tokens = kwargs.pop(
            "additional_special_tokens",
            DEFAULT_ADDITIONAL_SPECIAL_TOKENS,
        )
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            chat_template=self.chat_template,
            **kwargs,
        )

        self.image_token = self.core.image_token
        self.vision_start_token = self.core.vision_start_token
        self.vision_end_token = self.core.vision_end_token
        self.image_token_id = self.convert_tokens_to_ids(self.image_token)
        self.vision_start_token_id = self.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = self.convert_tokens_to_ids(self.vision_end_token)
        self.image_id = self.image_token_id
        self.vision_start_id = self.vision_start_token_id
        self.vision_end_id = self.vision_end_token_id
        self.image_placeholder_token = self.core.image_placeholder_token
        self.vision_image_token = self.core.vision_image_token

    @property
    def vocab_size(self):
        return self.core.vocab_size

    def get_vocab(self):
        vocab = self.core.get_vocab()
        vocab.update(self.added_tokens_encoder)
        return dict(sorted(vocab.items(), key=lambda item: item[1]))

    def _tokenize(self, text, split_special_tokens=False):
        del split_special_tokens
        return self.core.encode(text, add_bos=False, add_eos=False)

    def _convert_token_to_id(self, token):
        token_id = self.core.token_to_id(token)
        return token_id if token_id is not None else self.unk_token_id

    def _convert_id_to_token(self, index):
        token = self.core.id_to_token(int(index))
        return token if token is not None else self.unk_token

    def convert_tokens_to_string(self, tokens):
        return "".join(
            token.decode("utf-8", errors="replace")
            if isinstance(token, bytes)
            else str(token)
            for token in tokens
        )

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        self.core.save_vocabulary(vocab_file)
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_ids = [self.bos_token_id] if self.add_bos_token else []
        output = bos_token_ids + token_ids_0
        if token_ids_1 is None:
            return output
        return output + bos_token_ids + token_ids_1

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=False,
            )
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def expand_image_placeholders(
        self,
        rendered_text: str,
        image_token_counts: list[int],
    ) -> str:
        return self.core.expand_image_placeholders(rendered_text, image_token_counts)

    def render_mm_chat(
        self,
        messages: list[dict],
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
        messages: list[dict],
        image_token_counts_by_message: list[list[int]],
        *,
        add_bos: bool = True,
    ) -> list[tuple[int, int]]:
        return self.core.assistant_token_spans(
            messages,
            image_token_counts_by_message,
            add_bos=add_bos,
        )
