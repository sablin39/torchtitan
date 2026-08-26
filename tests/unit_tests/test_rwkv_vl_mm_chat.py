# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from scripts.rwkv7_exporter.export_hf_model import save_remote_code_assets

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.hf_datasets.multimodal.mm_chat_datasets import (
    build_image_token_counts_by_message,
    MMChatCollator,
    MMChatDataLoader,
    MMChatDataset,
    normalize_mm_chat_sample,
)
from torchtitan.hf_datasets.multimodal.processor import (
    process_images as process_rwkv_vl_images,
    RWKVVLImageProcessorConfig,
)
from torchtitan.models.rwkv7.tokenizer import CHAT_TEMPLATE
from transformers import BaseImageProcessor


DATASET_KWARGS = {
    "seq_len": 512,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "min_pixels": 1024,
    "max_pixels": 4096,
    "image_mean": (0.5, 0.5, 0.5),
    "image_std": (0.5, 0.5, 0.5),
    "max_aspect_ratio": 50.0,
}


def _write_tiny_rwkv_vocab(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for byte in range(256):
            token = bytes([byte])
            token_id = byte + 1
            f.write(f"{token_id} {repr(token)} {len(token)}\n")
        for token_id, token in (
            (65530, b"<|vision_start|>"),
            (65531, b"<|vision_end|>"),
            (65532, b"<|image_pad|>"),
            (65533, b"<tool_call>"),
            (65534, b"<tool_response>"),
            (65535, b"<tools>"),
        ):
            f.write(f"{token_id} {repr(token)} {len(token)}\n")


def _make_tokenizer(tmpdir: str) -> HuggingFaceTokenizer:
    _write_tiny_rwkv_vocab(os.path.join(tmpdir, "wr_vocab_v20230424.txt"))
    save_remote_code_assets(tmpdir)
    with open(os.path.join(tmpdir, "chat_template.jinja"), "w") as f:
        f.write(CHAT_TEMPLATE)
    with open(os.path.join(tmpdir, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "auto_map": {
                    "AutoTokenizer": ["tokenizer.RwkvTokenizer", None],
                },
                "tokenizer_class": "RwkvTokenizer",
                "bos_token": "<|endoftext|>",
                "eos_token": "✿",
                "pad_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
                "add_bos_token": False,
                "add_eos_token": False,
                "model_max_length": 8192,
            },
            f,
        )
    return HuggingFaceTokenizer(
        config=HuggingFaceTokenizer.Config(
            trust_remote_code=True,
            chat_template_add_bos=False,
            chat_template_append_eos=False,
        ),
        tokenizer_path=tmpdir,
    )


def _two_image_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe first."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "First answer."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe second."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Second answer."}],
        },
    ]


def _two_image_sample() -> dict:
    return {
        "messages": _two_image_messages(),
        "images": [
            Image.new("RGB", (32, 32), color="red"),
            Image.new("RGB", (64, 32), color="blue"),
        ],
    }


def _make_mm_chat_dataset(
    tokenizer: HuggingFaceTokenizer,
    samples: list[dict],
    **overrides,
) -> MMChatDataset:
    kwargs = dict(DATASET_KWARGS)
    kwargs.update(overrides)
    return MMChatDataset(
        Dataset.from_list(samples),
        tokenizer=tokenizer,
        **kwargs,
    )


def _load_exported_remote_code(tmpdir: str):
    export_dir = Path(tmpdir) / "remote"
    export_dir.mkdir()
    (export_dir / "__init__.py").write_text("", encoding="utf-8")
    save_remote_code_assets(str(export_dir), include_processor=True)
    package_name = export_dir.name
    module_names = (
        package_name,
        f"{package_name}.tokenizer",
        f"{package_name}.processor",
    )
    sys.path.insert(0, str(export_dir.parent))
    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        tokenizer_module = importlib.import_module(f"{package_name}.tokenizer")
        processor_module = importlib.import_module(f"{package_name}.processor")
        return tokenizer_module.RwkvTokenizer, processor_module.ModRWKVProcessor
    finally:
        sys.path.remove(str(export_dir.parent))


def _contains_tensor(value) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


class TinyImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self):
        super().__init__()
        self.patch_size = 16
        self.temporal_patch_size = 2
        self.merge_size = 2
        self.size = {"shortest_edge": 1024, "longest_edge": 4096}
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)


class TestRwkvVLTokenizer(unittest.TestCase):
    def test_exporter_vocab_contains_tool_and_vision_special_tags(self):
        vocab_path = (
            Path(__file__).parents[2]
            / "scripts"
            / "rwkv7_exporter"
            / "wr_vocab_v20230424.txt"
        )
        expected = {
            65530: "<|vision_start|>",
            65531: "<|vision_end|>",
            65532: "<|image_pad|>",
            65533: "<tool_call>",
            65534: "<tool_response>",
            65535: "<tools>",
        }
        found = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                token_id = int(line[: line.index(" ")])
                if token_id not in expected:
                    continue
                token = eval(line[line.index(" ") : line.rindex(" ")])
                found[token_id] = token
        self.assertEqual(found, expected)

    def test_exporter_has_no_static_core_copies(self):
        repo_root = Path(__file__).parents[2]
        exporter_dir = repo_root / "scripts" / "rwkv7_exporter"
        self.assertFalse((exporter_dir / "processor_core.py").exists())
        self.assertFalse((exporter_dir / "tokenizer_core.py").exists())
        self.assertFalse((exporter_dir / "processor.py").exists())
        self.assertFalse((exporter_dir / "tokenizer.py").exists())

    def test_chat_template_renders_user_bot_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            messages = [{"role": "user", "content": "problem"}]

            prompt = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            self.assertEqual(
                prompt,
                "User✿problem✿\nBot✿",
            )

            generation_prompt = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            self.assertTrue(generation_prompt.endswith("Bot✿"))
            self.assertFalse(generation_prompt.endswith("Bot✿ "))

            full = tok.apply_chat_template(
                [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "answer"},
                ],
                add_generation_prompt=False,
            )
            self.assertIn("Bot✿answer✿", full)
            self.assertTrue(full.startswith(generation_prompt))

    def test_thinking_content_renders_without_extra_spacing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            messages = [
                {"role": "user", "content": "problem"},
                {
                    "role": "assistant",
                    "content": "<think>\nreason\n</think>\n answer",
                },
            ]
            rendered = tok.render_mm_chat(messages, [[], []])
            self.assertIn("Bot✿<think>\nreason\n</think>\n answer✿", rendered)
            self.assertNotIn("Bot✿ <think>", rendered)

    def test_qwen_tools_tool_calls_and_tool_responses_render(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search docs.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Find it"},
                {
                    "role": "assistant",
                    "content": "I will check.",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": {"query": "rwkv"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "result one"},
                {"role": "tool", "content": "result two"},
                {"role": "assistant", "content": "Done."},
            ]
            rendered = tok.render_mm_chat(messages, [[] for _ in messages], tools=tools)
            self.assertEqual(rendered.count("# Tools"), 1)
            self.assertIn("<tools>\n", rendered)
            self.assertIn('"name": "search"', rendered)
            self.assertIn("<tool_call>\n", rendered)
            rendered_ids = tok.encode(rendered)
            self.assertIn(65533, rendered_ids)
            self.assertIn(65534, rendered_ids)
            self.assertIn(65535, rendered_ids)
            self.assertIn(
                '{"name": "search", "arguments": {"query": "rwkv"}}', rendered
            )
            self.assertIn(
                "User✿<tool_response>\nresult one\n</tool_response>\n"
                "<tool_response>\nresult two\n</tool_response>✿",
                rendered,
            )

            supervised = "".join(
                tok.decode(tok.encode(rendered, add_bos=True)[start:end])
                for start, end in tok.assistant_token_spans(
                    messages,
                    [[] for _ in messages],
                    tools=tools,
                )
            )
            self.assertIn("<tool_call>", supervised)
            self.assertIn('"query": "rwkv"', supervised)
            self.assertIn("Done.", supervised)
            self.assertNotIn("<tool_response>", supervised)
            self.assertNotIn("result one", supervised)

    def test_tool_tags_are_atomic_but_closing_tags_remain_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            text = (
                "<tools>x</tools>"
                "<tool_call>\n{}\n</tool_call>"
                "<tool_response>ok</tool_response>"
            )
            ids = tok.encode(text)
            self.assertEqual(ids.count(65533), 1)
            self.assertEqual(ids.count(65534), 1)
            self.assertEqual(ids.count(65535), 1)
            self.assertEqual(tok.decode(ids), text)
            self.assertNotEqual(tok.token_to_id("</tool_call>"), 65533)
            self.assertIsNone(tok.token_to_id("</tool_call>"))

    def test_vision_special_tags_keep_exact_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            text = f"{tok.vision_start_token}{tok.image_token}{tok.vision_end_token}"
            ids = tok.encode(text)
            self.assertEqual(ids, [65530, 65532, 65531])
            self.assertEqual(tok.decode(ids), text)

    def test_existing_qwen_tool_preamble_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            tools = [
                {
                    "type": "function",
                    "function": {"name": "search", "description": "", "parameters": {}},
                }
            ]
            messages = [
                {
                    "role": "system",
                    "content": "# Tools\n<tools>old</tools>\n<tool_call>x</tool_call>",
                },
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ]
            rendered = tok.render_mm_chat(messages, [[], [], []], tools=tools)
            self.assertEqual(rendered.count("# Tools"), 1)
            self.assertNotIn("You may call one or more functions", rendered)

    def test_expand_image_placeholders_adds_missing_and_drops_extra(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            text = "\x16User:<image><image>hello\x17"
            expanded = tok.expand_image_placeholders(text, [2])
            self.assertEqual(expanded.count(tok.vision_start_token), 1)
            self.assertEqual(expanded.count(tok.vision_end_token), 1)
            self.assertEqual(expanded.count(tok.image_token), 2)
            self.assertNotIn("<image>", expanded)

            expanded = tok.expand_image_placeholders("\x16User:hello\x17", [1, 3])
            self.assertEqual(expanded.count(tok.vision_start_token), 2)
            self.assertEqual(expanded.count(tok.vision_end_token), 2)
            self.assertEqual(expanded.count(tok.image_token), 4)

    def test_image_count_builder_caps_extra_tags_and_prepends_missing(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Question"},
                ],
            },
            {"role": "assistant", "content": "Answer"},
        ]
        counts = build_image_token_counts_by_message(
            messages,
            [5],
            image_placeholder_token="<image>",
        )
        self.assertEqual(counts, [[5], []])

        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": [{"type": "text", "text": "Question"}]},
            {"role": "assistant", "content": "Answer"},
        ]
        counts = build_image_token_counts_by_message(
            messages,
            [2, 4],
            image_placeholder_token="<image>",
        )
        self.assertEqual(counts, [[], [2, 4], []])

    def test_hf_exporter_processor_uses_shared_pixel_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            HFRwkvTokenizer, ModRWKVProcessor = _load_exported_remote_code(tmpdir)
            vocab_file = os.path.join(tmpdir, "wr_vocab_v20230424.txt")
            _write_tiny_rwkv_vocab(vocab_file)
            hf_tok = HFRwkvTokenizer(
                vocab_file=vocab_file,
                chat_template=CHAT_TEMPLATE,
            )
            processor = ModRWKVProcessor(
                tokenizer=hf_tok,
                image_processor=TinyImageProcessor(),
            )
            output = processor(
                images=[
                    Image.new("RGB", (128, 128), color="red"),
                    Image.new("RGB", (128, 128), color="blue"),
                ],
                text="\x16User:<image><image>\x17",
            )
            image_grid_thw = output["image_grid_thw"]
            image_pixels = int(
                torch.sum(image_grid_thw[:, 1] * 16 * image_grid_thw[:, 2] * 16)
            )
            raw_patches = int(image_grid_thw.prod(-1).sum().item())
            self.assertLessEqual(image_pixels, 4096)
            self.assertEqual(image_grid_thw.shape[0], 2)
            self.assertEqual(output["pixel_values"].shape[0], raw_patches)
            self.assertLessEqual(
                output["input_ids"][0].count(hf_tok.image_token_id),
                4096 // (16 * 2) ** 2,
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.image_token_id),
                2,
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.vision_start_token_id),
                2,
            )
            self.assertEqual(
                output["input_ids"][0].count(hf_tok.vision_end_token_id),
                2,
            )

    def test_hf_exporter_copies_remote_code_dependencies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "exported")
            save_remote_code_assets(output_dir, include_processor=True)
            self.assertEqual(
                set(os.listdir(output_dir)),
                {
                    "tokenizer.py",
                    "processor.py",
                },
            )

    def test_max_pixels_reduces_actual_patch_and_token_counts_for_odd_sizes(self):
        images = [
            Image.new("RGB", (513, 377), color="red"),
            Image.new("RGB", (641, 319), color="blue"),
        ]
        common = {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "min_pixels": 1024,
            "image_mean": (0.5, 0.5, 0.5),
            "image_std": (0.5, 0.5, 0.5),
        }
        large = process_rwkv_vl_images(
            images,
            RWKVVLImageProcessorConfig(max_pixels=262144, **common),
        )
        small = process_rwkv_vl_images(
            images,
            RWKVVLImageProcessorConfig(max_pixels=8192, **common),
        )

        large_raw_patches = int(large.grid_thw.prod(-1).sum().item())
        small_raw_patches = int(small.grid_thw.prod(-1).sum().item())
        large_llm_token_cap = 262144 // (16 * 2) ** 2
        small_llm_token_cap = 8192 // (16 * 2) ** 2
        self.assertEqual(large.grid_thw.tolist(), [[1, 18, 24], [1, 16, 32]])
        self.assertEqual(small.grid_thw.tolist(), [[1, 2, 4], [1, 2, 6]])
        self.assertEqual(large.flat_patches.shape[0], large_raw_patches)
        self.assertEqual(small.flat_patches.shape[0], small_raw_patches)
        self.assertEqual(large.image_token_counts, [108, 128])
        self.assertEqual(small.image_token_counts, [2, 3])
        self.assertLessEqual(sum(large.image_token_counts), large_llm_token_cap)
        self.assertLessEqual(sum(small.image_token_counts), small_llm_token_cap)
        self.assertLess(small_raw_patches, large_raw_patches)
        self.assertLess(sum(small.image_token_counts), sum(large.image_token_counts))

    def test_rwkv_vl_tokenizer_exposes_image_only_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            self.assertEqual(
                tok.TOKEN_FIELDS, ("image", "vision_start", "vision_end", "pad")
            )
            self.assertFalse(hasattr(tok, "video_id"))
            self.assertEqual(tok.image_id, 65532)
            self.assertEqual(tok.vision_start_id, 65530)
            self.assertEqual(tok.vision_end_id, 65531)
            self.assertEqual(tok.bos_id, 0)
            self.assertEqual(tok.eos_id, 10060)
            self.assertEqual(tok.pad_id, 0)
            self.assertEqual(tok.image_placeholder_token, "<image>")

    def test_render_mm_chat_expands_image_placeholders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                },
                {"role": "assistant", "content": "Done."},
            ]
            rendered = tok.render_mm_chat(
                messages,
                image_token_counts_by_message=[[3], []],
                add_generation_prompt=False,
            )
            self.assertNotIn("<image>", rendered)
            self.assertEqual(rendered.count(tok.vision_start_token), 1)
            self.assertEqual(rendered.count(tok.vision_end_token), 1)
            self.assertEqual(rendered.count(tok.image_token), 3)
            self.assertEqual(tok.encode(rendered).count(tok.image_id), 3)

    def test_render_mm_chat_expands_multiple_images_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            rendered = tok.render_mm_chat(
                _two_image_messages(),
                image_token_counts_by_message=[[1], [], [2], []],
                add_generation_prompt=False,
            )
            ids = tok.encode(rendered)
            self.assertEqual(ids.count(tok.image_id), 3)
            self.assertEqual(ids.count(tok.vision_start_id), 2)
            self.assertEqual(ids.count(tok.vision_end_id), 2)

            first_start = rendered.find(tok.vision_start_token)
            first_end = rendered.find(tok.vision_end_token, first_start) + len(
                tok.vision_end_token
            )
            second_start = rendered.find(tok.vision_start_token, first_end)
            second_end = rendered.find(tok.vision_end_token, second_start) + len(
                tok.vision_end_token
            )
            self.assertEqual(rendered[first_start:first_end].count(tok.image_token), 1)
            self.assertEqual(
                rendered[second_start:second_end].count(tok.image_token), 2
            )

    def test_assistant_token_spans_cover_only_assistant_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            messages = [
                {"role": "user", "content": "Question one"},
                {"role": "assistant", "content": "Answer one"},
                {"role": "user", "content": "Question two"},
                {"role": "assistant", "content": "Answer two"},
            ]
            counts = [[], [], [], []]
            rendered = tok.render_mm_chat(messages, counts, add_generation_prompt=False)
            full_tokens = tok.encode(rendered, add_bos=False, add_eos=False)
            supervised = "".join(
                tok.decode(full_tokens[start:end])
                for start, end in tok.assistant_token_spans(messages, counts)
            )
            self.assertIn("Answer one", supervised)
            self.assertIn("Answer two", supervised)
            self.assertNotIn("Question one", supervised)
            self.assertNotIn("Question two", supervised)


class TestMMChatDataset(unittest.TestCase):
    def test_normalize_mm_chat_sample_accepts_common_schemas(self):
        image = Image.new("RGB", (32, 32), color="red")
        cases = [
            {
                "conversations": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": "Answer"},
                ],
                "images": [image, None],
            },
            {
                "messages": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": "Answer"},
                ],
                "images": [image],
            },
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "search",
                            "arguments": '{"query": "rwkv"}',
                        },
                    },
                ],
                "images": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "tool_call", "content": '{"name": "search"}'},
                ],
                "images": [],
            },
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "image": image,
            },
            {
                "texts": [
                    {"user": "Question", "assistant": "Answer"},
                ],
                "images": [image],
            },
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ],
                "images": [],
                "available_tools": [],
            },
        ]
        for sample in cases:
            normalized = normalize_mm_chat_sample(sample)
            self.assertEqual(normalized["messages"][0]["role"], "user")
            self.assertEqual(normalized["messages"][1]["role"], "assistant")
            self.assertEqual(normalized["tools"], [])

    def test_normalize_mm_chat_sample_preserves_normalized_tools_and_tool_calls(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-search",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": {"query": "rwkv"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "search",
                    "tool_call_id": "call-search",
                    "content": "result",
                },
                {"role": "assistant", "content": "Answer"},
            ],
            "images": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search.",
                        "parameters": {},
                    },
                }
            ],
        }
        normalized = normalize_mm_chat_sample(sample)
        self.assertEqual(normalized["images"], [])
        self.assertEqual(normalized["tools"][0]["function"]["name"], "search")
        self.assertEqual(normalized["messages"][1]["role"], "assistant")
        self.assertEqual(
            normalized["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"query": "rwkv"},
        )
        self.assertEqual(
            normalized["messages"][1]["tool_calls"][0]["id"],
            "call-search",
        )
        self.assertEqual(normalized["messages"][2]["role"], "tool")
        self.assertEqual(normalized["messages"][2]["tool_call_id"], "call-search")

    def test_normalize_mm_chat_sample_preserves_plain_llava_answers(self):
        sample = {
            "conversations": [
                {"from": "human", "value": "Question"},
                {"from": "gpt", "value": "Answer."},
            ],
            "images": [],
        }
        normalized = normalize_mm_chat_sample(sample)
        self.assertEqual(
            normalized["messages"][1]["content"],
            "Answer.",
        )

    def test_normalize_mm_chat_sample_uses_reasoning_for_training(self):
        sample = {
            "messages": [
                {"from": "system", "value": "Policy."},
                {"from": "human", "value": "Question"},
                {
                    "from": "gpt",
                    "reasoning_content": "Need to inspect the policy.",
                    "value": "Answer.",
                },
                {"from": "human", "value": "Short?"},
                {
                    "from": "gpt",
                    "reasoning_content": "The answer is short.",
                    "value": "Yes.",
                },
            ],
            "images": [],
            "tools": [],
        }
        normalized = normalize_mm_chat_sample(sample)
        self.assertEqual(
            normalized["messages"][2]["content"],
            "<think>\nNeed to inspect the policy.\n</think>\n Answer.",
        )
        self.assertEqual(
            normalized["messages"][4]["content"],
            "<think>\nThe answer is short.\n</think>\n Yes.",
        )
        self.assertNotIn("<think>\n</think>", normalized["messages"][2]["content"])
        self.assertNotIn("<think>\n</think>", normalized["messages"][4]["content"])
        self.assertNotIn("reasoning_content", normalized["messages"][2])

    def test_mm_chat_dataset_loads_converted_sharegpt_tool_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = {
                "messages": [
                    {"from": "system", "value": "Policy."},
                    {"from": "human", "value": "Verify me"},
                    {
                        "from": "gpt",
                        "reasoning_content": "Need a verification lookup.",
                        "value": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "verify",
                                    "arguments": {"user_id": "u1"},
                                },
                            }
                        ],
                    },
                    {
                        "from": "tool",
                        "tool_call_id": "call-1",
                        "value": '{"ok": true}',
                    },
                    {
                        "from": "gpt",
                        "reasoning_content": "Tool confirmed the user.",
                        "value": "Verified.",
                    },
                ],
                "images": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "verify",
                            "description": "Verify a user.",
                            "parameters": {},
                        },
                    }
                ],
            }
            processed = next(iter(_make_mm_chat_dataset(tok, [sample], seq_len=2048)))
            self.assertIsNone(processed["pixel_values"])
            self.assertIsNone(processed["grid_thw"])
            input_text = tok.decode(processed["input_ids"].tolist())
            supervised = tok.decode(
                processed["labels"][processed["labels"] != IGNORE_INDEX].tolist()
            )
            self.assertIn("<tools>", input_text)
            self.assertIn(
                "<think>\nNeed a verification lookup.\n</think>\n ",
                supervised,
            )
            self.assertIn("<tool_call>", supervised)
            self.assertIn('"user_id": "u1"', supervised)
            self.assertIn(
                '<tool_response>\n{"ok": true}\n</tool_response>',
                input_text,
            )
            input_ids = processed["input_ids"].tolist()
            label_ids = processed["labels"][
                processed["labels"] != IGNORE_INDEX
            ].tolist()
            self.assertIn(65533, label_ids)
            self.assertIn(65534, input_ids)
            self.assertIn(65535, input_ids)
            self.assertNotIn(65534, label_ids)
            self.assertNotIn("<tool_response>", supervised)
            self.assertIn(
                "<think>\nTool confirmed the user.\n</think>\n Verified.",
                supervised,
            )

    def test_mm_chat_dataloader_streams_converted_text_parquet_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            image_path = os.path.join(tmpdir, "image.jsonl")
            text_path = os.path.join(tmpdir, "nemotron.parquet")
            image_row = {
                "messages": [
                    {"role": "user", "content": "image-free"},
                    {"role": "assistant", "content": "ok"},
                ],
                "images": [],
                "tools": [],
            }
            with open(image_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(image_row) + "\n")
            row = {
                "uuid": "row-1",
                "messages": [
                    {"role": "user", "content": "Verify"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "verify",
                                    "arguments": {"required": [None]},
                                    "required": [None],
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "ok"},
                    {"role": "assistant", "content": "done"},
                ],
                "license": "test",
                "used_in": ["unit"],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "verify",
                            "description": "Verify.",
                            "parameters": "{}",
                            "required": [None],
                        },
                    }
                ],
            }
            Dataset.from_list([row]).to_parquet(text_path)
            config = MMChatDataLoader.Config(
                dataset_path="json",
                data_files=image_path,
                text_dataset_path=text_path,
                text_sample_probability=1.0,
                infinite=False,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=1024,
                max_pixels=4096,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
            )
            loader = MMChatDataLoader(
                config,
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tok,
                seq_len=2048,
                local_batch_size=1,
            )
            input_dict, labels = next(iter(loader))
            self.assertIn(65535, input_dict["input"][0].tolist())
            supervised = tok.decode(labels[labels != IGNORE_INDEX].tolist())
            self.assertIn("<tool_call>", supervised)

    def test_mm_chat_dataloader_loads_converted_text_parquet_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            image_path = os.path.join(tmpdir, "image.jsonl")
            text_dir = os.path.join(tmpdir, "text")
            os.makedirs(os.path.join(text_dir, "nested"))
            image_row = {
                "messages": [
                    {"role": "user", "content": "image side"},
                    {"role": "assistant", "content": "ok"},
                ],
                "images": [],
            }
            with open(image_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(image_row) + "\n")
            text_row = {
                "uuid": "text-1",
                "source": "Nemotron-SFT-Science-v2/rqa",
                "source_file": "/tmp/rqa.jsonl",
                "source_line": 1,
                "messages": [
                    {"from": "human", "value": "Use the tool"},
                    {
                        "from": "gpt",
                        "value": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "evaluate_code",
                                    "arguments": {"source_code": "1 + 1"},
                                },
                            }
                        ],
                    },
                    {"from": "tool", "value": "2"},
                    {"from": "gpt", "value": "Two."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "evaluate_code",
                            "description": "Evaluate code.",
                            "parameters": {
                                "type": "object",
                                "properties": {"source_code": {"type": "string"}},
                                "required": ["source_code"],
                            },
                        },
                    }
                ],
                "license": "",
                "used_in": [],
                "metadata": {"dataset": "unit"},
            }
            Dataset.from_list([text_row]).to_parquet(
                os.path.join(text_dir, "nested", "part-00000.parquet")
            )
            config = MMChatDataLoader.Config(
                dataset_path="json",
                data_files=image_path,
                text_dataset_path=text_dir,
                text_sample_probability=1.0,
                infinite=False,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=1024,
                max_pixels=4096,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
            )
            loader = MMChatDataLoader(
                config,
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tok,
                seq_len=2048,
                local_batch_size=1,
            )
            input_dict, labels = next(iter(loader))
            input_text = tok.decode(input_dict["input"][0].tolist())
            supervised = tok.decode(labels[labels != IGNORE_INDEX].tolist())
            self.assertIn("<tools>", input_text)
            self.assertIn("<tool_response>\n2\n</tool_response>", input_text)
            self.assertIn("<tool_call>", supervised)

    def test_mm_chat_dataloader_loads_primary_converted_parquet_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            dataset_dir = os.path.join(tmpdir, "converted")
            os.makedirs(os.path.join(dataset_dir, "shard"))
            row = {
                "uuid": "text-1",
                "source": "Nemotron-SFT-Agentic-v2/tool_calling",
                "source_file": "/tmp/tool_calling.jsonl",
                "source_line": 1,
                "messages": [
                    {"from": "human", "value": "Use the tool"},
                    {
                        "from": "gpt",
                        "value": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": {"query": "rwkv"},
                                },
                            }
                        ],
                    },
                    {"from": "tool", "value": "found"},
                    {"from": "gpt", "value": "Done."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Lookup.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                    }
                ],
                "license": "",
                "used_in": [],
                "metadata": {"dataset": "unit"},
            }
            Dataset.from_list([row]).to_parquet(
                os.path.join(dataset_dir, "shard", "part-00000.parquet")
            )
            config = MMChatDataLoader.Config(
                dataset_path=dataset_dir,
                infinite=False,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=1024,
                max_pixels=4096,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
            )
            loader = MMChatDataLoader(
                config,
                dp_world_size=1,
                dp_rank=0,
                tokenizer=tok,
                seq_len=2048,
                local_batch_size=1,
            )
            input_dict, labels = next(iter(loader))
            input_text = tok.decode(input_dict["input"][0].tolist())
            supervised = tok.decode(labels[labels != IGNORE_INDEX].tolist())
            self.assertIn("<tools>", input_text)
            self.assertIn("<tool_response>\nfound\n</tool_response>", input_text)
            self.assertIn("<tool_call>", supervised)

    def test_mm_chat_dataset_blends_llava_and_converted_text_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            llava_sample = {
                "conversations": [
                    {"from": "human", "value": "Describe this."},
                    {"from": "gpt", "value": "A red square."},
                ],
                "images": [Image.new("RGB", (32, 32), color="red")],
            }
            text_sample = {
                "messages": [
                    {"from": "human", "value": "Verify me"},
                    {
                        "from": "gpt",
                        "value": (
                            "<think>\nNeed to verify before answering.\n</think>\n "
                            "Verified."
                        ),
                    },
                ],
                "images": [],
                "tools": [],
            }
            dataset = _make_mm_chat_dataset(
                tok,
                [llava_sample],
                text_dataset=Dataset.from_list([text_sample]),
                text_sample_probability=0.5,
                seq_len=2048,
            )

            samples = list(dataset)
            self.assertEqual(len(samples), 2)
            supervised = "\n".join(
                tok.decode(sample["labels"][sample["labels"] != IGNORE_INDEX].tolist())
                for sample in samples
            )
            self.assertIn("A red square.", supervised)
            self.assertNotIn("<think>\n</think>\n A red square.", supervised)
            self.assertIn(
                "<think>\nNeed to verify before answering.\n</think>\n Verified.",
                supervised,
            )
            self.assertNotIn("<think>\n</think>", supervised)

    def test_mm_chat_dataset_blend_uses_project_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            image_samples = [
                {"messages": [], "images": [], "source": "image0"},
                {"messages": [], "images": [], "source": "image1"},
            ]
            text_samples = [
                {"messages": [], "images": [], "source": "text0"},
                {"messages": [], "images": [], "source": "text1"},
            ]

            def source_order(seed: int) -> list[str]:
                dataset = _make_mm_chat_dataset(
                    tok,
                    image_samples,
                    text_dataset=Dataset.from_list(text_samples),
                    text_sample_probability=0.5,
                    seed=seed,
                    seq_len=2048,
                )
                return [sample["source"] for sample, _, _ in dataset._iter_samples()]

            self.assertEqual(source_order(1234), source_order(1234))
            self.assertNotEqual(source_order(1234), source_order(1235))

    def test_mm_chat_dataset_counts_image_tokens(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = next(iter(_make_mm_chat_dataset(tok, [_two_image_sample()])))
            input_ids = sample["input_ids"]
            self.assertEqual((input_ids == tok.image_id).sum().item(), 3)
            self.assertEqual((input_ids == tok.vision_start_id).sum().item(), 2)
            self.assertEqual((input_ids == tok.vision_end_id).sum().item(), 2)
            self.assertEqual(sample["grid_thw"].shape[0], 2)
            self.assertEqual(
                sample["pixel_values"].shape[0],
                int(sample["grid_thw"].prod(-1).sum().item()),
            )

    def test_mm_chat_dataset_can_store_bfloat16_pixel_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = next(
                iter(
                    _make_mm_chat_dataset(
                        tok,
                        [_two_image_sample()],
                        pixel_values_dtype="bfloat16",
                    )
                )
            )
            self.assertEqual(sample["pixel_values"].dtype, torch.bfloat16)

    def test_mm_chat_image_processing_uses_shared_pixel_budget(self):
        processed = process_rwkv_vl_images(
            [
                Image.new("RGB", (128, 128), color="red"),
                Image.new("RGB", (128, 128), color="blue"),
            ],
            RWKVVLImageProcessorConfig(
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                min_pixels=1024,
                max_pixels=4096,
                image_mean=(0.5, 0.5, 0.5),
                image_std=(0.5, 0.5, 0.5),
                max_aspect_ratio=50.0,
            ),
        )
        self.assertEqual(processed.grid_thw.shape[0], 2)
        self.assertLessEqual(
            sum(image.shape[1] * image.shape[2] for image in processed.images),
            4096,
        )
        self.assertEqual(processed.image_token_counts, [1, 1])
        self.assertEqual(
            processed.flat_patches.shape[0],
            int(processed.grid_thw.prod(-1).sum().item()),
        )

    def test_mm_chat_dataset_masks_only_assistant_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = next(iter(_make_mm_chat_dataset(tok, [_two_image_sample()])))
            labels = sample["labels"]
            supervised = tok.decode(labels[labels != IGNORE_INDEX].tolist())
            self.assertIn("First answer.", supervised)
            self.assertIn("Second answer.", supervised)
            self.assertNotIn("Describe first.", supervised)
            self.assertNotIn("Describe second.", supervised)
            self.assertNotIn(tok.image_token, supervised)
            self.assertNotIn(tok.vision_start_token, supervised)
            self.assertNotIn(tok.vision_end_token, supervised)

    def test_mm_chat_dataset_accepts_text_only_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                    },
                ],
                "images": [],
                "tools": [],
            }
            processed = next(iter(_make_mm_chat_dataset(tok, [sample])))
            self.assertIsNone(processed["pixel_values"])
            self.assertIsNone(processed["grid_thw"])
            supervised = tok.decode(
                processed["labels"][processed["labels"] != IGNORE_INDEX].tolist()
            )
            self.assertIn("Answer", supervised)
            self.assertNotIn("<think>\n</think>", supervised)

            collator = MMChatCollator(
                batch_size=1,
                seq_len=512,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                tokenizer=tok,
            )
            input_dict, labels = collator([processed])
            self.assertIsNone(input_dict["pixel_values"])
            self.assertIsNone(input_dict["grid_thw"])
            self.assertGreater(input_dict["input_token_mask"].sum().item(), 0)
            self.assertGreater((labels != IGNORE_INDEX).sum().item(), 0)

    def test_mm_chat_dataset_masks_tool_responses_but_supervises_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": {"q": "x"},
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": "tool result"},
                    {"role": "assistant", "content": "final"},
                ],
                "images": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search.",
                            "parameters": {},
                        },
                    }
                ],
            }
            processed = next(iter(_make_mm_chat_dataset(tok, [sample], seq_len=2048)))
            input_text = tok.decode(processed["input_ids"].tolist())
            supervised = tok.decode(
                processed["labels"][processed["labels"] != IGNORE_INDEX].tolist()
            )
            self.assertIn("<tools>", input_text)
            self.assertIn("<tool_response>\ntool result\n</tool_response>", input_text)
            self.assertIn("<tool_call>", supervised)
            self.assertIn('"q": "x"', supervised)
            self.assertIn("final", supervised)
            self.assertNotIn("tool result", supervised)

    def test_mm_chat_collator_handles_mixed_text_and_image_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            text_sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Question"}],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": "Answer",
                            }
                        ],
                    },
                ],
                "images": [],
                "tools": [],
            }
            samples = list(
                _make_mm_chat_dataset(
                    tok,
                    [text_sample, _two_image_sample()],
                    seq_len=1024,
                )
            )
            self.assertEqual(len(samples), 2)
            self.assertIsNone(samples[0]["pixel_values"])
            self.assertIsInstance(samples[1]["pixel_values"], torch.Tensor)

            collator = MMChatCollator(
                batch_size=2,
                seq_len=1024,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                tokenizer=tok,
            )
            input_dict, labels = collator(samples)
            self.assertIsNotNone(input_dict["pixel_values"])
            self.assertIsNotNone(input_dict["grid_thw"])
            self.assertEqual(input_dict["grid_thw"].shape[0], 2)
            self.assertGreater(input_dict["input_token_mask"][0].sum().item(), 0)
            self.assertGreater(input_dict["input_token_mask"][1].sum().item(), 0)
            self.assertGreater((labels[0] != IGNORE_INDEX).sum().item(), 0)
            self.assertGreater((labels[1] != IGNORE_INDEX).sum().item(), 0)

    def test_mm_chat_dataset_masks_system_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = _two_image_sample()
            sample["messages"] = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "System prompt."}],
                },
                *sample["messages"],
            ]
            processed = next(iter(_make_mm_chat_dataset(tok, [sample])))
            input_text = tok.decode(processed["input_ids"].tolist())
            supervised = tok.decode(
                processed["labels"][processed["labels"] != IGNORE_INDEX].tolist()
            )
            self.assertIn("System prompt.", input_text)
            self.assertNotIn("System prompt.", supervised)

    def test_mm_chat_dataset_drops_overlength_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            dataset = _make_mm_chat_dataset(tok, [_two_image_sample()], seq_len=4)
            self.assertEqual(list(dataset), [])

    def test_mm_chat_dataset_packed_positions_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            dataset = _make_mm_chat_dataset(
                tok,
                [_two_image_sample(), _two_image_sample()],
                packing_buffer_size=2,
                batch_size=2,
            )
            packed = next(iter(dataset))
            reset_points = (packed["positions"][1:] == 0).nonzero(as_tuple=True)[0]
            self.assertGreater(len(reset_points), 0)
            self.assertIsInstance(packed["pixel_values"], list)
            self.assertIsInstance(packed["grid_thw"], list)

    def test_mm_chat_dataset_packing_buffer_one_yields_promptly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            dataset = _make_mm_chat_dataset(
                tok,
                [_two_image_sample(), _two_image_sample(), _two_image_sample()],
                packing_buffer_size=1,
                batch_size=2,
            )
            iterator = iter(dataset)
            first = next(iterator)
            second = next(iterator)
            self.assertGreater(first["input_ids"].numel(), 0)
            self.assertGreater(second["input_ids"].numel(), 0)
            self.assertLessEqual(len(dataset.packer._sample_buffer), 1)

    def test_mm_chat_dataset_checkpoint_drops_packer_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            dataset = _make_mm_chat_dataset(
                tok,
                [_two_image_sample()],
                packing_buffer_size=64,
                batch_size=2,
            )

            processed = dataset._tokenize_sample(_two_image_sample())
            self.assertIsNotNone(processed)
            self.assertIsInstance(processed["pixel_values"], torch.Tensor)
            dataset.packer.add_sample(processed)
            self.assertEqual(len(dataset.packer._sample_buffer), 1)

            state = dataset.state_dict()
            self.assertNotIn("packer_state", state)
            self.assertFalse(_contains_tensor(state))

            dataset.load_state_dict(state)
            self.assertEqual(dataset.packer._sample_buffer, {})
            self.assertEqual(dataset.packer._next_id, 0)
            self.assertEqual(len(dataset.packer.packed_samples), 0)

    def test_mm_chat_collator_does_not_shift_again(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            sample = next(iter(_make_mm_chat_dataset(tok, [_two_image_sample()])))
            collator = MMChatCollator(
                batch_size=1,
                seq_len=512,
                max_images_per_batch=8,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                tokenizer=tok,
            )
            input_dict, labels = collator([sample])
            n = sample["input_ids"].numel()
            self.assertTrue(
                torch.equal(input_dict["input"][0, :n], sample["input_ids"])
            )
            self.assertTrue(torch.equal(labels[0, :n], sample["labels"]))
            self.assertIn("pixel_values", input_dict)
            self.assertIn("grid_thw", input_dict)
            self.assertIn("special_tokens", input_dict)
            self.assertIn("input_token_mask", input_dict)
            self.assertEqual(input_dict["input_token_mask"].sum().item(), n)
            self.assertTrue(input_dict["input_token_mask"][0, :n].all().item())
            self.assertFalse(input_dict["input_token_mask"][0, n:].any().item())
            self.assertEqual(input_dict["grid_thw"].shape[0], 2)
            self.assertEqual(input_dict["pixel_values"].dim(), 2)
            self.assertEqual(
                input_dict["pixel_values"].shape[0],
                int(input_dict["grid_thw"].prod(-1).sum().item()),
            )
            self.assertIn("data_stats", input_dict)
            self.assertEqual(input_dict["data_stats"]["num_images"], 2)
            self.assertEqual(input_dict["data_stats"]["packed_rows"], 1)
            self.assertEqual(input_dict["data_stats"]["packed_docs"], 1)
            self.assertEqual(input_dict["data_stats"]["nonpad_tokens"], n)

    def test_mm_chat_collator_rejects_vit_patch_bucketing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            with self.assertRaisesRegex(ValueError, "Qwen3.5 vision encoder"):
                MMChatCollator(
                    batch_size=1,
                    seq_len=512,
                    max_images_per_batch=8,
                    patch_size=16,
                    temporal_patch_size=2,
                    spatial_merge_size=2,
                    tokenizer=tok,
                    vit_patch_bucket_size=64,
                )

    def test_mm_chat_collator_zero_image_cap_keeps_all_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            samples = list(
                _make_mm_chat_dataset(tok, [_two_image_sample(), _two_image_sample()])
            )
            collator = MMChatCollator(
                batch_size=2,
                seq_len=512,
                max_images_per_batch=0,
                patch_size=16,
                temporal_patch_size=2,
                spatial_merge_size=2,
                tokenizer=tok,
            )
            input_dict, labels = collator(samples)
            self.assertEqual(input_dict["grid_thw"].shape[0], 4)
            self.assertEqual(input_dict["pixel_values"].dim(), 2)
            self.assertEqual(
                input_dict["pixel_values"].shape[0],
                int(input_dict["grid_thw"].prod(-1).sum().item()),
            )
            self.assertGreater(input_dict["input_token_mask"][1].sum().item(), 0)
            self.assertGreater((labels[1] != IGNORE_INDEX).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
