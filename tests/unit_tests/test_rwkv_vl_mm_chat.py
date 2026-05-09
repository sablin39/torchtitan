# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import tempfile
import unittest

import torch
from datasets import Dataset
from PIL import Image
from transformers import BaseImageProcessor

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.processor_core import (
    RWKVVLImageProcessorConfig,
    process_images as process_rwkv_vl_images,
)
from torchtitan.hf_datasets.multimodal.mm_chat_datasets import (
    build_image_token_counts_by_message,
    MMChatCollator,
    MMChatDataset,
    normalize_mm_chat_sample,
    process_mm_chat_images,
)
from torchtitan.models.rwkv_vl.tokenizer import RwkvVLMultiModalTokenizer
from scripts.rwkv7_exporter.export_hf_model import save_processor_core
from scripts.rwkv7_exporter.processor import ModRWKVProcessor
from scripts.rwkv7_exporter.tokenizer import RwkvTokenizer as HFRwkvTokenizer


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '\x16' + ('Assistant' if message['role'] == 'assistant' else 'System' if message['role'] == 'system' else 'User') + ':' }}"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<image>' }}{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "{{ '\x17' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\x16Assistant:' }}{% endif %}"
)


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
        ):
            f.write(f"{token_id} {repr(token)} {len(token)}\n")


def _make_tokenizer(tmpdir: str) -> RwkvVLMultiModalTokenizer:
    _write_tiny_rwkv_vocab(os.path.join(tmpdir, "wr_vocab_v20230424.txt"))
    with open(os.path.join(tmpdir, "chat_template.jinja"), "w") as f:
        f.write(CHAT_TEMPLATE)
    return RwkvVLMultiModalTokenizer(tokenizer_path=tmpdir)


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
    tokenizer: RwkvVLMultiModalTokenizer,
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
    def test_exporter_has_no_static_processor_core_copy(self):
        repo_root = Path(__file__).parents[2]
        exporter_core = repo_root / "scripts" / "rwkv7_exporter" / "processor_core.py"
        self.assertFalse(exporter_core.exists())

    def test_torchtitan_and_hf_exporter_tokenizers_align(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_file = os.path.join(tmpdir, "wr_vocab_v20230424.txt")
            _write_tiny_rwkv_vocab(vocab_file)
            with open(os.path.join(tmpdir, "chat_template.jinja"), "w") as f:
                f.write(CHAT_TEMPLATE)

            tt_tok = RwkvVLMultiModalTokenizer(tokenizer_path=tmpdir)
            hf_tok = HFRwkvTokenizer(
                vocab_file=vocab_file,
                bos_token="\x16",
                eos_token="\x17",
                pad_token="\x17",
                unk_token="\x16",
                chat_template=CHAT_TEMPLATE,
            )
            messages = _two_image_messages()
            counts = [[1], [], [2], []]
            tt_rendered = tt_tok.render_mm_chat(messages, counts)
            hf_rendered = hf_tok.render_mm_chat(messages, counts)
            self.assertEqual(tt_rendered, hf_rendered)
            self.assertEqual(
                tt_tok.encode(tt_rendered, add_bos=True, add_eos=False),
                hf_tok.core.encode(hf_rendered, add_bos=True, add_eos=False),
            )
            self.assertEqual(
                tt_tok.assistant_token_spans(messages, counts),
                hf_tok.assistant_token_spans(messages, counts),
            )

    def test_expand_image_placeholders_adds_missing_and_drops_extra(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = _make_tokenizer(tmpdir)
            text = "\x16User:<image><image>hello\x17"
            expanded = tok.expand_image_placeholders(text, [2])
            self.assertEqual(expanded.count(tok.core.vision_start_token), 1)
            self.assertEqual(expanded.count(tok.core.vision_end_token), 1)
            self.assertEqual(expanded.count(tok.core.image_token), 2)
            self.assertNotIn("<image>", expanded)

            expanded = tok.expand_image_placeholders("\x16User:hello\x17", [1, 3])
            self.assertEqual(expanded.count(tok.core.vision_start_token), 2)
            self.assertEqual(expanded.count(tok.core.vision_end_token), 2)
            self.assertEqual(expanded.count(tok.core.image_token), 4)

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
            vocab_file = os.path.join(tmpdir, "wr_vocab_v20230424.txt")
            _write_tiny_rwkv_vocab(vocab_file)
            hf_tok = HFRwkvTokenizer(
                vocab_file=vocab_file,
                bos_token="\x16",
                eos_token="\x17",
                pad_token="\x17",
                unk_token="\x16",
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

    def test_hf_exporter_fetches_processor_core_during_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_file = os.path.join(tmpdir, "wr_vocab_v20230424.txt")
            _write_tiny_rwkv_vocab(vocab_file)
            hf_tok = HFRwkvTokenizer(
                vocab_file=vocab_file,
                bos_token="\x16",
                eos_token="\x17",
                pad_token="\x17",
                unk_token="\x16",
                chat_template=CHAT_TEMPLATE,
            )
            processor = ModRWKVProcessor(
                tokenizer=hf_tok,
                image_processor=TinyImageProcessor(),
            )
            output_dir = os.path.join(tmpdir, "exported")
            processor.save_pretrained(output_dir)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "processor.py")))
            self.assertFalse(
                os.path.isfile(os.path.join(output_dir, "processor_core.py"))
            )
            save_processor_core(output_dir)
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "processor_core.py"))
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
            self.assertEqual(tok.pad_id, 24)
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
            first_end = (
                rendered.find(tok.vision_end_token, first_start)
                + len(tok.vision_end_token)
            )
            second_start = rendered.find(tok.vision_start_token, first_end)
            second_end = (
                rendered.find(tok.vision_end_token, second_start)
                + len(tok.vision_end_token)
            )
            self.assertEqual(rendered[first_start:first_end].count(tok.image_token), 1)
            self.assertEqual(rendered[second_start:second_end].count(tok.image_token), 2)

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
            full_tokens = tok.encode(rendered, add_bos=True, add_eos=False)
            supervised = "".join(
                tok.decode(full_tokens[start:end])
                for start, end in tok.assistant_token_spans(messages, counts)
            )
            self.assertIn("Answer one", supervised)
            self.assertIn("Answer two", supervised)
            self.assertNotIn("Question one", supervised)
            self.assertNotIn("Question two", supervised)
            self.assertNotIn("User:", supervised)
            self.assertNotIn("Assistant:", supervised)


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
        ]
        for sample in cases:
            normalized = normalize_mm_chat_sample(sample)
            self.assertEqual(normalized["messages"][0]["role"], "user")
            self.assertEqual(normalized["messages"][1]["role"], "assistant")
            self.assertEqual(len(normalized["images"]), 1)

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
        processed = process_mm_chat_images(
            [
                Image.new("RGB", (128, 128), color="red"),
                Image.new("RGB", (128, 128), color="blue"),
            ],
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            min_pixels=1024,
            max_pixels=4096,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            max_aspect_ratio=50.0,
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
            self.assertNotIn("User:", supervised)
            self.assertNotIn("Assistant:", supervised)
            self.assertNotIn(tok.image_token, supervised)
            self.assertNotIn(tok.vision_start_token, supervised)
            self.assertNotIn(tok.vision_end_token, supervised)

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
            self.assertTrue(torch.equal(input_dict["input"][0, :n], sample["input_ids"]))
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
