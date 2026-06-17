# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest

from datasets import Dataset, load_dataset
from PIL import Image

from scripts.text_preprocess.normalize_rwkv_vl_sft import (
    adapt_honey,
    adapt_llava_onevision,
    adapt_nemotron,
    adapt_toucan_oss,
    adapt_toucan_oss_qwen3,
    adapt_toucan_qwen3,
    adapt_toucan_sft,
    main as normalize_main,
)


QWEN_PREAMBLE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Search.", "parameters": {}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def _tool_schema(name="search"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


class TestRwkvVLSftNormalizer(unittest.TestCase):
    def test_llava_non_thinking_image_qa_keeps_plain_nemotron_answer(self):
        row = {
            "id": "llava-1",
            "image": Image.new("RGB", (8, 8), color="red"),
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion?"},
                {"from": "gpt", "value": "Answer."},
            ],
            "data_source": "fixture",
        }
        normalized = adapt_llava_onevision(row, idx=0, image_root=None)
        self.assertEqual(normalized["id"], "llava-1")
        self.assertEqual(normalized["source"], "llava_onevision")
        self.assertEqual(len(normalized["images"]), 1)
        self.assertEqual(normalized["messages"][1]["content"], "Answer.")
        self.assertNotIn("reasoning_content", normalized["messages"][1])
        self.assertEqual(normalized["tools"], [])

    def test_honey_thinking_is_split_into_nemotron_reasoning(self):
        row = {
            "id": "honey-1",
            "images": [Image.new("RGB", (8, 8), color="blue")],
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion?"},
                {"from": "gpt", "value": "<think>\n\n</think>\n\nAnswer."},
                {"from": "human", "value": "Why?"},
                {"from": "gpt", "value": "<think>\nBecause.\n</think>\n\nFinal."},
            ],
        }
        normalized = adapt_honey(row, idx=0, image_root=None)
        self.assertEqual(normalized["messages"][1]["content"], "Answer.")
        self.assertNotIn("reasoning_content", normalized["messages"][1])
        self.assertEqual(normalized["messages"][3]["content"], "Final.")
        self.assertEqual(normalized["messages"][3]["reasoning_content"], "Because.")

    def test_toucan_qwen3_tools_strip_duplicate_system_preamble(self):
        row = {
            "uuid": "toucan-oss-1",
            "subset_name": "fixture",
            "available_tools": json.dumps([_tool_schema()]),
            "messages": json.dumps(
                [
                    {"role": "system", "content": QWEN_PREAMBLE},
                    {"role": "user", "content": "Find it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Need a lookup.",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "search",
                            "arguments": '{"query": "rwkv"}',
                        },
                    },
                    {"role": "function", "name": "search", "content": "result"},
                ]
            ),
            "target_tools": "search",
        }
        normalized = adapt_toucan_qwen3(row, idx=0, image_root=None)
        self.assertEqual(normalized["source"], "toucan_qwen3")
        self.assertEqual(normalized["tools"][0]["function"]["name"], "search")
        self.assertNotIn("# Tools", normalized["messages"][0]["content"])
        self.assertEqual(normalized["messages"][2]["content"], "")
        self.assertEqual(
            normalized["messages"][2]["reasoning_content"], "Need a lookup."
        )
        self.assertEqual(
            normalized["messages"][2]["tool_calls"][0]["function"]["arguments"],
            {"query": "rwkv"},
        )
        self.assertEqual(normalized["messages"][3]["role"], "tool")

    def test_toucan_oss_preserves_call_ids_between_calls_and_results(self):
        row = {
            "uuid": "toucan-oss-1",
            "subset_name": "fixture",
            "available_tools": json.dumps([_tool_schema("price")]),
            "messages": json.dumps(
                [
                    {"role": "system", "content": QWEN_PREAMBLE},
                    {"role": "user", "content": "Find it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Need a lookup.",
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "price",
                            "arguments": '{"symbol": "BTC-USDT"}',
                            "call_id": "call-price",
                        },
                    },
                    {"role": "function", "name": "price", "content": "100"},
                    {"role": "assistant", "content": "Done."},
                ]
            ),
        }
        normalized = adapt_toucan_oss(row, idx=0, image_root=None)
        self.assertEqual(normalized["source"], "toucan_oss")
        self.assertEqual(
            normalized["messages"][2]["tool_calls"][0]["id"],
            "call-price",
        )
        self.assertEqual(normalized["messages"][3]["tool_call_id"], "call-price")
        self.assertEqual(normalized["messages"][3]["role"], "tool")

    def test_toucan_adapter_extracts_embedded_qwen_tool_call_blocks(self):
        row = {
            "uuid": "toucan-embedded-1",
            "tools": json.dumps([_tool_schema()]),
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Find it"},
                    {
                        "role": "assistant",
                        "content": (
                            "Calling now.\n"
                            "<tool_call>\n"
                            '{"name": "search", "arguments": {"query": "rwkv"}}\n'
                            "</tool_call>"
                        ),
                    },
                    {"role": "tool_response", "content": "result"},
                ]
            ),
        }
        normalized = adapt_toucan_sft(row, idx=0, image_root=None)
        self.assertEqual(normalized["messages"][1]["content"], "Calling now.")
        self.assertEqual(
            normalized["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"query": "rwkv"},
        )
        self.assertEqual(normalized["messages"][2]["role"], "tool")

    def test_toucan_sft_tool_call_roles_are_collapsed(self):
        row = {
            "uuid": "toucan-sft-1",
            "tools": json.dumps([_tool_schema()]),
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Find it"},
                    {"role": "assistant", "content": "Calling."},
                    {
                        "role": "tool_call",
                        "content": "{'name': 'search', 'arguments': '{\"query\": \"rwkv\"}'}",
                    },
                    {"role": "tool_response", "content": "result"},
                    {"role": "assistant", "content": "Done."},
                ]
            ),
        }
        normalized = adapt_toucan_sft(row, idx=0, image_root=None)
        self.assertEqual(normalized["messages"][1]["content"], "Calling.")
        self.assertEqual(normalized["messages"][1]["role"], "assistant")
        self.assertEqual(
            normalized["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"query": "rwkv"},
        )
        self.assertEqual(normalized["messages"][2]["role"], "tool")

    def test_legacy_toucan_oss_qwen3_adapter_alias_still_works(self):
        row = {
            "uuid": "toucan-alias-1",
            "available_tools": json.dumps([_tool_schema()]),
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Find it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "function_call": {
                            "name": "search",
                            "arguments": '{"query": "rwkv"}',
                        },
                    },
                ]
            ),
        }
        normalized = adapt_toucan_oss_qwen3(row, idx=0, image_root=None)
        self.assertEqual(normalized["source"], "toucan_oss_qwen3")
        self.assertEqual(
            normalized["messages"][1]["tool_calls"][0]["function"]["arguments"],
            {"query": "rwkv"},
        )

    def test_nemotron_preserves_qwen_tool_schema_and_tool_rows(self):
        row = {
            "model": "fixture",
            "tools": [_tool_schema("verify")],
            "messages": [
                {"role": "system", "content": "Policy."},
                {"role": "user", "content": "Verify me"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "verify",
                                "arguments": '{"user_id": "u1"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call-1"},
                {"role": "assistant", "content": "Verified."},
            ],
        }
        normalized = adapt_nemotron(row, idx=0, image_root=None)
        self.assertEqual(normalized["tools"][0]["function"]["name"], "verify")
        self.assertEqual(
            normalized["messages"][2]["tool_calls"][0]["function"]["arguments"],
            {"user_id": "u1"},
        )
        self.assertEqual(normalized["messages"][3]["role"], "tool")
        self.assertEqual(normalized["images"], [])

    def test_cli_dry_run_and_temp_parquet_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "llava.parquet")
            output_path = os.path.join(tmpdir, "normalized.parquet")
            Dataset.from_list(
                [
                    {
                        "id": "llava-1",
                        "image": Image.new("RGB", (8, 8), color="green"),
                        "conversations": [
                            {"from": "human", "value": "<image>\nQuestion?"},
                            {"from": "gpt", "value": "Answer."},
                        ],
                        "data_source": "fixture",
                    }
                ]
            ).to_parquet(input_path)

            self.assertEqual(
                normalize_main(
                    [
                        "--source",
                        f"llava_onevision={input_path}",
                        "--dry-run",
                        "--limit",
                        "1",
                    ]
                ),
                0,
            )
            self.assertEqual(
                normalize_main(
                    [
                        "--source",
                        f"llava_onevision={input_path}",
                        "--output",
                        output_path,
                        "--limit",
                        "1",
                    ]
                ),
                0,
            )
            loaded = load_dataset("parquet", data_files=output_path, split="train")
            messages = json.loads(loaded[0]["messages"])
            self.assertEqual(messages[1]["content"], "Answer.")
            self.assertNotIn("reasoning_content", messages[1])
            self.assertEqual(json.loads(loaded[0]["tools"]), [])
            self.assertEqual(len(loaded[0]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
