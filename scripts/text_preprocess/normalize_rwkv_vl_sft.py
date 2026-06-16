#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize mixed RWKV-VL SFT datasets.

The normalized row contract is:
    required: id, source, messages, images, tools
    optional: metadata

Messages are normalized into a Nemotron/OpenAI-style semantic schema. Assistant
thinking is stored as ``reasoning_content`` when present; RWKV ``<think>`` text
is rendered by the TorchTitan dataloader immediately before tokenization.

This script never writes unless --output is provided.
"""

from __future__ import annotations

import argparse
import ast
import gc
import glob
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Image as HFImage, load_dataset, Sequence, Value


QWEN_TOOL_MARKER = "# Tools"
QWEN_TOOL_XML_OPEN = "<tools>"
QWEN_TOOL_XML_CLOSE = "</tools>"
QWEN_TOOL_CALL_XML_OPEN = "<tool_call>"
QWEN_TOOL_CALL_XML_CLOSE = "</tool_call>"
QWEN_TOOL_PREAMBLE_RE = re.compile(
    r"(?:^|\n)\s*# Tools\s*\n\n"
    r"You may call one or more functions to assist with the user query\.\s*\n\n"
    r"You are provided with function signatures within <tools></tools> XML tags:\s*\n"
    r"<tools>\s*.*?\s*</tools>\s*\n\n"
    r"For each function call, return a json object with function name and arguments "
    r"within <tool_call></tool_call> XML tags:\s*\n"
    r"<tool_call>\s*\n"
    r"\{\"name\": <function-name>, \"arguments\": <args-json-object>\}\s*\n"
    r"</tool_call>\s*",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class SourceSpec:
    type: str
    path: str
    limit: int | None = None
    image_root: str | None = None


def parse_json_maybe(
    value: Any,
    *,
    warnings: list[str] | None = None,
    context: str = "",
) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as json_error:
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as literal_error:
            if warnings is not None and context:
                warnings.append(
                    f"json_parse_failed:{context}:{json_error.msg}; literal_eval:{type(literal_error).__name__}"
                )
            return value


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def make_id(row: dict[str, Any], source: str, idx: int) -> str:
    for key in ("id", "uuid", "uid", "sample_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{source}:{idx}"


def normalize_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized = []
        for item in content:
            if isinstance(item, str):
                normalized.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"image", "image_url"}:
                    normalized.append({"type": "image"})
                elif item_type == "text":
                    normalized.append(
                        {"type": "text", "text": str(item.get("text", ""))}
                    )
                else:
                    normalized.append(
                        {
                            "type": "text",
                            "text": str(item.get("text", item.get("value", ""))),
                        }
                    )
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized
    return str(content)


def split_think_content(content: str) -> tuple[str | None, str]:
    start = content.find("<think>")
    end = content.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start < 0 or end < 0:
        return None, content
    reasoning = content[start + len("<think>") : end]
    before = content[:start].strip()
    after = content[end + len("</think>") :].lstrip()
    answer = "\n".join(part for part in (before, after) if part)
    return reasoning.strip("\n "), answer.lstrip()


def normalize_assistant_content(
    content: Any,
    reasoning: Any | None,
) -> tuple[Any, str | None]:
    normalized_content = normalize_content(content)
    normalized_reasoning = (
        str(reasoning).strip("\n ") if reasoning not in (None, "") else None
    )
    if not isinstance(normalized_content, str):
        return normalized_content, normalized_reasoning

    existing_reasoning, answer = split_think_content(normalized_content)
    if existing_reasoning is not None:
        normalized_content = answer.strip()
        normalized_reasoning = (
            existing_reasoning.strip("\n ") if existing_reasoning.strip("\n ") else None
        )
    else:
        normalized_content = normalized_content.strip()
    return normalized_content, normalized_reasoning


def has_qwen_tool_preamble(content: str) -> bool:
    return (
        QWEN_TOOL_MARKER in content
        and QWEN_TOOL_XML_OPEN in content
        and QWEN_TOOL_XML_CLOSE in content
        and QWEN_TOOL_CALL_XML_OPEN in content
        and QWEN_TOOL_CALL_XML_CLOSE in content
    )


def strip_qwen_tool_preamble(content: str) -> tuple[str, bool]:
    if not has_qwen_tool_preamble(content):
        return content, False
    stripped = QWEN_TOOL_PREAMBLE_RE.sub("\n", content, count=1).strip()
    if stripped == content.strip():
        return content, False
    return stripped, True


def normalize_tool_schema(raw_tool: Any) -> dict[str, Any] | None:
    tool = parse_json_maybe(raw_tool)
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
    elif "function" in tool and isinstance(tool["function"], dict):
        function = tool["function"]
    else:
        function = tool

    name = function.get("name") or function.get("tool_name") or tool.get("name")
    if not name:
        return None
    description = function.get("description", tool.get("description", ""))
    parameters = function.get("parameters", tool.get("parameters", {}))
    parameters = parse_json_maybe(parameters)
    if not isinstance(parameters, dict):
        parameters = {}
    normalized = {
        "type": "function",
        "function": {
            "name": str(name),
            "description": "" if description is None else str(description),
            "parameters": parameters,
        },
    }
    if "strict" in function:
        normalized["function"]["strict"] = bool(function["strict"])
    elif "strict" in tool:
        normalized["function"]["strict"] = bool(tool["strict"])
    return normalized


def normalize_tools(raw_tools: Any, *, warnings: list[str]) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(raw_tools, warnings=warnings, context="tools")
    if not parsed:
        return []
    tools = as_list(parsed)
    normalized = []
    for idx, tool in enumerate(tools):
        normalized_tool = normalize_tool_schema(tool)
        if normalized_tool is None:
            warnings.append(f"tool_schema_dropped:{idx}")
            continue
        normalized.append(normalized_tool)
    return normalized


def normalize_tool_call(
    raw_tool_call: Any,
    *,
    warnings: list[str],
    context: str,
) -> dict[str, Any]:
    parsed = parse_json_maybe(raw_tool_call, warnings=warnings, context=context)
    if not isinstance(parsed, dict):
        parsed = {"name": "", "arguments": parsed}

    function = parsed.get("function")
    if isinstance(function, dict):
        name = function.get("name", parsed.get("name", ""))
        arguments = function.get("arguments", parsed.get("arguments", {}))
    else:
        name = parsed.get("name", parsed.get("tool_name", ""))
        arguments = parsed.get("arguments", parsed.get("args", {}))
    arguments = parse_json_maybe(
        arguments,
        warnings=warnings,
        context=f"{context}.arguments",
    )
    return {
        "type": "function",
        "function": {
            "name": "" if name is None else str(name),
            "arguments": arguments,
        },
    }


def maybe_strip_system_tool_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    if not messages or messages[0].get("role") != "system":
        return
    content = messages[0].get("content")
    if not isinstance(content, str) or not has_qwen_tool_preamble(content):
        return
    if tools:
        stripped, changed = strip_qwen_tool_preamble(content)
        if changed:
            messages[0]["content"] = stripped
        else:
            warnings.append("qwen_tool_preamble_detected_but_not_stripped")
    else:
        warnings.append("qwen_tool_preamble_left_without_parsed_tools")


def normalize_chat_messages(
    raw_messages: Any,
    *,
    warnings: list[str],
) -> list[dict[str, Any]]:
    parsed = parse_json_maybe(raw_messages, warnings=warnings, context="messages")
    if not parsed:
        raise ValueError("source row has no messages")

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise TypeError(f"messages must be a list, got {type(parsed).__name__}")

    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []
    for idx, raw_message in enumerate(parsed):
        if not isinstance(raw_message, dict):
            warnings.append(f"message_dropped:{idx}:not_dict")
            continue

        raw_role = raw_message.get("role", raw_message.get("from"))
        role = str(raw_role) if raw_role is not None else ""
        role = {
            "human": "user",
            "gpt": "assistant",
            "function": "tool",
            "tool_response": "tool",
        }.get(role, role)
        content = raw_message.get("content", raw_message.get("value", ""))

        if role == "tool_call":
            pending_tool_calls.append(
                normalize_tool_call(
                    content,
                    warnings=warnings,
                    context=f"message:{idx}.tool_call",
                )
            )
            continue

        if pending_tool_calls and role != "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": pending_tool_calls,
                }
            )
            pending_tool_calls = []

        if role == "assistant":
            content, reasoning = normalize_assistant_content(
                content,
                raw_message.get("reasoning_content"),
            )
            normalized: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if reasoning not in (None, ""):
                normalized["reasoning_content"] = reasoning

            tool_calls = raw_message.get("tool_calls")
            if tool_calls is None and raw_message.get("function_call") is not None:
                tool_calls = [raw_message["function_call"]]
            if tool_calls is not None:
                parsed_tool_calls = parse_json_maybe(
                    tool_calls,
                    warnings=warnings,
                    context=f"message:{idx}.tool_calls",
                )
                if isinstance(parsed_tool_calls, dict):
                    parsed_tool_calls = [parsed_tool_calls]
                if isinstance(parsed_tool_calls, list):
                    normalized["tool_calls"] = [
                        normalize_tool_call(
                            tool_call,
                            warnings=warnings,
                            context=f"message:{idx}.tool_calls:{tool_idx}",
                        )
                        for tool_idx, tool_call in enumerate(parsed_tool_calls)
                    ]
                else:
                    warnings.append(f"tool_calls_dropped:{idx}:not_list")
            messages.append(normalized)
            continue

        if role == "tool":
            normalized_tool = {"role": "tool", "content": normalize_content(content)}
            if raw_message.get("name") is not None:
                normalized_tool["name"] = str(raw_message["name"])
            if raw_message.get("tool_call_id") is not None:
                normalized_tool["tool_call_id"] = str(raw_message["tool_call_id"])
            messages.append(normalized_tool)
            continue

        if role in {"system", "user"}:
            messages.append({"role": role, "content": normalize_content(content)})
            continue

        warnings.append(f"message_role_dropped:{idx}:{role}")

    if pending_tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": pending_tool_calls,
            }
        )
    if not messages:
        raise ValueError("source row produced no normalized messages")
    return messages


def normalize_images(
    raw_images: Any,
    *,
    image_root: str | None = None,
    warnings: list[str],
) -> list[Any]:
    images = []
    for idx, image in enumerate(as_list(raw_images)):
        if image is None:
            continue
        if isinstance(image, dict):
            path = image.get("path") or image.get("image") or image.get("file_name")
            if path and image_root:
                images.append(os.path.join(image_root, str(path)))
            elif path:
                warnings.append(f"image_path_skipped:{idx}:missing_image_root")
            elif image.get("bytes") is not None:
                images.append({"bytes": image["bytes"]})
            else:
                warnings.append(f"image_dict_skipped:{idx}")
            continue
        if isinstance(image, str):
            if image_root:
                images.append(os.path.join(image_root, image))
            else:
                warnings.append(f"image_path_skipped:{idx}:missing_image_root")
            continue
        images.append(image)
    return images


def base_row(
    row: dict[str, Any],
    *,
    source_type: str,
    idx: int,
    messages: list[dict[str, Any]],
    images: list[Any],
    tools: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "data_source",
        "source",
        "subset_name",
        "target_tools",
        "domain",
        "model",
        "metadata",
    ):
        if key in row and row[key] not in (None, ""):
            metadata[key] = row[key]
    if warnings:
        metadata["warnings"] = warnings
    return {
        "id": make_id(row, source_type, idx),
        "source": source_type,
        "messages": messages,
        "images": images,
        "tools": tools,
        "metadata": metadata,
    }


def adapt_llava_onevision(
    row: dict[str, Any],
    *,
    idx: int,
    image_root: str | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    messages = normalize_chat_messages(
        row.get("conversations", row.get("messages")),
        warnings=warnings,
    )
    images = normalize_images(
        row.get("images", row.get("image", [])),
        image_root=image_root,
        warnings=warnings,
    )
    return base_row(
        row,
        source_type="llava_onevision",
        idx=idx,
        messages=messages,
        images=images,
        tools=[],
        warnings=warnings,
    )


def adapt_honey(
    row: dict[str, Any],
    *,
    idx: int,
    image_root: str | None,
) -> dict[str, Any]:
    warnings: list[str] = []
    messages = normalize_chat_messages(
        row.get("conversations", row.get("messages")),
        warnings=warnings,
    )
    images = normalize_images(
        row.get("images", row.get("image", [])),
        image_root=image_root,
        warnings=warnings,
    )
    return base_row(
        row,
        source_type="honey",
        idx=idx,
        messages=messages,
        images=images,
        tools=[],
        warnings=warnings,
    )


def adapt_toucan_oss_qwen3(
    row: dict[str, Any],
    *,
    idx: int,
    image_root: str | None,
) -> dict[str, Any]:
    del image_root
    warnings: list[str] = []
    tools = normalize_tools(row.get("available_tools", []), warnings=warnings)
    messages = normalize_chat_messages(
        row.get("messages"),
        warnings=warnings,
    )
    maybe_strip_system_tool_prompt(messages, tools, warnings)
    return base_row(
        row,
        source_type="toucan_oss_qwen3",
        idx=idx,
        messages=messages,
        images=[],
        tools=tools,
        warnings=warnings,
    )


def adapt_toucan_sft(
    row: dict[str, Any],
    *,
    idx: int,
    image_root: str | None,
) -> dict[str, Any]:
    del image_root
    warnings: list[str] = []
    tools = normalize_tools(row.get("tools", []), warnings=warnings)
    messages = normalize_chat_messages(
        row.get("messages"),
        warnings=warnings,
    )
    maybe_strip_system_tool_prompt(messages, tools, warnings)
    return base_row(
        row,
        source_type="toucan_sft",
        idx=idx,
        messages=messages,
        images=[],
        tools=tools,
        warnings=warnings,
    )


def adapt_nemotron(
    row: dict[str, Any],
    *,
    idx: int,
    image_root: str | None,
) -> dict[str, Any]:
    del image_root
    warnings: list[str] = []
    tools = normalize_tools(row.get("tools", []), warnings=warnings)
    messages = normalize_chat_messages(
        row.get("messages"),
        warnings=warnings,
    )
    maybe_strip_system_tool_prompt(messages, tools, warnings)
    return base_row(
        row,
        source_type="nemotron",
        idx=idx,
        messages=messages,
        images=[],
        tools=tools,
        warnings=warnings,
    )


ADAPTERS = {
    "llava_onevision": adapt_llava_onevision,
    "honey": adapt_honey,
    "toucan_oss_qwen3": adapt_toucan_oss_qwen3,
    "toucan_sft": adapt_toucan_sft,
    "nemotron": adapt_nemotron,
}


def parse_source_spec(value: str) -> SourceSpec:
    chunks = value.split(",")
    parts = chunks[0].split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            "--source must use adapter=glob, for example honey=/path/**/*.parquet"
        )
    source_type, source_path = parts
    kwargs: dict[str, Any] = {}
    for chunk in chunks[1:]:
        if not chunk:
            continue
        option = chunk.split("=", 1)
        if len(option) != 2 or not option[0]:
            raise argparse.ArgumentTypeError(
                f"Invalid --source option {chunk!r}; expected key=value"
            )
        key, raw_value = option
        if key == "limit":
            kwargs["limit"] = int(raw_value)
        elif key == "image_root":
            kwargs["image_root"] = raw_value
        else:
            raise argparse.ArgumentTypeError(f"Unsupported --source option {key!r}")
    return SourceSpec(type=source_type, path=source_path, **kwargs)


def source_specs_from_manifest(path: str) -> list[SourceSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("sources", [])
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list or an object with a sources list")
    specs = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest source {idx} must be an object")
        source_type = item.get("type")
        source_path = item.get("path") or item.get("glob") or item.get("data_files")
        if source_type not in ADAPTERS:
            raise ValueError(f"Unsupported source type in manifest: {source_type!r}")
        if not source_path:
            raise ValueError(f"Manifest source {idx} is missing path/glob/data_files")
        specs.append(
            SourceSpec(
                type=str(source_type),
                path=str(source_path),
                limit=int(item["limit"]) if item.get("limit") is not None else None,
                image_root=item.get("image_root"),
            )
        )
    return specs


def source_dataset_kind(path_pattern: str) -> str:
    matches = glob.glob(path_pattern, recursive=True)
    probe = matches[0] if matches else path_pattern
    suffix = Path(probe).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".jsonl", ".json"}:
        return "json"
    raise ValueError(f"Unsupported source file suffix for {path_pattern!r}")


def iter_source_rows(spec: SourceSpec) -> Iterator[dict[str, Any]]:
    kind = source_dataset_kind(spec.path)
    dataset = load_dataset(kind, data_files=spec.path, split="train", streaming=True)
    if spec.type == "llava_onevision":
        dataset = dataset.cast_column("image", HFImage(decode=False))
    elif spec.type == "honey":
        dataset = dataset.cast_column("images", Sequence(HFImage(decode=False)))
    adapter = ADAPTERS[spec.type]
    row_iter = iter(dataset)
    try:
        for idx, row in enumerate(row_iter):
            if spec.limit is not None and idx >= spec.limit:
                break
            yield adapter(row, idx=idx, image_root=spec.image_root)
    finally:
        close_iter = getattr(row_iter, "close", None)
        if close_iter is not None:
            close_iter()
        close = getattr(dataset, "close", None)
        if close is not None:
            close()
        del row_iter
        del dataset
        gc.collect()


def iter_normalized_rows(
    specs: Iterable[SourceSpec],
    *,
    global_limit: int | None,
) -> Iterator[dict[str, Any]]:
    emitted = 0
    for spec in specs:
        if spec.type not in ADAPTERS:
            raise ValueError(f"Unsupported source type: {spec.type!r}")
        for row in iter_source_rows(spec):
            yield row
            emitted += 1
            if global_limit is not None and emitted >= global_limit:
                return


def _jsonl_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(row)
    if row["images"]:
        serializable["images"] = []
        metadata = dict(serializable.get("metadata") or {})
        metadata["warnings"] = list(metadata.get("warnings", [])) + [
            "images_omitted_from_jsonl_output"
        ]
        serializable["metadata"] = metadata
    return serializable


PARQUET_FEATURES = Features(
    {
        "id": Value("string"),
        "source": Value("string"),
        "messages": Value("string"),
        "images": Sequence(HFImage(decode=False)),
        "tools": Value("string"),
        "metadata": Value("string"),
    }
)


def _parquet_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a normalized row into a stable Arrow storage schema.

    The in-memory normalized contract deliberately allows rich nested values:
    message content can be a string or a typed content list, tool arguments can
    be objects or strings, and tool parameter schemas can differ per row. Arrow
    has no practical union type for that shape, so parquet output stores those
    irregular fields as JSON strings. ``MMChatDataset`` already parses JSON
    strings for ``messages`` and ``tools`` on load, preserving training behavior.
    """

    return {
        "id": str(row["id"]),
        "source": str(row["source"]),
        "messages": json.dumps(row["messages"], ensure_ascii=False),
        "images": row.get("images") or [],
        "tools": json.dumps(row.get("tools") or [], ensure_ascii=False),
        "metadata": json.dumps(row.get("metadata") or {}, ensure_ascii=False),
    }


def _image_preview(image: Any) -> str:
    if isinstance(image, dict):
        if image.get("path"):
            return f"ImageDict(path={image['path']})"
        if image.get("bytes") is not None:
            return f"ImageDict(bytes={len(image['bytes'])})"
        return "ImageDict"
    return type(image).__name__


def _parquet_shard_path(output: str, shard_idx: int, *, multi_shard: bool) -> str:
    path = Path(output)
    if path.suffix.lower() == ".parquet":
        if shard_idx == 0 and not multi_shard:
            return str(path)
        return str(path.with_name(f"{path.stem}-{shard_idx:05d}{path.suffix}"))
    return str(path / f"part-{shard_idx:05d}.parquet")


def write_rows(
    rows: Iterable[dict[str, Any]],
    output: str,
    *,
    output_format: str,
    shard_size: int,
) -> int:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    output_path = Path(output)
    output_dir = output_path if output_path.suffix == "" else output_path.parent
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    if output_format == "jsonl":
        with open(output, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_jsonl_safe_row(row), ensure_ascii=False) + "\n")
                count += 1
        return count
    if output_format != "parquet":
        raise ValueError(f"Unsupported output_format={output_format!r}")

    shard: list[dict[str, Any]] = []
    shard_idx = 0
    for row in rows:
        shard.append(_parquet_safe_row(row))
        count += 1
        if len(shard) >= shard_size:
            Dataset.from_list(shard, features=PARQUET_FEATURES).to_parquet(
                _parquet_shard_path(output, shard_idx, multi_shard=True)
            )
            shard = []
            shard_idx += 1
    if shard:
        Dataset.from_list(shard, features=PARQUET_FEATURES).to_parquet(
            _parquet_shard_path(output, shard_idx, multi_shard=shard_idx > 0)
        )
    return count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize mixed RWKV-VL SFT datasets into a common row schema."
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source_spec,
        default=[],
        help=(
            "Source in adapter=glob form. May be repeated. Optional per-source "
            "settings use commas, e.g. honey=/path/**/*.parquet,limit=10."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional JSON manifest with a sources list.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output file. Required for writes.",
    )
    parser.add_argument(
        "--output-format",
        choices=("parquet", "jsonl"),
        default=None,
        help="Output format. Defaults from --output suffix, or parquet.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print normalized examples and counts without writing.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Alias for --dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Global maximum number of normalized rows to read.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="Number of normalized sample previews to print in dry-run mode.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Rows per parquet shard when writing normalized parquet.",
    )
    return parser


def infer_output_format(output: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if output and output.lower().endswith(".jsonl"):
        return "jsonl"
    return "parquet"


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    specs = list(args.source)
    if args.manifest:
        specs.extend(source_specs_from_manifest(args.manifest))
    if not specs:
        parser.error("Provide at least one --source or --manifest.")

    dry_run = args.dry_run or args.sample_only
    if not dry_run and not args.output:
        parser.error("--output is required unless --dry-run/--sample-only is set.")

    if dry_run:
        count = 0
        for row in iter_normalized_rows(specs, global_limit=args.limit):
            if count < args.print_samples:
                preview = dict(row)
                preview["images"] = [_image_preview(image) for image in row["images"]]
                print(
                    json.dumps(
                        {"sample": count, "row": preview},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            count += 1
        print(f"normalized_rows={count}")
        return 0

    output_format = infer_output_format(args.output, args.output_format)
    count = write_rows(
        iter_normalized_rows(specs, global_limit=args.limit),
        args.output,
        output_format=output_format,
        shard_size=args.shard_size,
    )
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
