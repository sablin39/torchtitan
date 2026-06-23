# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal chat SFT dataset and dataloader."""

import json
import math
import os
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import get_worker_info, IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.processor_core import (
    process_images as process_rwkv_vl_images,
    RWKVVLImageProcessorConfig,
    RWKVVLProcessedImages,
)
from torchtitan.hf_datasets.multimodal.utils.packing import MMSamplePacker
from torchtitan.hf_datasets.multimodal.utils.text import pad_batch_dim, pad_seq_len
from torchtitan.tools.logging import logger


ROLE_TABLE = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "tool_response": "tool",
    "tool_call": "tool_call",
    "human": "user",
    "gpt": "assistant",
}

EMPTY_THINK_PREFIX = "<think>\n</think>\n "


_PIXEL_VALUE_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _resolve_pixel_values_dtype(dtype: str | None) -> torch.dtype | None:
    if dtype is None or dtype == "":
        return None
    normalized = str(dtype).lower()
    if normalized in {"none", "auto"}:
        return None
    if normalized not in _PIXEL_VALUE_DTYPES:
        raise ValueError(
            f"Unsupported pixel_values_dtype={dtype!r}; expected one of "
            f"{sorted(_PIXEL_VALUE_DTYPES)} or 'auto'"
        )
    return _PIXEL_VALUE_DTYPES[normalized]


def _local_parquet_files(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".parquet"):
        return [path]
    if not os.path.isdir(path):
        return []
    parquet_files = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(".parquet"):
                parquet_files.append(os.path.join(root, filename))
    return sorted(parquet_files)


def _load_local_chat_dataset(path: str, *, split: str | None = "train"):
    parquet_files = _local_parquet_files(path)
    if parquet_files:
        return load_dataset(
            "parquet",
            data_files=parquet_files,
            split=split or "train",
            streaming=True,
        )
    load_kwargs = {
        "split": split or "train",
        "streaming": True,
    }
    if os.path.isfile(path) and path.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=path, **load_kwargs)
    return None


def _load_text_chat_dataset(path: str, *, split: str | None = "train"):
    local_dataset = _load_local_chat_dataset(path, split=split)
    if local_dataset is not None:
        return local_dataset

    load_kwargs = {
        "split": split or "train",
        "streaming": True,
    }
    dataset = load_dataset(path, **load_kwargs)
    if isinstance(dataset, DatasetDict):
        split_name = split or "train"
        if split_name not in dataset:
            raise ValueError(
                f"MMChatDataLoader could not find chat split "
                f"{split_name!r}; available splits are {sorted(dataset)}"
            )
        dataset = dataset[split_name]
    return dataset


def _tensor_chunks(value: Any) -> list[torch.Tensor]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [value]
    return [chunk for chunk in value if isinstance(chunk, torch.Tensor)]


def _num_grid_items(value: Any) -> int:
    return sum(int(chunk.shape[0]) for chunk in _tensor_chunks(value))


def _num_grid_patches(grid_thw: torch.Tensor | None) -> int:
    return 0 if grid_thw is None else int(grid_thw.prod(-1).sum().item())


def _flatten_images(images: Any) -> list[Any]:
    if images is None:
        return []
    if isinstance(images, dict) and (
        images.get("bytes") is not None or images.get("path") is not None
    ):
        return [images]
    if isinstance(images, (str, bytes)) or hasattr(images, "convert"):
        return [images]
    if isinstance(images, list | tuple):
        flattened = []
        for image in images:
            flattened.extend(_flatten_images(image))
        return flattened
    return [images]


def normalize_mm_chat_images(images: Any) -> list[Any]:
    return [image for image in _flatten_images(images) if image is not None]


def _normalize_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    normalized = []
    for item in content:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            normalized.append({"type": "text", "text": str(item)})
            continue
        item_type = item.get("type")
        if item_type == "text":
            normalized.append({"type": "text", "text": item.get("text", "")})
        elif item_type in {"image", "image_url"}:
            normalized.append({"type": "image"})
        else:
            normalized.append(
                {"type": "text", "text": str(item.get("text", item.get("value", "")))}
            )
    return normalized


def _text_content_to_string(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_parts = []
    for item in content:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict) or item.get("type") != "text":
            return None
        text_parts.append(str(item.get("text", "")))
    return "".join(text_parts)


def _split_think_content(content: str) -> tuple[str | None, str]:
    start = content.find("<think>")
    end = content.find("</think>", start + len("<think>")) if start >= 0 else -1
    if start < 0 or end < 0:
        return None, content
    reasoning = content[start + len("<think>") : end]
    answer = content[end + len("</think>") :]
    return reasoning.strip("\n "), answer.lstrip()


def _ensure_think_content(
    content: Any,
    *,
    reasoning: Any | None = None,
) -> Any:
    text = _text_content_to_string(content)
    if text is None:
        return content

    existing_reasoning, existing_answer = _split_think_content(text)
    if reasoning not in (None, ""):
        reasoning_text = str(reasoning).strip("\n ")
        answer_text = (
            existing_answer.strip() if existing_reasoning is not None else text.strip()
        )
        return f"<think>\n{reasoning_text}\n</think>\n {answer_text}"

    if existing_reasoning is not None:
        reasoning_text = existing_reasoning.strip("\n ")
        answer_text = existing_answer.strip()
        if reasoning_text:
            return f"<think>\n{reasoning_text}\n</think>\n {answer_text}"
        return EMPTY_THINK_PREFIX + answer_text

    return EMPTY_THINK_PREFIX + text.strip()


def _parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_tool_call(raw_tool_call: Any) -> dict[str, Any]:
    tool_call = _parse_json_maybe(raw_tool_call)
    if not isinstance(tool_call, dict):
        tool_call = {"name": "", "arguments": tool_call}

    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name", tool_call.get("name", ""))
        arguments = function.get("arguments", tool_call.get("arguments", {}))
    else:
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
    normalized = {
        "type": tool_call.get("type", "function"),
        "function": {
            "name": "" if name is None else str(name),
            "arguments": _parse_json_maybe(arguments),
        },
    }
    if "id" in tool_call:
        normalized["id"] = tool_call["id"]
    return normalized


def _normalize_tools(raw_tools: Any) -> list[dict[str, Any]]:
    tools = _parse_json_maybe(raw_tools)
    if not tools:
        return []
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        return []
    return [tool for tool in tools if isinstance(tool, dict)]


def normalize_mm_chat_messages(raw_messages: Any) -> list[dict[str, Any]]:
    raw_messages = _parse_json_maybe(raw_messages)
    if not raw_messages:
        raise ValueError("MM chat sample has no messages")
    if not isinstance(raw_messages, list):
        raise TypeError(
            f"Expected MM chat messages to be a list, got {type(raw_messages).__name__}"
        )
    first = raw_messages[0]
    if not isinstance(first, dict):
        raise TypeError(
            f"Expected each chat turn to be a dict, got {type(first).__name__}"
        )

    if "user" in first and "assistant" in first:
        flattened = []
        for message in raw_messages:
            if not isinstance(message, dict):
                raise TypeError(
                    f"Expected each chat turn to be a dict, got "
                    f"{type(message).__name__}"
                )
            flattened.append({"role": "user", "content": message["user"]})
            flattened.append({"role": "assistant", "content": message["assistant"]})
        raw_messages = flattened

    messages = []
    for message_idx, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"Expected chat turn {message_idx} to be a dict, "
                f"got {type(message).__name__}"
            )
        raw_role = message.get("role", message.get("from"))
        role = raw_role
        if role is None:
            raise ValueError("MM chat message is missing role/from")
        role = ROLE_TABLE.get(str(role), str(role))
        content = message.get("content", message.get("value", ""))
        if role == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_normalize_tool_call(content)],
                }
            )
            continue

        normalized_message = {
            "role": role,
            "content": _normalize_content(content),
        }
        if role == "assistant":
            normalized_message["content"] = _ensure_think_content(
                normalized_message["content"],
                reasoning=message.get("reasoning_content"),
            )
        if "name" in message:
            normalized_message["name"] = message["name"]
        if "tool_call_id" in message:
            normalized_message["tool_call_id"] = message["tool_call_id"]
        tool_calls = message.get("tool_calls")
        if tool_calls is None and message.get("function_call") is not None:
            tool_calls = [message["function_call"]]
        if tool_calls is not None:
            if isinstance(tool_calls, str):
                tool_calls = _parse_json_maybe(tool_calls)
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if isinstance(tool_calls, list):
                normalized_message["tool_calls"] = [
                    _normalize_tool_call(tool_call) for tool_call in tool_calls
                ]
        messages.append(normalized_message)
    return messages


def _count_image_markers(messages: list[dict[str, Any]], image_placeholder: str) -> int:
    count = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            count += content.count(image_placeholder)
            continue
        if content is None:
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"image", "image_url"}:
                count += 1
            elif item.get("type") == "text":
                count += item.get("text", "").count(image_placeholder)
    return count


def _prepend_missing_image_markers(
    messages: list[dict[str, Any]],
    *,
    num_missing: int,
) -> None:
    if num_missing <= 0:
        return
    for message in messages:
        if message["role"] != "user":
            continue
        markers = [{"type": "image"} for _ in range(num_missing)]
        content = message["content"]
        if isinstance(content, str):
            message["content"] = markers + [{"type": "text", "text": content}]
        else:
            message["content"] = markers + content
        return
    raise ValueError("Cannot attach images because chat sample has no user turn")


def normalize_mm_chat_sample(sample: dict[str, Any]) -> dict[str, Any]:
    raw_messages = (
        sample.get("messages") or sample.get("conversations") or sample.get("texts")
    )
    if raw_messages is None:
        raise ValueError(
            "MM chat sample must contain messages, conversations, or texts"
        )

    images = normalize_mm_chat_images(sample.get("images", sample.get("image", [])))
    messages = normalize_mm_chat_messages(raw_messages)
    tools = _normalize_tools(sample.get("tools", sample.get("available_tools", [])))
    existing_markers = _count_image_markers(messages, "<image>")
    _prepend_missing_image_markers(
        messages,
        num_missing=max(len(images) - existing_markers, 0),
    )
    return {"messages": messages, "images": images, "tools": tools}


def validate_mm_chat_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("MM chat sample has no messages")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("MM chat sample has no assistant turn")


def process_mm_chat_images(
    images: list[Any],
    *,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    min_pixels: int,
    max_pixels: int,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    max_aspect_ratio: float,
) -> RWKVVLProcessedImages:
    return process_rwkv_vl_images(
        images,
        RWKVVLImageProcessorConfig(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_merge_size=spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
            max_aspect_ratio=max_aspect_ratio,
        ),
    )


def build_image_token_counts_by_message(
    messages: list[dict[str, Any]],
    image_token_counts: list[int],
    *,
    image_placeholder_token: str,
) -> list[list[int]]:
    counts_by_message = []
    image_idx = 0
    for message in messages:
        counts = []
        content = message["content"]
        if isinstance(content, str):
            items = [{"type": "text", "text": content}]
        elif content is None:
            items = []
        else:
            items = content
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"image", "image_url"}:
                if image_idx >= len(image_token_counts):
                    continue
                counts.append(image_token_counts[image_idx])
                image_idx += 1
            elif item.get("type") == "text":
                for _ in range(item.get("text", "").count(image_placeholder_token)):
                    if image_idx >= len(image_token_counts):
                        continue
                    counts.append(image_token_counts[image_idx])
                    image_idx += 1
        counts_by_message.append(counts)

    if image_idx < len(image_token_counts):
        target_idx = next(
            (
                idx
                for idx, message in enumerate(messages)
                if message.get("role") == "user"
            ),
            0,
        )
        counts_by_message[target_idx] = (
            image_token_counts[image_idx:] + counts_by_message[target_idx]
        )
    return counts_by_message


def _get_source_data_iter(
    data: Any,
    *,
    sample_idx: int,
    hf_state_restored: bool,
) -> Any:
    if hf_state_restored:
        return iter(data)
    if isinstance(data, Dataset):
        worker_info = get_worker_info()
        stride = 1 if worker_info is None else worker_info.num_workers
        offset = 0 if worker_info is None else worker_info.id
        start = sample_idx * stride + offset
        if start >= len(data):
            return iter([])
        if stride == 1:
            return iter(data.select(range(start, len(data))))
        return (data[idx] for idx in range(start, len(data), stride))
    return iter(data)


class MMChatDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        sample_processor: Callable = normalize_mm_chat_sample,
        text_dataset: Dataset | None = None,
        text_sample_probability: float = 0.5,
        seed: int | None = None,
        seq_len: int = 2048,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
        image_mean: tuple[float, ...] = (0.5, 0.5, 0.5),
        image_std: tuple[float, ...] = (0.5, 0.5, 0.5),
        packing_buffer_size: int = 0,
        batch_size: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        max_aspect_ratio: float = 50.0,
        pixel_values_dtype: str | None = "float32",
    ) -> None:
        if text_dataset is not None and not 0.0 <= text_sample_probability <= 1.0:
            raise ValueError(
                "text_sample_probability must be between 0 and 1 when "
                f"text_dataset is set, got {text_sample_probability}"
            )
        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self._text_data = (
            split_dataset_by_node(text_dataset, dp_rank, dp_world_size)
            if text_dataset is not None
            else None
        )
        self._tokenizer = tokenizer
        self._sample_processor = sample_processor
        self.text_sample_probability = float(text_sample_probability)
        self._mix_rng = random.Random((0 if seed is None else seed) + dp_rank)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_aspect_ratio = max_aspect_ratio
        self.pixel_values_dtype = _resolve_pixel_values_dtype(pixel_values_dtype)
        self.infinite = infinite
        self._sample_idx = 0
        self._text_sample_idx = 0
        self._hf_state_restored = False
        self._text_hf_state_restored = False
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = MMSamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )

    def _get_data_iter(self):
        iterator = _get_source_data_iter(
            self._data,
            sample_idx=self._sample_idx,
            hf_state_restored=self._hf_state_restored,
        )
        self._hf_state_restored = False
        return iterator

    def _get_text_data_iter(self):
        if self._text_data is None:
            return iter([])
        iterator = _get_source_data_iter(
            self._text_data,
            sample_idx=self._text_sample_idx,
            hf_state_restored=self._text_hf_state_restored,
        )
        self._text_hf_state_restored = False
        return iterator

    def _iter_samples(self):
        if self._text_data is None:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                yield sample, self._sample_processor, "MM chat"
            return

        image_iter = self._get_data_iter()
        text_iter = self._get_text_data_iter()
        sources = {
            "image": (image_iter, self._sample_processor, "image-text"),
            "text": (text_iter, self._sample_processor, "text"),
        }
        done = {"image": False, "text": False}

        def next_from(source_name: str):
            if done[source_name]:
                return None
            iterator, processor, label = sources[source_name]
            try:
                sample = next(iterator)
            except StopIteration:
                done[source_name] = True
                return None
            if source_name == "text":
                self._text_sample_idx += 1
            else:
                self._sample_idx += 1
            return sample, processor, label

        while not all(done.values()):
            first = (
                "text"
                if self._mix_rng.random() < self.text_sample_probability
                else "image"
            )
            fallback = "image" if first == "text" else "text"
            item = next_from(first) or next_from(fallback)
            if item is not None:
                yield item

    def _tokenize_sample(
        self,
        sample: dict[str, Any],
        sample_processor: Callable | None = None,
    ) -> dict[str, Any] | None:
        processed_sample = (sample_processor or self._sample_processor)(sample)
        messages = processed_sample["messages"]
        images = processed_sample["images"]
        tools = processed_sample.get("tools", [])
        validate_mm_chat_messages(messages)
        processed_images = None
        if images:
            processed_images = process_mm_chat_images(
                images,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                spatial_merge_size=self.spatial_merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
                image_mean=self.image_mean,
                image_std=self.image_std,
                max_aspect_ratio=self.max_aspect_ratio,
            )
            image_counts_by_message = build_image_token_counts_by_message(
                messages,
                processed_images.image_token_counts,
                image_placeholder_token=self._tokenizer.image_placeholder_token,
            )
        else:
            image_counts_by_message = [[] for _ in messages]

        full_text = self._tokenizer.render_mm_chat(
            messages,
            image_counts_by_message,
            add_generation_prompt=False,
            tools=tools,
        )
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._tokenizer.eos_id:
            full_tokens.append(self._tokenizer.eos_id)
        if len(full_tokens) - 1 > self.seq_len:
            return None

        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        spans = self._tokenizer.assistant_token_spans(
            messages,
            image_counts_by_message,
            add_bos=True,
            tools=tools,
        )
        for start, end in spans:
            label_start = max(start - 1, 0)
            source_start = label_start + 1
            source_end = min(end, len(full_tokens))
            if source_start >= source_end:
                continue
            labels[
                label_start : label_start + source_end - source_start
            ] = torch.tensor(full_tokens[source_start:source_end], dtype=torch.long)

        vision_ids = [
            self._tokenizer.vision_start_id,
            self._tokenizer.vision_end_id,
            self._tokenizer.image_id,
        ]
        for token_id in vision_ids:
            labels = torch.where(labels == token_id, IGNORE_INDEX, labels)

        if processed_images is None:
            flat_patches = None
            grid_thw = None
        else:
            flat_patches = processed_images.flat_patches
            if self.pixel_values_dtype is not None and flat_patches.is_floating_point():
                flat_patches = flat_patches.to(self.pixel_values_dtype)
            grid_thw = processed_images.grid_thw

        return {
            "input_ids": input_ids,
            "labels": labels,
            "positions": torch.arange(input_ids.numel(), dtype=torch.long),
            "pixel_values": flat_patches,
            "grid_thw": grid_thw,
            "num_packed_samples": 1,
        }

    def __iter__(self):
        while True:
            for sample, sample_processor, source_name in self._iter_samples():
                try:
                    processed = self._tokenize_sample(sample, sample_processor)
                except Exception as e:
                    logger.warning(f"Skipping {source_name} sample: {e}")
                    continue
                if processed is None:
                    continue
                if self.enable_packing:
                    self.packer.add_sample(processed)
                    if self.packer.has_batch_ready():
                        batch = self.packer.get_next_batch()
                        if batch:
                            yield from batch
                else:
                    yield processed

            if self.enable_packing:
                self.packer.flush()
                while self.packer.has_batch_ready():
                    yield from self.packer.get_next_batch()
                while self.packer.packed_samples:
                    yield self.packer.packed_samples.popleft()

            if not self.infinite:
                break
            self._sample_idx = 0
            self._text_sample_idx = 0

    def state_dict(self):
        state = {
            "sample_idx": self._sample_idx,
        }
        if hasattr(self._data, "state_dict"):
            state["hf_dataset_state"] = self._data.state_dict()
        if self._text_data is not None:
            state["text_sample_idx"] = self._text_sample_idx
            state["mix_rng_state"] = self._mix_rng.getstate()
            if hasattr(self._text_data, "state_dict"):
                state["text_hf_dataset_state"] = self._text_data.state_dict()
        # Packer buffers hold processed image tensors. They are data-dependent,
        # can be multi-GiB with VLM inputs, and are cheap to refill after resume.
        return state

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        if "hf_dataset_state" in state_dict and hasattr(self._data, "load_state_dict"):
            self._data.load_state_dict(state_dict["hf_dataset_state"])
            self._hf_state_restored = True
        if self._text_data is not None:
            self._text_sample_idx = state_dict.get("text_sample_idx", 0)
            if "mix_rng_state" in state_dict:
                self._mix_rng.setstate(state_dict["mix_rng_state"])
            if "text_hf_dataset_state" in state_dict and hasattr(
                self._text_data,
                "load_state_dict",
            ):
                self._text_data.load_state_dict(state_dict["text_hf_dataset_state"])
                self._text_hf_state_restored = True
        if self.enable_packing:
            self.packer._sample_buffer.clear()
            self.packer._next_id = 0
            self.packer.packed_samples.clear()
            if "packer_state" in state_dict:
                packer_state = state_dict["packer_state"]
                self.packer._sample_buffer = {
                    i: s for i, s in enumerate(packer_state["sample_buffer"])
                }
                self.packer._next_id = len(packer_state["sample_buffer"])
                self.packer.packed_samples.extend(packer_state["packed_samples"])


@dataclass
class MMChatCollator:
    batch_size: int
    seq_len: int
    max_images_per_batch: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    tokenizer: Any
    vit_patch_bucket_size: int = 0

    def __post_init__(self) -> None:
        self.vit_patch_bucket_size = int(self.vit_patch_bucket_size)
        if self.vit_patch_bucket_size < 0:
            raise ValueError(
                "vit_patch_bucket_size must be non-negative, "
                f"got {self.vit_patch_bucket_size}"
            )

    @property
    def vit_patch_bucket_unit(self) -> int:
        if self.vit_patch_bucket_size == 0:
            return 0
        return math.lcm(
            self.vit_patch_bucket_size,
            128,
            self.spatial_merge_size**2,
        )

    def collate_images(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_patches = [
            chunk
            for sample in batch
            for chunk in _tensor_chunks(sample.get("pixel_values"))
        ]
        grid_thw_list = [
            chunk
            for sample in batch
            for chunk in _tensor_chunks(sample.get("grid_thw"))
        ]
        patches = torch.cat(all_patches, dim=0)
        grid_thw = torch.cat(grid_thw_list, dim=0)
        real_num_patch = _num_grid_patches(grid_thw)
        if patches.shape[0] != real_num_patch:
            raise ValueError(
                f"Collated pixel_values has {patches.shape[0]} patches, but "
                f"grid_thw describes {real_num_patch} patches"
            )

        bucket_unit = self.vit_patch_bucket_unit
        if bucket_unit > 0 and real_num_patch > 0:
            bucket_num_patch = (
                (real_num_patch + bucket_unit - 1) // bucket_unit
            ) * bucket_unit
            pad_len = bucket_num_patch - real_num_patch
            if pad_len > 0:
                patches = torch.cat(
                    [
                        patches,
                        patches.new_zeros((pad_len, patches.shape[1])),
                    ],
                    dim=0,
                )

        return patches, grid_thw

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = pad_sequence(
            [sample["input_ids"] for sample in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_id,
        )
        labels = pad_sequence(
            [sample["labels"] for sample in batch],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        positions = pad_sequence(
            [sample["positions"] for sample in batch],
            batch_first=True,
            padding_value=0,
        )
        input_ids, labels = pad_seq_len(
            input_ids,
            labels,
            self.seq_len,
            padding_idx=self.tokenizer.pad_id,
            ignore_idx=IGNORE_INDEX,
        )
        if positions.shape[1] < self.seq_len:
            positions = torch.nn.functional.pad(
                positions,
                (0, self.seq_len - positions.shape[1]),
                value=0,
            )
        else:
            positions = positions[:, : self.seq_len]
        input_ids, labels = pad_batch_dim(
            input_ids,
            labels,
            self.batch_size,
            padding_idx=self.tokenizer.pad_id,
            ignore_idx=IGNORE_INDEX,
        )
        if positions.shape[0] < self.batch_size:
            positions = torch.nn.functional.pad(
                positions,
                (0, 0, 0, self.batch_size - positions.shape[0]),
                value=0,
            )
        input_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for sample_idx, sample in enumerate(batch[: self.batch_size]):
            valid_len = min(sample["input_ids"].numel(), self.seq_len)
            input_token_mask[sample_idx, :valid_len] = True
        return input_ids, labels, positions, input_token_mask

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        images_per_sample = [
            _num_grid_items(sample.get("grid_thw")) for sample in batch
        ]
        total_images = sum(images_per_sample)
        while (
            self.max_images_per_batch > 0
            and total_images > self.max_images_per_batch
            and batch
        ):
            removed = images_per_sample.pop()
            total_images -= removed
            batch.pop()
            logger.warning(
                f"Removed sample with {removed} images to keep "
                f"total <= {self.max_images_per_batch}"
            )

        total_images = sum(images_per_sample)
        patches, grids = (
            self.collate_images(batch) if total_images > 0 else (None, None)
        )
        input_ids, labels, positions, input_token_mask = self.collate_text(batch)
        real_num_patch = _num_grid_patches(grids)
        bucket_num_patch = 0 if patches is None else int(patches.shape[0])
        patch_padding = bucket_num_patch - real_num_patch
        pixel_values_bytes = (
            0 if patches is None else patches.numel() * patches.element_size()
        )
        data_stats = {
            "num_images": total_images,
            "num_vit_patches": real_num_patch,
            "num_vit_patches_bucketed": bucket_num_patch,
            "vit_patch_padding": patch_padding,
            "vit_patch_padding_ratio": (
                patch_padding / real_num_patch if real_num_patch > 0 else 0.0
            ),
            "pixel_values_bytes": pixel_values_bytes,
            "nonpad_tokens": int(input_token_mask.sum().item()),
            "sequence_tokens": int(input_ids.numel()),
            "packed_rows": len(batch),
            "packed_docs": sum(
                int(sample.get("num_packed_samples", 1)) for sample in batch
            ),
        }
        input_dict = {
            "input": input_ids,
            "positions": positions,
            "input_token_mask": input_token_mask,
            "pixel_values": patches,
            "grid_thw": grids,
            "pixel_values_videos": None,
            "grid_thw_videos": None,
            "data_stats": data_stats,
            "special_tokens": {
                f"{name}_id": getattr(self.tokenizer, f"{name}_id")
                for name in self.tokenizer.TOKEN_FIELDS
            },
        }
        return input_dict, labels


class MMChatDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset_path: str | None = None
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        data_files: str | None = None
        split: str | None = "train"
        sample_processor: Callable = normalize_mm_chat_sample
        text_dataset_path: str | None = None
        text_split: str | None = "train"
        text_sample_probability: float = 0.5
        infinite: bool = True
        packing_buffer_size: int = 0
        max_images_per_batch: int
        patch_size: int
        temporal_patch_size: int
        spatial_merge_size: int
        min_pixels: int
        max_pixels: int
        image_mean: tuple[float, ...]
        image_std: tuple[float, ...]
        max_aspect_ratio: float = 50.0
        pixel_values_dtype: str | None = "float32"
        vit_patch_bucket_size: int = 0

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        seed: int | None = None,
        **kwargs,
    ):
        if not config.dataset_path:
            raise ValueError("MMChatDataLoader requires dataset_path")
        if (
            config.text_dataset_path
            and not 0.0 <= config.text_sample_probability <= 1.0
        ):
            raise ValueError(
                "text_sample_probability must be between 0 and 1 when "
                f"text_dataset_path is set, got {config.text_sample_probability}"
            )
        local_dataset = None
        if config.data_files is None and not config.load_dataset_kwargs:
            local_dataset = _load_local_chat_dataset(
                config.dataset_path,
                split=config.split,
            )
        if local_dataset is not None:
            dataset = local_dataset
        else:
            load_kwargs = dict(config.load_dataset_kwargs)
            if config.data_files is not None:
                load_kwargs["data_files"] = config.data_files
            if config.split is not None:
                load_kwargs["split"] = config.split
            dataset = load_dataset(config.dataset_path, **load_kwargs)
            if isinstance(dataset, DatasetDict):
                split = config.split or "train"
                if split not in dataset:
                    raise ValueError(
                        f"MMChatDataLoader could not find split {split!r}; "
                        f"available splits are {sorted(dataset)}"
                    )
                dataset = dataset[split]

        text_dataset = None
        if config.text_dataset_path:
            text_dataset = _load_text_chat_dataset(
                config.text_dataset_path,
                split=config.text_split,
            )
        chat_dataset = MMChatDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            sample_processor=config.sample_processor,
            text_dataset=text_dataset,
            text_sample_probability=config.text_sample_probability,
            seed=seed,
            seq_len=seq_len,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
            image_mean=config.image_mean,
            image_std=config.image_std,
            packing_buffer_size=config.packing_buffer_size,
            batch_size=local_batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            max_aspect_ratio=config.max_aspect_ratio,
            pixel_values_dtype=config.pixel_values_dtype,
        )
        collate_fn = MMChatCollator(
            batch_size=local_batch_size,
            seq_len=seq_len,
            max_images_per_batch=config.max_images_per_batch,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            tokenizer=tokenizer,
            vit_patch_bucket_size=config.vit_patch_bucket_size,
        )
        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
            "collate_fn": collate_fn,
        }
        super().__init__(
            chat_dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
