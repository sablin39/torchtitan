# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multimodal chat SFT dataset and dataloader."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.utils.image import (
    calculate_vision_tokens,
    process_image,
    vision_to_patches,
)
from torchtitan.hf_datasets.multimodal.utils.packing import MMSamplePacker
from torchtitan.hf_datasets.multimodal.utils.text import pad_batch_dim, pad_seq_len
from torchtitan.tools.logging import logger


ROLE_TABLE = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
    "gpt": "assistant",
}


def _flatten_images(images: Any) -> list[Any]:
    if images is None:
        return []
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


def normalize_mm_chat_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not raw_messages:
        raise ValueError("MM chat sample has no messages")
    first = raw_messages[0]
    if not isinstance(first, dict):
        raise TypeError(
            f"Expected each chat turn to be a dict, got {type(first).__name__}"
        )

    if "user" in first and "assistant" in first:
        flattened = []
        for message in raw_messages:
            flattened.append({"role": "user", "content": message["user"]})
            flattened.append({"role": "assistant", "content": message["assistant"]})
        raw_messages = flattened

    messages = []
    for message in raw_messages:
        role = message.get("role", message.get("from"))
        if role is None:
            raise ValueError("MM chat message is missing role/from")
        content = message.get("content", message.get("value", ""))
        messages.append(
            {
                "role": ROLE_TABLE.get(str(role), str(role)),
                "content": _normalize_content(content),
            }
        )
    return messages


def _count_image_markers(messages: list[dict[str, Any]], image_placeholder: str) -> int:
    count = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            count += content.count(image_placeholder)
            continue
        for item in content:
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
        sample.get("messages")
        or sample.get("conversations")
        or sample.get("texts")
    )
    if raw_messages is None:
        raise ValueError(
            "MM chat sample must contain messages, conversations, or texts"
        )

    images = normalize_mm_chat_images(sample.get("images", sample.get("image", [])))
    messages = normalize_mm_chat_messages(raw_messages)
    existing_markers = _count_image_markers(messages, "<image>")
    _prepend_missing_image_markers(
        messages,
        num_missing=max(len(images) - existing_markers, 0),
    )
    return {"messages": messages, "images": images}


def validate_mm_chat_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("MM chat sample has no messages")
    if not any(message.get("role") == "assistant" for message in messages):
        raise ValueError("MM chat sample has no assistant turn")


def _normalize_pil_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return image


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
) -> tuple[list[torch.Tensor], list[int]]:
    processed_images = []
    image_token_counts = []
    for image in images:
        image = _normalize_pil_image(image)
        if hasattr(image, "size"):
            width, height = image.size
            if width == 0 or height == 0:
                raise ValueError("Image has zero width or height")
            ratio = max(width / height, height / width)
            if ratio > max_aspect_ratio:
                raise ValueError(
                    f"Image aspect ratio {ratio:.1f} exceeds {max_aspect_ratio}"
                )

        processed = process_image(
            image,
            patch_size=patch_size,
            merge_size=spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_mean=image_mean,
            image_std=image_std,
        )
        if processed is None:
            raise ValueError("Could not process image")

        num_tokens, _, _ = calculate_vision_tokens(
            num_frames=1,
            height=processed.shape[1],
            width=processed.shape[2],
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        processed_images.append(processed)
        image_token_counts.append(num_tokens)
    return processed_images, image_token_counts


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
        items = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for item in items:
            if item.get("type") in {"image", "image_url"}:
                if image_idx >= len(image_token_counts):
                    raise ValueError("More image placeholders than images")
                counts.append(image_token_counts[image_idx])
                image_idx += 1
            elif item.get("type") == "text":
                for _ in range(item.get("text", "").count(image_placeholder_token)):
                    if image_idx >= len(image_token_counts):
                        raise ValueError("More image placeholders than images")
                    counts.append(image_token_counts[image_idx])
                    image_idx += 1
        counts_by_message.append(counts)

    if image_idx != len(image_token_counts):
        raise ValueError(
            "Not all processed images were consumed by chat image markers: "
            f"consumed {image_idx}, got {len(image_token_counts)}"
        )
    return counts_by_message


class MMChatDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        sample_processor: Callable = normalize_mm_chat_sample,
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
    ) -> None:
        self._data = split_dataset_by_node(dataset, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self._sample_processor = sample_processor
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_aspect_ratio = max_aspect_ratio
        self.infinite = infinite
        self._sample_idx = 0
        self._hf_state_restored = False
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = MMSamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )

    def _get_data_iter(self):
        if self._hf_state_restored:
            self._hf_state_restored = False
            return iter(self._data)
        if isinstance(self._data, Dataset):
            if self._sample_idx >= len(self._data):
                return iter([])
            return iter(self._data.select(range(self._sample_idx, len(self._data))))
        return iter(self._data)

    def _tokenize_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        processed_sample = self._sample_processor(sample)
        messages = processed_sample["messages"]
        images = processed_sample["images"]
        validate_mm_chat_messages(messages)
        if not images:
            return None

        processed_images, image_token_counts = process_mm_chat_images(
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
            image_token_counts,
            image_placeholder_token=self._tokenizer.image_placeholder_token,
        )

        full_text = self._tokenizer.render_mm_chat(
            messages,
            image_counts_by_message,
            add_generation_prompt=False,
        )
        full_tokens = self._tokenizer.encode(
            full_text, add_bos=True, add_eos=False
        )
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
        )
        for start, end in spans:
            label_start = max(start - 1, 0)
            source_start = label_start + 1
            source_end = min(end, len(full_tokens))
            if source_start >= source_end:
                continue
            labels[label_start : label_start + source_end - source_start] = (
                torch.tensor(full_tokens[source_start:source_end], dtype=torch.long)
            )

        vision_ids = [
            self._tokenizer.vision_start_id,
            self._tokenizer.vision_end_id,
            self._tokenizer.image_id,
        ]
        for token_id in vision_ids:
            labels = torch.where(labels == token_id, IGNORE_INDEX, labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "positions": torch.arange(input_ids.numel(), dtype=torch.long),
            "pixel_values": processed_images,
        }

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                self._sample_idx += 1
                try:
                    processed = self._tokenize_sample(sample)
                except Exception as e:
                    logger.warning(f"Skipping MM chat sample: {e}")
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

    def state_dict(self):
        state = {"sample_idx": self._sample_idx}
        if hasattr(self._data, "state_dict"):
            state["hf_dataset_state"] = self._data.state_dict()
        if self.enable_packing:
            state["packer_state"] = {
                "sample_buffer": list(self.packer._sample_buffer.values()),
                "packed_samples": list(self.packer.packed_samples),
            }
        return state

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        if "hf_dataset_state" in state_dict and hasattr(self._data, "load_state_dict"):
            self._data.load_state_dict(state_dict["hf_dataset_state"])
            self._hf_state_restored = True
        if self.enable_packing and "packer_state" in state_dict:
            packer_state = state_dict["packer_state"]
            self.packer._sample_buffer = {
                i: s for i, s in enumerate(packer_state["sample_buffer"])
            }
            self.packer._next_id = len(packer_state["sample_buffer"])
            self.packer.packed_samples.clear()
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

    def collate_images(
        self, all_images: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        results = [
            vision_to_patches(
                image,
                self.patch_size,
                self.temporal_patch_size,
                self.spatial_merge_size,
            )
            for image in all_images
        ]
        all_patches = [result[0] for result in results]
        grid_thw_list = [result[1] for result in results]
        merge_unit = self.spatial_merge_size**2
        max_num_patch = max(patches.shape[0] for patches in all_patches)
        if max_num_patch % merge_unit != 0:
            max_num_patch = ((max_num_patch // merge_unit) + 1) * merge_unit
        patch_dim = all_patches[0].shape[1]
        padded_patches = torch.zeros(len(all_patches), max_num_patch, patch_dim)
        for idx, patches in enumerate(all_patches):
            padded_patches[idx, : patches.shape[0]] = patches
        return padded_patches, torch.stack(grid_thw_list, dim=0)

    def collate_text(
        self,
        batch: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return input_ids, labels, positions

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        images_per_sample = [len(sample.get("pixel_values", [])) for sample in batch]
        total_images = sum(images_per_sample)
        while total_images > self.max_images_per_batch and batch:
            removed = images_per_sample.pop()
            total_images -= removed
            batch.pop()
            logger.warning(
                f"Removed sample with {removed} images to keep "
                f"total <= {self.max_images_per_batch}"
            )

        all_images = [
            image
            for sample in batch
            for image in sample.get("pixel_values", [])
        ]
        patches, grids = self.collate_images(all_images) if all_images else (None, None)
        input_ids, labels, positions = self.collate_text(batch)
        input_dict = {
            "input": input_ids,
            "positions": positions,
            "pixel_values": patches,
            "grid_thw": grids,
            "pixel_values_videos": None,
            "grid_thw_videos": None,
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

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if not config.dataset_path:
            raise ValueError("MMChatDataLoader requires dataset_path")
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
        chat_dataset = MMChatDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            sample_processor=config.sample_processor,
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
        )
        collate_fn = MMChatCollator(
            batch_size=local_batch_size,
            seq_len=seq_len,
            max_images_per_batch=config.max_images_per_batch,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            tokenizer=tokenizer,
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
