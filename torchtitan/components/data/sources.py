# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Storage adapters for Grain datasets."""

import copy
import fnmatch
import glob
import json
import os
import threading
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import datasets
import grain.python as grain
from datasets.distributed import split_dataset_by_node

from torchtitan.components.data.types import DatasetIterationPolicy
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger


@runtime_checkable
class RandomAccessDataSource(Protocol):
    """Finite data source addressable by integer index."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> Any:
        ...


class SourceConfig(Protocol):
    """Builds a random-access or streaming source."""

    def build(
        self,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> RandomAccessDataSource | grain.IterDataset:
        ...


class IndexedJsonlSource(Configurable):
    """Provides random access to JSONL rows through compact byte offsets."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        patterns: tuple[str, ...]

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        del dataset_iteration_policy
        self._paths = _file_patterns_to_paths(config.patterns)
        self._path_ids = array("I")
        self._byte_offsets = array("Q")
        # TODO(data-jsonl-sidecar): Startup rescans every JSONL file per rank and
        # worker. Build one validated offset index that all processes can memory-map.
        for path_id, path in enumerate(self._paths):
            with open(path, "rb") as file:
                while True:
                    offset = file.tell()
                    line = file.readline()
                    if not line:
                        break
                    if line.strip():
                        self._path_ids.append(path_id)
                        self._byte_offsets.append(offset)

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        path = self._paths[self._path_ids[index]]
        with open(path, "rb") as file:
            file.seek(self._byte_offsets[index])
            return json.loads(file.readline())


class HuggingFaceRandomAccessSource(Configurable):
    """Provides random access to a materialized Hugging Face dataset."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        split: str
        name: str | None = None
        revision: str | None = None
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            duplicated = {"split", "name", "revision", "streaming"} & (
                self.load_dataset_kwargs.keys()
            )
            if duplicated:
                raise ValueError(
                    "first-class Hugging Face fields repeated in kwargs: "
                    f"{sorted(duplicated)}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        del dataset_iteration_policy
        dataset = datasets.load_dataset(
            config.path,
            name=config.name,
            split=config.split,
            revision=config.revision,
            streaming=False,
            **config.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.Dataset):
            raise TypeError(
                "random-access Hugging Face source requires one Dataset; "
                f"got {type(dataset).__qualname__}"
            )
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[index]


class HuggingFaceStreamingSource(Configurable, grain.IterDataset):
    """Provides a DP-sharded Hugging Face stream with cursor checkpointing."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        path: str
        split: str
        name: str | None = None
        revision: str | None = None
        load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
        decode_images: bool = True
        """When False, Image columns yield ``{"bytes": ..., "path": ...}``
        instead of decoded PIL images (HF ``Image(decode=False)``). The row
        processor then owns decoding; this keeps multi-MB pixel payloads out of
        the trainer process when rows cross a process-pool boundary."""
        readahead_mb: int = 0
        """When > 0, spawn a daemon that preads this many MiB ahead of the
        read position of every open local data file, keeping the page cache
        warm for the streaming consumer. Helps local datasets on latency-bound
        storage (e.g. USB-attached drives), where the kernel's small readahead
        window caps per-stream throughput far below the device's aggregate
        bandwidth."""
        num_readahead_threads: int = 2
        """Threads issuing readahead preads per open data file. Each thread
        reads on its own fd because the kernel tracks readahead per open file
        description, so threads sharing one fd serialize on a single window."""

        def __post_init__(self) -> None:
            duplicated = {"split", "name", "revision", "streaming"} & (
                self.load_dataset_kwargs.keys()
            )
            if duplicated:
                raise ValueError(
                    "first-class Hugging Face fields repeated in kwargs: "
                    f"{sorted(duplicated)}"
                )
            if self.readahead_mb < 0:
                raise ValueError(f"readahead_mb must be >= 0, got {self.readahead_mb}")
            if self.readahead_mb > 0 and self.num_readahead_threads < 1:
                raise ValueError(
                    "num_readahead_threads must be >= 1 when readahead_mb > 0, "
                    f"got {self.num_readahead_threads}"
                )

    def __init__(
        self,
        config: Config,
        *,
        dataset_iteration_policy: DatasetIterationPolicy,
    ) -> None:
        super().__init__()
        dataset = datasets.load_dataset(
            config.path,
            name=config.name,
            split=config.split,
            revision=config.revision,
            streaming=True,
            **config.load_dataset_kwargs,
        )
        if not isinstance(dataset, datasets.IterableDataset):
            raise TypeError(
                "streaming Hugging Face source requires one IterableDataset; "
                f"got {type(dataset).__qualname__}"
            )
        if not hasattr(dataset, "state_dict") or not hasattr(
            dataset, "load_state_dict"
        ):
            raise TypeError(
                "Hugging Face streaming source does not support exact resume"
            )
        if not config.decode_images:
            features = dataset.features
            if features is None:
                raise ValueError(
                    "decode_images=False requires a dataset with known features"
                )
            for column, feature in features.items():
                if isinstance(feature, datasets.Image):
                    dataset = dataset.cast_column(
                        column, datasets.Image(mode=feature.mode, decode=False)
                    )
        self._dataset = split_dataset_by_node(
            dataset,
            rank=dataset_iteration_policy.dp_rank,
            world_size=dataset_iteration_policy.dp_world_size,
        )
        self._repeat = dataset_iteration_policy.repeat
        self._shuffle = dataset_iteration_policy.shuffle
        self._current_epoch = 0
        self._readahead_warmer: _LocalFileReadaheadWarmer | None = None
        if config.readahead_mb > 0:
            patterns = _data_files_name_patterns(
                config.load_dataset_kwargs.get("data_files")
            )
            local_match = any(
                glob.glob(os.path.join(config.path, pattern)) for pattern in patterns
            )
            if not patterns or not local_match:
                logger.warning(
                    f"readahead_mb={config.readahead_mb} is set but the "
                    f"data_files patterns match no local files under "
                    f"{config.path!r}; readahead warming is disabled"
                )
            else:
                self._readahead_warmer = _LocalFileReadaheadWarmer(
                    file_name_patterns=patterns,
                    window_bytes=config.readahead_mb * 2**20,
                    num_threads=config.num_readahead_threads,
                )

    @property
    def current_epoch(self) -> int:
        """Epochs completed by this rank's stream (1 after the first wrap).

        The counter advances when the last row of the shard is pulled into the
        downstream pipeline, so buffered rows of the finished epoch may still
        be in flight.
        """
        return self._current_epoch

    def __iter__(self) -> grain.DatasetIterator:
        return _HuggingFaceCursorIterator(
            self._dataset,
            repeat=self._repeat,
            shuffle=self._shuffle,
            source=self,
        )


_READAHEAD_IO_BYTES = 8 * 2**20
# A tar stream is consumed strictly forward, so pages this far behind the read
# position are never re-read; evicting them bounds the cache footprint.
_READAHEAD_KEEP_BEHIND_BYTES = 512 * 2**20
_READAHEAD_POLL_INTERVAL_S = 1.0


def _data_files_name_patterns(data_files: Any) -> tuple[str, ...]:
    """Flatten an HF ``data_files`` value into a tuple of glob patterns."""
    if isinstance(data_files, str):
        values: list[Any] = [data_files]
    elif isinstance(data_files, dict):
        values = [
            value
            for split_values in data_files.values()
            for value in (
                split_values if isinstance(split_values, list) else [split_values]
            )
        ]
    elif isinstance(data_files, list):
        values = data_files
    else:
        return ()
    return tuple(value for value in values if isinstance(value, str))


class _LocalFileReadaheadWarmer:
    """Keeps the page cache warm ahead of open local data files.

    On latency-bound storage the kernel caps buffered readahead at one small
    window per stream, so a sequential consumer gets a fraction of the device
    bandwidth. Worker threads pread large chunks ahead of each matched file's
    read position -- each on its own fd, because the kernel tracks readahead
    per open file description and threads sharing one fd serialize on a
    single window -- and the consumer then reads at page-cache speed.
    posix_fadvise(DONTNEED) well behind the read position bounds the cache
    footprint so a full cache never throttles readahead.
    """

    def __init__(
        self,
        *,
        file_name_patterns: tuple[str, ...],
        window_bytes: int,
        num_threads: int,
    ) -> None:
        self._file_name_patterns = tuple(
            os.path.basename(pattern) for pattern in file_name_patterns
        )
        self._window_bytes = window_bytes
        self._num_threads = num_threads
        self._positions: dict[str, int] = {}
        self._warmed_through: dict[str, int] = {}
        threading.Thread(target=self._run, name="tar-readahead", daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                self._warm_once()
            except OSError:
                pass  # files open and close between the scan and the preads
            time.sleep(_READAHEAD_POLL_INTERVAL_S)

    def _warm_once(self) -> None:
        for path, position in self._scan_read_positions().items():
            warmed_through = self._warmed_through.get(path, 0)
            if position < self._positions.get(path, 0):
                # The stream rewound (epoch restart): re-warm from the new
                # position, since trailing DONTNEED evicted those pages.
                warmed_through = position
            self._positions[path] = position
            warm_end = position + self._window_bytes
            if warm_end <= warmed_through:
                continue
            self._pread_range(path, warmed_through, warm_end)
            self._warmed_through[path] = warm_end
            evict_through = position - _READAHEAD_KEEP_BEHIND_BYTES
            if evict_through > 0:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, evict_through, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)

    def _scan_read_positions(self) -> dict[str, int]:
        positions: dict[str, int] = {}
        for fd_name in os.listdir("/proc/self/fd"):
            try:
                path = os.readlink(f"/proc/self/fd/{fd_name}")
                if not any(
                    fnmatch.fnmatch(os.path.basename(path), pattern)
                    for pattern in self._file_name_patterns
                ):
                    continue
                with open(f"/proc/self/fdinfo/{fd_name}") as fdinfo:
                    pos_line = fdinfo.readline()
                position = int(pos_line.split()[1])
            except (OSError, IndexError, ValueError):
                continue  # not a regular file, or closed mid-scan
            positions[path] = max(positions.get(path, 0), position)
        return positions

    def _pread_range(self, path: str, start: int, end: int) -> None:
        bounds = [
            start + (end - start) * i // self._num_threads
            for i in range(self._num_threads + 1)
        ]
        threads = [
            threading.Thread(
                target=self._pread_subrange,
                args=(path, bounds[i], bounds[i + 1]),
                daemon=True,
            )
            for i in range(self._num_threads)
            if bounds[i + 1] > bounds[i]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    @staticmethod
    def _pread_subrange(path: str, start: int, end: int) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            for offset in range(start, end, _READAHEAD_IO_BYTES):
                os.pread(fd, min(_READAHEAD_IO_BYTES, end - offset), offset)
        finally:
            os.close(fd)


def _file_patterns_to_paths(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """Return sorted, unique absolute paths matched by file patterns.

    Every pattern must match at least one file. Duplicate resolved paths are
    rejected so one file cannot be indexed twice.
    """
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"pattern matched no files: {pattern!r}")
        paths.extend(str(Path(match).resolve()) for match in matches)
    if len(paths) != len(set(paths)):
        raise ValueError("patterns resolve to the same file more than once")
    return tuple(paths)


def _extract_shard_position(state: dict[str, Any]) -> tuple[int, int, int] | None:
    """Effective next-to-yield position in an HF examples-iterable state tree.

    Returns (shard_idx, shard_example_idx, examples_since): the first two are
    a position HF can seek a shard reader to directly (a table/chunk boundary
    held in the ``previous_state`` resume bookkeeping) and the third counts
    examples consumed past that boundary. Layers whose inner reader runs
    ahead of what they yielded (arrow table readers buffer whole tables) are
    still exact. ``previous_state`` subtrees are never descended into; they
    are consumed by their parent node instead.
    """
    previous = state.get("previous_state")
    if isinstance(previous, dict):
        for counter in (
            "num_chunks_since_previous_state",
            "num_examples_since_previous_state",
        ):
            if counter in state:
                return (
                    previous["shard_idx"],
                    previous["shard_example_idx"],
                    state[counter],
                )
    if "shard_idx" in state and "shard_example_idx" in state:
        return state["shard_idx"], state["shard_example_idx"], 0
    for key, value in state.items():
        if key != "previous_state" and isinstance(value, dict):
            position = _extract_shard_position(value)
            if position is not None:
                return position
    return None


def _patch_shard_position(
    state: dict[str, Any],
    shard_idx: int,
    shard_example_idx: int,
    examples_since: int,
) -> bool:
    """Write a shard position into a fresh HF state tree, in place."""
    if "previous_state" in state:
        for counter in (
            "num_chunks_since_previous_state",
            "num_examples_since_previous_state",
        ):
            if counter in state:
                state["previous_state"] = {
                    "shard_idx": shard_idx,
                    "shard_example_idx": shard_example_idx,
                }
                state[counter] = examples_since
                if "cropped_chunk_length" in state:
                    state["cropped_chunk_length"] = 0
                return True
    if "shard_idx" in state and "shard_example_idx" in state:
        state["shard_idx"] = shard_idx
        state["shard_example_idx"] = shard_example_idx
        return True
    for key, value in state.items():
        if key != "previous_state" and isinstance(value, dict):
            if _patch_shard_position(
                value, shard_idx, shard_example_idx, examples_since
            ):
                return True
    return False


class _HuggingFaceCursorIterator(grain.DatasetIterator):
    """Exposes a Hugging Face streaming cursor to Grain checkpoint recursion."""

    def __init__(
        self,
        dataset: datasets.IterableDataset,
        *,
        repeat: bool,
        shuffle: bool,
        source: HuggingFaceStreamingSource | None = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._repeat = repeat
        self._shuffle = shuffle
        self._source = source
        self._epoch = 0
        self._initial_state = dataset.state_dict()
        self._iterator = iter(dataset)

    def __next__(self) -> dict[str, Any]:
        try:
            return next(self._iterator)
        except StopIteration:
            if not self._repeat:
                raise
            self._epoch += 1
            if self._source is not None:
                self._source._current_epoch = self._epoch
            if self._shuffle:
                self._dataset.set_epoch(self._epoch)
            self._dataset.load_state_dict(self._initial_state)
            self._iterator = iter(self._dataset)
            return next(self._iterator)

    def get_state(self) -> dict[str, Any]:
        # Checkpoint only the epoch and the shard position. HF's full
        # state_dict changes shape with iteration position (previous_state
        # resume bookkeeping is None fresh and a nested dict mid-iteration),
        # which a static-layout checkpoint loader (DCP) cannot round-trip;
        # the bare position has the same layout at every position.
        position = _extract_shard_position(self._dataset.state_dict())
        if position is None:
            raise ValueError(
                "Hugging Face streaming state does not expose a shard position"
            )
        shard_idx, shard_example_idx, examples_since = position
        return {
            "epoch": self._epoch,
            "shard_idx": shard_idx,
            "shard_example_idx": shard_example_idx,
            "examples_since_shard_position": examples_since,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._epoch = state["epoch"]
        if self._source is not None:
            self._source._current_epoch = self._epoch
        if self._shuffle:
            self._dataset.set_epoch(self._epoch)
        # Seek directly to the shard position: HF re-initializes the examples
        # chain on the next __iter__ and merge-patches it with the starting
        # state. The merge rejects keys the fresh chain does not have, so
        # patch the chain's own fresh layout (captured at construction)
        # instead of hand-building a chain-shaped dict.
        seek_state = copy.deepcopy(self._initial_state)
        if not _patch_shard_position(
            seek_state,
            state["shard_idx"],
            state["shard_example_idx"],
            state["examples_since_shard_position"],
        ):
            raise ValueError(
                "Hugging Face streaming state does not expose a shard position"
            )
        seek_state["epoch"] = self._dataset.epoch
        self._dataset.load_state_dict(seek_state)
        self._iterator = iter(self._dataset)
