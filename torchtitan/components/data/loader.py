# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grain-backed TorchTitan dataloader."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any

import grain.python as grain
import tyro
from grain import experimental as grain_experimental
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.components.data.collators import Collator, TextCollator, TrainerBatch
from torchtitan.components.data.dataset import DatasetConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable


# NOTE: This class deliberately inherits from `Exception` and not `StopIteration`.
# According to PEP 479, raising a `StopIteration` or its subclass from within a
# generator will wrap it in a `RuntimeError`. Since this exception is designed
# to be raised from a generator-based dataloader and caught by the training loop,
# inheriting from `StopIteration` would make it uncatchable and would crash the
# program.
# See: https://peps.python.org/pep-0479/
class DataloaderExhaustedError(Exception):
    """An exception that indicates dataloader exhaustion."""

    pass


def _find_epoch_source(dataset: Any) -> Any | None:
    """Return the pipeline node that tracks epoch completion, if any.

    Walks the Grain parent tree for a node exposing ``current_epoch`` (e.g.
    ``HuggingFaceStreamingSource``). Returns None for pipelines without one.
    """
    seen: set[int] = set()
    stack = [dataset]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(getattr(node, "current_epoch", None), int):
            return node
        parents = getattr(node, "parents", None)
        if parents is not None:
            stack.extend(parents)
    return None


class _TokenBudgetBatchIterator(grain.DatasetIterator):
    """Accumulate rows until the next row would exceed a token budget.

    ``row_cost`` prices each row before it joins the batch. The first row that
    would overflow is buffered and becomes the next batch's first row, so no
    row is dropped or duplicated. ``get_state`` reports the parent state at the
    last emitted batch boundary (excluding the buffered row), which keeps
    checkpoint restore exact for variable-size batches.
    """

    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        batch_fn,
        row_cost,
        token_budget: int,
    ) -> None:
        super().__init__(parent)
        self._batch_fn = batch_fn
        self._row_cost = row_cost
        self._token_budget = token_budget
        self._buffered_row: Any | None = None
        self._resume_state = self._parent.get_state()

    def __next__(self):
        rows: list[Any] = []
        total_cost = 0
        if self._buffered_row is not None:
            rows.append(self._buffered_row)
            total_cost = self._row_cost(self._buffered_row)
            self._buffered_row = None
        resume_state = None
        while True:
            pre_pull_state = self._parent.get_state()
            try:
                row = next(self._parent)
            except StopIteration:
                resume_state = self._parent.get_state()
                break
            cost = self._row_cost(row)
            if rows and total_cost + cost > self._token_budget:
                self._buffered_row = row
                resume_state = pre_pull_state
                break
            rows.append(row)
            total_cost += cost
        if not rows:
            raise StopIteration
        self._resume_state = resume_state
        with self._stats.record_self_time():
            return self._stats.record_output_spec(self._batch_fn(rows))

    def get_state(self):
        return self._resume_state

    def set_state(self, state) -> None:
        self._buffered_row = None
        self._parent.set_state(state)
        self._resume_state = self._parent.get_state()


class _TokenBudgetBatchIterDataset(grain.IterDataset):
    """Batch rows by a token budget instead of a fixed row count.

    Selected when the collator provides ``row_cost(row) -> int`` and
    ``packing_token_budget() -> int``; the batch callable receives a variable
    number of rows whose costs sum to at most the budget.
    """

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        batch_fn,
        row_cost,
        token_budget: int,
    ) -> None:
        super().__init__(parent)
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self._batch_fn = batch_fn
        self._row_cost = row_cost
        self._token_budget = token_budget

    def __iter__(self) -> grain.DatasetIterator:
        return _TokenBudgetBatchIterator(
            self._parent.__iter__(),
            batch_fn=self._batch_fn,
            row_cost=self._row_cost,
            token_budget=self._token_budget,
        )


class BaseDataLoader(Stateful, ABC, Configurable):
    """Enforces the `Stateful`, `state_dict()`, and `load_state_dict()` contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[TrainerBatch]:
        ...

    def close(self) -> None:
        pass


class GrainDataLoader(BaseDataLoader):
    """Batches and checkpoints one composed Grain dataset graph."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        dataset: Annotated[DatasetConfig, tyro.conf.Suppress]
        collator: Annotated[Collator.Config, tyro.conf.Suppress] = field(
            default_factory=TextCollator.Config
        )
        seed: int = 42
        shuffle: Annotated[bool, tyro.conf.Suppress] = True
        repeat: Annotated[bool, tyro.conf.Suppress] = True
        streaming_shuffle_buffer_size: Annotated[int, tyro.conf.Suppress] = 1_000
        """Streaming rows retained per rank for approximate shuffling."""
        read_options: Annotated[grain.ReadOptions, tyro.conf.Suppress] = field(
            default_factory=grain.ReadOptions
        )
        """Concurrent indexed reads used when a `MapDataset` becomes an `IterDataset`."""
        num_prefetch_batches: Annotated[int, tyro.conf.Suppress] = 2
        """Collated batches queued per rank for trainer consumption."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        max_context_length: int,
        num_tokens_per_batch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        # Validate the run policy.
        # TODO(data-finite-dp): Support finite distributed datasets with a global
        # remainder policy. Simple map datasets can truncate or pad before DP
        # sharding; filtered, mixed, packed, and streaming datasets need coordinated
        # exhaustion so every rank runs the same number of steps.
        if dp_world_size > 1 and not config.repeat:
            raise ValueError(
                "repeat=False with data parallelism can exhaust ranks at different "
                "steps and hang collectives; use repeat=True with a trainer-"
                "controlled step count"
            )
        self._dp_world_size = dp_world_size
        self._rank_id = f"dp_rank_{dp_rank}"

        # Build the dataset graph and collator.
        read_options = config.read_options
        context = DatasetBuildContext(
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
            read_options=read_options,
        )
        dataset_iteration_policy = DatasetIterationPolicy(
            seed=config.seed,
            shuffle=config.shuffle,
            repeat=config.repeat,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            streaming_shuffle_buffer_size=config.streaming_shuffle_buffer_size,
        )

        dataset = config.dataset.build(
            context=context,
            dataset_iteration_policy=dataset_iteration_policy,
        )
        self._epoch_source = _find_epoch_source(dataset)
        collator = config.collator.build(context=context)

        # TODO(data-multiprocessing): CPU-heavy processing should use multiple
        # processes rather than only threads. Grain can divide map-style data among
        # workers, but packing and mixing map data with a stream produce an iterable
        # before the loader sees it. Investigate an earlier boundary where one
        # shared worker pool processes samples, instead of creating a pool per
        # dataset or packing per worker.
        if isinstance(dataset, grain.MapDataset):
            dataset = dataset.to_iter_dataset(read_options=read_options)

        # Batch and collate samples. Collators that expose ``row_cost`` and a
        # positive ``packing_token_budget()`` are batched by token budget with
        # variable row counts; all others use a fixed row count.
        row_cost = getattr(collator, "row_cost", None)
        packing_token_budget = getattr(collator, "packing_token_budget", None)
        token_budget = (
            packing_token_budget() if packing_token_budget is not None else None
        )
        if row_cost is not None and token_budget:
            dataset = _TokenBudgetBatchIterDataset(
                dataset,
                batch_fn=collator,
                row_cost=row_cost,
                token_budget=token_budget,
            )
        else:
            dataset = dataset.batch(
                collator.num_rows_per_batch(),
                drop_remainder=config.repeat,
                batch_fn=collator,
            )

        # Queue completed batches while the trainer consumes the previous batch.
        dataset = grain_experimental.ThreadPrefetchIterDataset(
            dataset, prefetch_buffer_size=config.num_prefetch_batches
        )
        self._iterator = iter(dataset)

    def __iter__(self) -> Iterator[TrainerBatch]:
        return self._iterator

    @property
    def epochs_completed(self) -> int | None:
        """Epochs completed by this rank's source stream, if it tracks them."""
        if self._epoch_source is None:
            return None
        return self._epoch_source.current_epoch

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dp_world_size": self._dp_world_size,
            self._rank_id: self._iterator.get_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["version"] != 1:
            raise ValueError(
                f"unsupported GrainDataLoader state version {state_dict['version']}"
            )
        if state_dict["dp_world_size"] != self._dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        if self._rank_id not in state_dict:
            raise ValueError(
                f"checkpoint is missing dataloader state for {self._rank_id}"
            )
        try:
            self._iterator.set_state(state_dict[self._rank_id])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._iterator.close()
