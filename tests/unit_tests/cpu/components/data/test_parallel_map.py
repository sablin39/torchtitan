# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import grain.python as grain
import pytest

from torchtitan.components.data.dataset import SampleProcessor, SingleDatasetConfig
from torchtitan.components.data.parallel_map import ProcessPoolMapIterDataset
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


class _FakeTokenizer:
    bos_id = 1
    eos_id = 2

    def encode(self, text, add_bos=False, add_eos=False):
        return []


CONTEXT = DatasetBuildContext(
    tokenizer=_FakeTokenizer(),
    max_context_length=9,
    num_tokens_per_batch=18,
    read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=1),
)


class NoisyScale(SampleProcessor):
    """Scales ``value`` and tags the row with deterministic rng noise."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        factor: int = 2

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del context
        self._factor = config.factor

    def __call__(self, sample, rng):
        return {
            "value": sample["value"] * self._factor,
            "noise": int(rng.integers(0, 1 << 31)),
        }


def _policy(**overrides):
    values = {
        "seed": 42,
        "shuffle": False,
        "repeat": False,
        "dp_rank": 0,
        "dp_world_size": 1,
        "streaming_shuffle_buffer_size": 4,
    }
    return DatasetIterationPolicy(**(values | overrides))


def _pool_dataset(num_rows, *, num_workers=2, seed=42):
    parent = grain.MapDataset.source(
        [{"value": index} for index in range(num_rows)]
    ).to_iter_dataset()
    return ProcessPoolMapIterDataset(
        parent,
        processor_config=NoisyScale.Config(),
        context=CONTEXT,
        num_workers=num_workers,
        seed=seed,
    )


def test_process_pool_preserves_pull_order():
    iterator = iter(_pool_dataset(20))
    rows = list(iterator)

    assert [row["value"] for row in rows] == [2 * index for index in range(20)]
    with pytest.raises(StopIteration):
        next(iterator)
    iterator.close()


def test_process_pool_rng_is_deterministic_across_runs():
    first = list(iter(_pool_dataset(12)))
    second = list(iter(_pool_dataset(12)))

    assert first == second
    assert len({row["noise"] for row in first}) == len(first)


def test_process_pool_restore_is_exact():
    expected = list(iter(_pool_dataset(16)))

    iterator = iter(_pool_dataset(16))
    head = [next(iterator) for _ in range(5)]
    state = iterator.get_state()
    iterator.close()

    restored = iter(_pool_dataset(16))
    restored.set_state(state)

    assert head + list(restored) == expected
    restored.close()


def test_process_pool_rejects_zero_workers():
    with pytest.raises(ValueError, match="num_workers"):
        _pool_dataset(4, num_workers=0)


def test_process_pool_through_single_dataset_config_advances_epoch():
    class StreamingSourceConfig:
        def build(self, *, dataset_iteration_policy):
            del dataset_iteration_policy
            return (
                grain.MapDataset.source([{"value": index} for index in range(6)])
                .repeat()
                .to_iter_dataset()
            )

    config = SingleDatasetConfig(
        source=StreamingSourceConfig(),
        processor=NoisyScale.Config(),
    )
    dataset = config.build(
        context=CONTEXT,
        dataset_iteration_policy=_policy(repeat=True, num_processor_workers=2),
    )
    iterator = iter(dataset)
    rows = [next(iterator) for _ in range(14)]
    iterator.close()

    # Repeat re-enters the map dataset, so values cycle while noise counters
    # keep advancing.
    assert [row["value"] for row in rows] == [2 * (index % 6) for index in range(14)]
    assert len({row["noise"] for row in rows}) == len(rows)
