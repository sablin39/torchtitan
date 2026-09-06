# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-pool row processing for streaming datasets.

Grain's IterDataset maps run on the calling thread, so a CPU-heavy processor
competes with the trainer's main thread for the GIL. The node in this module
pulls rows on a background thread and applies the row processor in a pool of
worker processes, leaving the main thread with only submit/result handling.
Results are yielded in pull order and checkpoint state excludes in-flight
rows, so restore re-processes exactly the rows that had not been emitted.
"""

import multiprocessing
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import grain.python as grain
import numpy as np

from .types import DatasetBuildContext

_WORKER_PROCESSOR: Callable[[Any, np.random.Generator], Any] | None = None


def _worker_init(processor_config: Any, context: DatasetBuildContext) -> None:
    """Build one row processor per worker process."""
    global _WORKER_PROCESSOR
    import torch

    # Workers do pure CPU row processing; keep each on a single thread so a
    # pool of them does not oversubscribe the trainer's cores.
    torch.set_num_threads(1)
    _WORKER_PROCESSOR = processor_config.build(context=context)


def _worker_apply(row: Any, seed: int, counter: int) -> Any:
    if _WORKER_PROCESSOR is None:
        raise RuntimeError("process-pool worker was not initialized")
    rng = np.random.Generator(np.random.Philox(key=[seed, counter]))
    return _WORKER_PROCESSOR(row, rng)


class ProcessPoolMapIterDataset(grain.IterDataset):
    """Maps rows through a processor in a pool of worker processes.

    ``processor_config`` and ``context`` must be picklable: workers are spawned
    (never forked, to stay safe after CUDA initialization) and build their own
    processor via ``processor_config.build(context=context)``. Each row gets a
    deterministic RNG derived from ``seed`` and a pull counter assigned at
    submit time, so per-row randomness survives checkpoint restore independent
    of thread/process scheduling.
    """

    def __init__(
        self,
        parent: grain.IterDataset,
        *,
        processor_config: Any,
        context: DatasetBuildContext,
        num_workers: int,
        seed: int,
        max_in_flight: int | None = None,
    ) -> None:
        super().__init__(parent)
        if num_workers < 1:
            raise ValueError(f"num_workers must be positive, got {num_workers}")
        self._processor_config = processor_config
        self._context = context
        self._num_workers = num_workers
        self._seed = seed
        self._max_in_flight = (
            max_in_flight if max_in_flight is not None else 2 * num_workers
        )

    def __iter__(self) -> grain.DatasetIterator:
        return _ProcessPoolMapIterator(
            self._parent.__iter__(),
            processor_config=self._processor_config,
            context=self._context,
            num_workers=self._num_workers,
            seed=self._seed,
            max_in_flight=self._max_in_flight,
        )


class _ProcessPoolMapIterator(grain.DatasetIterator):
    def __init__(
        self,
        parent: grain.DatasetIterator,
        *,
        processor_config: Any,
        context: DatasetBuildContext,
        num_workers: int,
        seed: int,
        max_in_flight: int,
    ) -> None:
        super().__init__(parent)
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(processor_config, context),
        )
        self._seed = seed
        self._max_in_flight = max_in_flight
        # (parent state before the pull, pull counter, future) per in-flight row.
        self._in_flight: deque[tuple[Any, int, Any]] = deque()
        self._pull_counter = 0
        self._parent_exhausted = False
        self._resume_state: Any = {
            "parent": self._parent.get_state(),
            "counter": 0,
        }
        # Rows the pull thread has read but the main thread has not submitted.
        # Each entry is (parent state before the pull, row).
        self._pulled: deque[tuple[Any, Any]] = deque()
        self._pulled_cond = threading.Condition()
        self._pull_done = False
        self._pull_error: BaseException | None = None
        self._stop_pull = threading.Event()
        self._pull_thread: threading.Thread | None = None
        self._start_pull_thread()

    def _pull_loop(self) -> None:
        """Read rows from the parent ahead of the main thread's submissions."""
        try:
            while not self._stop_pull.is_set():
                pre_pull_state = self._parent.get_state()
                try:
                    row = next(self._parent)
                except StopIteration:
                    with self._pulled_cond:
                        self._pull_done = True
                        self._pulled_cond.notify_all()
                    return
                with self._pulled_cond:
                    while (
                        len(self._pulled) >= self._max_in_flight
                        and not self._stop_pull.is_set()
                    ):
                        self._pulled_cond.wait(timeout=0.1)
                    if self._stop_pull.is_set():
                        # Discard the row; set_state restores the parent cursor.
                        return
                    self._pulled.append((pre_pull_state, row))
                    self._pulled_cond.notify()
        except BaseException as exc:
            with self._pulled_cond:
                self._pull_error = exc
                self._pull_done = True
                self._pulled_cond.notify_all()

    def _start_pull_thread(self) -> None:
        self._pull_thread = threading.Thread(
            target=self._pull_loop, name="process-pool-pull", daemon=True
        )
        self._pull_thread.start()

    def _stop_pull_thread(self) -> None:
        self._stop_pull.set()
        with self._pulled_cond:
            self._pulled_cond.notify_all()
        if self._pull_thread is not None:
            self._pull_thread.join()
        self._pulled.clear()
        self._pull_done = False
        self._pull_error = None
        self._stop_pull.clear()

    def __next__(self) -> Any:
        while not self._parent_exhausted and len(self._in_flight) < self._max_in_flight:
            with self._pulled_cond:
                if not self._pulled:
                    if self._pull_done:
                        self._parent_exhausted = True
                        if self._pull_error is not None:
                            raise self._pull_error
                        break
                    if self._in_flight:
                        # In-flight rows keep the pipeline moving; do not block.
                        break
                    # Nothing to submit or emit; wait for the pull thread.
                    while not self._pulled and not self._pull_done:
                        self._pulled_cond.wait(timeout=0.1)
                    continue
                pre_pull_state, row = self._pulled.popleft()
                self._pulled_cond.notify()
            future = self._executor.submit(
                _worker_apply, row, self._seed, self._pull_counter
            )
            self._in_flight.append((pre_pull_state, self._pull_counter, future))
            self._pull_counter += 1
        if not self._in_flight:
            raise StopIteration
        _, _, future = self._in_flight.popleft()
        # Blocking on the oldest future preserves pull order in the output.
        with self._stats.record_self_time():
            result = future.result()
        # Resume state points at the first row not yet emitted: the pre-pull
        # state and pull counter of the oldest in-flight or pulled row.
        if self._in_flight:
            next_state, next_counter, _ = self._in_flight[0]
        else:
            with self._pulled_cond:
                parent_state = (
                    self._pulled[0][0] if self._pulled else self._parent.get_state()
                )
            next_state, next_counter = parent_state, self._pull_counter
        self._resume_state = {"parent": next_state, "counter": next_counter}
        return self._stats.record_output_spec(result)

    def get_state(self) -> dict[str, Any]:
        return self._resume_state

    def set_state(self, state: dict[str, Any]) -> None:
        self._stop_pull_thread()
        # In-flight rows were pulled from the old position; cancel what is
        # still queued and let stale results drain unread in the pool.
        for _, _, future in self._in_flight:
            future.cancel()
        self._in_flight.clear()
        self._parent.set_state(state["parent"])
        self._pull_counter = state["counter"]
        self._parent_exhausted = False
        self._resume_state = state
        self._start_pull_thread()

    def close(self) -> None:
        self._stop_pull_thread()
        self._executor.shutdown(wait=False, cancel_futures=True)
