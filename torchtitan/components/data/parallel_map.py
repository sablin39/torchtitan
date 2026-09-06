# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-pool row processing for streaming datasets.

Grain's IterDataset maps run on the calling thread, so a CPU-heavy processor
competes with the trainer's main thread for the GIL. The node in this module
runs a three-stage background pipeline: a pull thread reads parent rows and
submits them to a pool of worker processes, and a collect thread resolves the
futures in pull order. The trainer's main thread only pops rows that are
already fully unpickled, so neither the submit pickle nor the (much larger)
result unpickle serializes into the training step.

Results are yielded in pull order and checkpoint state excludes in-flight
rows, so restore re-processes exactly the rows that had not been emitted.
"""

import multiprocessing
import os
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
    # NCCL pins the trainer thread to the GPU-local NUMA node when a
    # communicator initializes, and spawned workers inherit that mask. Let
    # workers float across all cores instead.
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    _WORKER_PROCESSOR = processor_config.build(context=context)


def _worker_apply(row: Any, seed: int, counter: int) -> Any:
    if _WORKER_PROCESSOR is None:
        raise RuntimeError("process-pool worker was not initialized")
    rng = np.random.Generator(np.random.Philox(key=[seed, counter]))
    return _WORKER_PROCESSOR(row, rng)


class _PendingRow:
    """One submitted row: future first, resolved result (or error) later."""

    __slots__ = ("pre_pull_state", "counter", "future", "result", "error", "resolved")

    def __init__(self, pre_pull_state: Any, counter: int, future: Any) -> None:
        self.pre_pull_state = pre_pull_state
        self.counter = counter
        self.future = future
        self.result: Any = None
        self.error: BaseException | None = None
        self.resolved = False


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
        # Pipeline rows in pull order: unresolved futures first, then resolved
        # results waiting for the main thread to pop them.
        self._pending: deque[_PendingRow] = deque()
        self._pull_counter = 0
        # (parent state before the pull, pull counter) of the row the pull
        # thread currently holds between reading the cursor and appending to
        # _pending; that window must stay checkpointable.
        self._pull_position: tuple[Any, int] | None = None
        self._pull_done = False
        self._pull_error: BaseException | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._resume_state: Any = {
            "parent": self._parent.get_state(),
            "counter": 0,
        }
        self._threads: list[threading.Thread] = []
        self._start_threads()

    def _pull_loop(self) -> None:
        """Read parent rows and submit them to the pool ahead of consumption."""
        try:
            while not self._stop.is_set():
                with self._cond:
                    while (
                        len(self._pending) >= self._max_in_flight
                        and not self._stop.is_set()
                    ):
                        self._cond.wait(timeout=0.1)
                    if self._stop.is_set():
                        return
                    pre_pull_state = self._parent.get_state()
                    counter = self._pull_counter
                    self._pull_position = (pre_pull_state, counter)
                try:
                    row = next(self._parent)
                except StopIteration:
                    with self._cond:
                        self._pull_done = True
                        self._cond.notify_all()
                    return
                future = self._executor.submit(_worker_apply, row, self._seed, counter)
                with self._cond:
                    self._pending.append(_PendingRow(pre_pull_state, counter, future))
                    self._pull_counter = counter + 1
                    self._cond.notify_all()
        except BaseException as exc:
            with self._cond:
                if self._pull_error is None:
                    self._pull_error = exc
                self._pull_done = True
                self._cond.notify_all()

    def _collect_loop(self) -> None:
        """Resolve futures strictly in pull order, off the main thread.

        The result unpickle (multi-MB tensors per row) is the largest chunk of
        GIL work in the pipeline; doing it here keeps it out of the trainer's
        critical path.
        """
        try:
            while not self._stop.is_set():
                with self._cond:
                    while True:
                        if self._stop.is_set():
                            return
                        target = next(
                            (row for row in self._pending if not row.resolved), None
                        )
                        if target is not None:
                            break
                        if self._pull_done:
                            return
                        self._cond.wait(timeout=0.1)
                try:
                    result = target.future.result()
                    error = None
                except BaseException as exc:
                    result = None
                    error = exc
                with self._cond:
                    target.result = result
                    target.error = error
                    target.future = None
                    target.resolved = True
                    self._cond.notify_all()
        except BaseException as exc:
            with self._cond:
                if self._pull_error is None:
                    self._pull_error = exc
                self._pull_done = True
                self._cond.notify_all()

    def _start_threads(self) -> None:
        self._threads = [
            threading.Thread(
                target=self._run_unpinned,
                args=(self._pull_loop,),
                name="process-pool-pull",
                daemon=True,
            ),
            threading.Thread(
                target=self._run_unpinned,
                args=(self._collect_loop,),
                name="process-pool-collect",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    @staticmethod
    def _run_unpinned(loop: Callable[[], None]) -> None:
        # NCCL pins the trainer thread that initializes a communicator to the
        # GPU-local NUMA node; threads spawned afterwards inherit that mask.
        # The pull thread does tar reads and the collect thread unpickles
        # rows, so let both float across all cores.
        os.sched_setaffinity(0, range(os.cpu_count() or 1))
        loop()

    def _stop_threads(self) -> None:
        self._stop.set()
        # Cancel queued-but-not-started futures so a collector blocked in
        # result() wakes promptly; running futures still finish on their own.
        for row in self._pending:
            if row.future is not None:
                row.future.cancel()
        with self._cond:
            self._cond.notify_all()
        for thread in self._threads:
            thread.join()
        self._threads = []
        self._pending.clear()
        self._pull_position = None
        self._pull_done = False
        self._pull_error = None
        self._stop.clear()

    def __next__(self) -> Any:
        with self._stats.record_self_time():
            with self._cond:
                while True:
                    if self._pending and self._pending[0].resolved:
                        row = self._pending.popleft()
                        self._cond.notify()
                        break
                    if self._pull_done and not self._pending:
                        if self._pull_error is not None:
                            raise self._pull_error
                        raise StopIteration
                    self._cond.wait(timeout=0.1)
                # Resume state points at the first row not yet emitted: the
                # oldest pending row, else the row the pull thread is holding,
                # else the current parent cursor.
                if self._pending:
                    first = self._pending[0]
                    self._resume_state = {
                        "parent": first.pre_pull_state,
                        "counter": first.counter,
                    }
                elif self._pull_position is not None:
                    state, counter = self._pull_position
                    self._resume_state = {"parent": state, "counter": counter}
                else:
                    self._resume_state = {
                        "parent": self._parent.get_state(),
                        "counter": self._pull_counter,
                    }
        if row.error is not None:
            raise row.error
        return self._stats.record_output_spec(row.result)

    def get_state(self) -> dict[str, Any]:
        return self._resume_state

    def set_state(self, state: dict[str, Any]) -> None:
        # In-flight rows were pulled from the old position; _stop_threads
        # cancels the queued ones and stale running results drain unread.
        self._stop_threads()
        self._parent.set_state(state["parent"])
        self._pull_counter = state["counter"]
        self._resume_state = state
        self._start_threads()

    def close(self) -> None:
        self._stop_threads()
        self._executor.shutdown(wait=False, cancel_futures=True)
