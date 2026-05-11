from __future__ import annotations

import os
import queue
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import contextmanager
from typing import Any, Protocol

PROGRESS_FLUSH_STEPS = 1000


class ProgressBar(Protocol):
    def update(self, n: float | None = 1) -> object: ...


class NullProgress:
    def update(self, n: float | None = 1) -> object:
        del n
        return None


class QueuedProgress:
    """Batch worker-side progress updates before sending them to a parent bar."""

    def __init__(self, progress_queue: Any, *, flush_steps: int = PROGRESS_FLUSH_STEPS) -> None:
        self._queue = progress_queue
        self._flush_steps = flush_steps
        self._pending = 0

    def update(self, n: float | None = 1) -> object:
        if n is None:
            return None
        self._pending += int(n)
        if self._pending >= self._flush_steps:
            self.flush()
        return None

    def flush(self) -> None:
        if self._pending:
            self._queue.put(self._pending)
            self._pending = 0


def completed_futures_with_progress(
    futures: dict[Future[Any], int],
    *,
    progress_queue: Any | None,
    progress: ProgressBar | None,
):
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        drain_progress_queue(progress_queue, progress)
        yield from done
    drain_progress_queue(progress_queue, progress)


def drain_progress_queue(progress_queue: Any | None, progress: ProgressBar | None) -> None:
    if progress_queue is None or progress is None:
        return
    while True:
        try:
            progress.update(progress_queue.get_nowait())
        except queue.Empty:
            return


def progress_requested(flag: bool = False) -> bool:
    value = os.environ.get("DMC_PROGRESS", "")
    return flag or value.lower() in {"1", "true", "yes", "on"}


@contextmanager
def progress_bar(
    *,
    total: int,
    label: str,
    enabled: bool,
) -> Iterator[ProgressBar]:
    if not enabled:
        yield NullProgress()
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield NullProgress()
        return

    with tqdm(total=total, desc=label, unit="step") as bar:
        yield bar
