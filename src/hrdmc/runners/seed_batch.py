from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Manager
from typing import Any, TypeVar

from hrdmc.io.progress import ProgressBar, completed_futures_with_progress

SeedResult = TypeVar("SeedResult")
SeedSubmitter = Callable[
    [ProcessPoolExecutor, int, Any | None],
    Future[tuple[int, SeedResult]],
]
SeedRunner = Callable[[int], SeedResult]


def run_seed_batch(
    seeds: list[int],
    *,
    worker_count: int,
    progress: ProgressBar | None,
    submit_seed: SeedSubmitter[SeedResult],
    run_serial_seed: SeedRunner[SeedResult],
) -> tuple[list[SeedResult], int]:
    """Run independent seed jobs with ordered results and live parent progress."""
    if worker_count <= 1 or len(seeds) == 1:
        return [run_serial_seed(seed) for seed in seeds], 1

    results_by_seed: dict[int, SeedResult] = {}
    try:
        tempfile.gettempdir()
        with Manager() as manager, ProcessPoolExecutor(max_workers=worker_count) as executor:
            progress_queue = manager.Queue() if progress is not None else None
            futures = {submit_seed(executor, seed, progress_queue): seed for seed in seeds}
            for future in completed_futures_with_progress(
                futures,
                progress_queue=progress_queue,
                progress=progress,
            ):
                seed, result = future.result()
                results_by_seed[seed] = result
    except (OSError, PermissionError, EOFError):
        return [run_serial_seed(seed) for seed in seeds], 1
    return [results_by_seed[seed] for seed in seeds], worker_count
