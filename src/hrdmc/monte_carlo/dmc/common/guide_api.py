from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from hrdmc.wavefunctions.api import DMCGuide

FloatArray = NDArray[np.float64]
GuideBatchLog = Callable[[FloatArray], tuple[FloatArray, NDArray[np.bool_]]]
GuideBatchGradLapLocal = Callable[
    [FloatArray],
    tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]],
]
GuideValidBatch = Callable[[FloatArray], NDArray[np.bool_]]


class ValidatingSystem(Protocol):
    def is_valid(self, positions: FloatArray) -> bool: ...


def evaluate_guide(
    guide: DMCGuide,
    positions: FloatArray,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    batch = getattr(cast(Any, guide), "batch_grad_lap_local", None)
    if callable(batch):
        batch_fn = cast(GuideBatchGradLapLocal, batch)
        _grad, _lap, local, finite = batch_fn(positions)
        return np.asarray(local, dtype=float), np.asarray(finite, dtype=bool)

    energies = np.empty(positions.shape[0], dtype=float)
    valid = np.empty(positions.shape[0], dtype=bool)
    for idx, row in enumerate(positions):
        row_valid = bool(guide.is_valid(row))
        if row_valid:
            energy = float(guide.local_energy(row))
            row_valid = bool(np.isfinite(energy))
        else:
            energy = float("nan")
        energies[idx] = energy
        valid[idx] = row_valid
    return energies, valid


def guide_log_values(guide: DMCGuide, positions: FloatArray) -> FloatArray:
    batch = getattr(cast(Any, guide), "batch_log_value", None)
    if callable(batch):
        batch_fn = cast(GuideBatchLog, batch)
        log_values, _finite = batch_fn(positions)
        return np.asarray(log_values, dtype=float)
    return np.asarray([guide.log_value(row) for row in positions], dtype=float)


def guide_grad_energy_valid(
    guide: DMCGuide,
    positions: FloatArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.bool_]]:
    batch = getattr(cast(Any, guide), "batch_grad_lap_local", None)
    if callable(batch):
        batch_fn = cast(GuideBatchGradLapLocal, batch)
        grad, _lap, local, finite = batch_fn(positions)
        return (
            np.asarray(grad, dtype=float),
            np.asarray(local, dtype=float),
            np.asarray(finite, dtype=bool),
        )
    grad = np.vstack([guide.grad_log_value(row) for row in positions])
    local, valid = evaluate_guide(guide, positions)
    return grad, local, valid


def guide_batch_backend(guide: DMCGuide) -> str:
    backend = getattr(cast(Any, guide), "batch_backend", None)
    if isinstance(backend, str):
        return backend
    if callable(getattr(cast(Any, guide), "batch_grad_lap_local", None)):
        return "batch"
    return "scalar"


def valid_rows(
    system: ValidatingSystem,
    guide: DMCGuide,
    positions: FloatArray,
) -> NDArray[np.bool_]:
    valid_batch = getattr(cast(Any, guide), "valid_batch", None)
    if callable(valid_batch):
        valid_batch_fn = cast(GuideValidBatch, valid_batch)
        guide_valid = np.asarray(valid_batch_fn(positions), dtype=bool)
        system_valid = np.asarray([system.is_valid(row) for row in positions], dtype=bool)
        return system_valid & guide_valid
    return np.asarray(
        [system.is_valid(row) and guide.is_valid(row) for row in positions],
        dtype=bool,
    )
