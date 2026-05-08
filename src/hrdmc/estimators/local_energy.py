from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalEnergyResult:
    values: FloatArray
    mean: float
    stderr: float


def estimate_local_energy(
    snapshots: FloatArray,
    trial: HardRodJastrowTrial,
) -> LocalEnergyResult:
    """Estimate the local kinetic energy from sampled coordinates."""

    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_snapshots, n_particles)")
    if snapshots.shape[1] != trial.system.n_particles:
        raise ValueError("snapshot particle count does not match trial system")
    if snapshots.shape[0] == 0:
        raise ValueError("at least one snapshot is required")

    values = np.asarray([trial.local_kinetic_energy(x) for x in snapshots], dtype=float)
    mean = float(np.mean(values))
    if values.size == 1:
        stderr = 0.0
    else:
        stderr = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return LocalEnergyResult(values=values, mean=mean, stderr=stderr)
