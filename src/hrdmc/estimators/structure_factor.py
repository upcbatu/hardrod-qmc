from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.hard_rods import HardRodSystem


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StructureFactorResult:
    k: FloatArray
    s_k: FloatArray
    stderr: FloatArray


def estimate_static_structure_factor(
    snapshots: FloatArray,
    system: HardRodSystem,
    n_modes: int = 24,
) -> StructureFactorResult:
    """Estimate the static structure factor S(k).

    Source equation
    ---------------
    The static structure factor S(k) is a central observable in
    [Mazzanti2008HardRods]. We estimate it from density Fourier modes:

        rho_k = sum_j exp(i*k*x_j)
        S(k)  = < rho_k rho_-k > / N = < |rho_k|^2 > / N
    """
    snapshots = np.asarray(snapshots, dtype=float)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_samples, n_particles)")
    if snapshots.shape[1] != system.n_particles:
        raise ValueError("snapshot particle dimension does not match system")

    k_values = system.allowed_k_values(n_modes)
    values = np.empty((snapshots.shape[0], n_modes), dtype=float)
    for i, x in enumerate(snapshots):
        phase = np.exp(1j * np.outer(k_values, system.wrap(x)))
        rho_k = np.sum(phase, axis=1)
        values[i] = (np.abs(rho_k) ** 2) / system.n_particles

    mean = np.mean(values, axis=0)
    if snapshots.shape[0] > 1:
        stderr = np.std(values, axis=0, ddof=1) / np.sqrt(snapshots.shape[0])
    else:
        stderr = np.zeros_like(mean)
    return StructureFactorResult(k=k_values, s_k=mean, stderr=stderr)
