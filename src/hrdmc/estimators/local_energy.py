from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.wavefunctions.jastrow import HardRodJastrowTrial
from hrdmc.wavefunctions.trapped import TrappedHardRodTrial

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LocalEnergyResult:
    values: FloatArray
    mean: float
    stderr: float


@dataclass(frozen=True)
class TrappedLocalEnergyComponents:
    kinetic: float
    trap: float
    total: float


@dataclass(frozen=True)
class TrappedLocalEnergyResult:
    kinetic_values: FloatArray
    trap_values: FloatArray
    total_values: FloatArray
    kinetic_mean: float
    kinetic_stderr: float
    trap_mean: float
    trap_stderr: float
    total_mean: float
    total_stderr: float


def _mean_stderr(values: FloatArray) -> tuple[float, float]:
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


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
    mean, stderr = _mean_stderr(values)
    return LocalEnergyResult(values=values, mean=mean, stderr=stderr)


def trapped_hard_rod_local_energy(
    positions: FloatArray,
    trial: TrappedHardRodTrial,
    trap: HarmonicTrap,
) -> TrappedLocalEnergyComponents:
    """Return local kinetic, trap, and total energy for a trapped hard-rod trial.

    The kinetic term uses

        -sum_i [d_i^2 log(Psi_T) + (d_i log(Psi_T))^2]

    in units hbar^2/(2m)=1. Invalid hard-rod configurations raise, because
    they should not enter a VMC estimator.
    """

    positions = np.asarray(positions, dtype=float)
    if positions.shape != (trial.system.n_particles,):
        raise ValueError("position count does not match trial system")
    if not trial.system.is_valid(positions):
        raise ValueError("invalid hard-rod configuration")

    x = trial.system.sorted_positions(positions)
    shifted = x - trial.system.center
    alpha = trial.gaussian_alpha
    power = trial.contact_power

    grad_log = -alpha * shifted
    lap_log = np.full(trial.system.n_particles, -alpha, dtype=float)

    if power != 0.0:
        free_gaps = np.diff(x) - trial.system.rod_length
        if np.any(free_gaps <= 0.0):
            raise ValueError("invalid hard-rod contact gap")

        inv_gaps = 1.0 / free_gaps
        inv_gaps_sq = inv_gaps**2
        grad_log[:-1] -= power * inv_gaps
        grad_log[1:] += power * inv_gaps
        lap_log[:-1] -= power * inv_gaps_sq
        lap_log[1:] -= power * inv_gaps_sq

    kinetic = float(-np.sum(lap_log + grad_log**2))
    trap_energy = trap.total(positions)
    return TrappedLocalEnergyComponents(
        kinetic=kinetic,
        trap=trap_energy,
        total=kinetic + trap_energy,
    )


def estimate_trapped_local_energy(
    snapshots: FloatArray,
    trial: TrappedHardRodTrial,
    trap: HarmonicTrap,
) -> TrappedLocalEnergyResult:
    if snapshots.ndim != 2:
        raise ValueError("snapshots must have shape (n_snapshots, n_particles)")
    if snapshots.shape[1] != trial.system.n_particles:
        raise ValueError("snapshot particle count does not match trial system")
    if snapshots.shape[0] == 0:
        raise ValueError("at least one snapshot is required")

    components = [trapped_hard_rod_local_energy(snapshot, trial, trap) for snapshot in snapshots]
    kinetic_values = np.asarray([component.kinetic for component in components], dtype=float)
    trap_values = np.asarray([component.trap for component in components], dtype=float)
    total_values = np.asarray([component.total for component in components], dtype=float)
    kinetic_mean, kinetic_stderr = _mean_stderr(kinetic_values)
    trap_mean, trap_stderr = _mean_stderr(trap_values)
    total_mean, total_stderr = _mean_stderr(total_values)

    return TrappedLocalEnergyResult(
        kinetic_values=kinetic_values,
        trap_values=trap_values,
        total_values=total_values,
        kinetic_mean=kinetic_mean,
        kinetic_stderr=kinetic_stderr,
        trap_mean=trap_mean,
        trap_stderr=trap_stderr,
        total_mean=total_mean,
        total_stderr=total_stderr,
    )
