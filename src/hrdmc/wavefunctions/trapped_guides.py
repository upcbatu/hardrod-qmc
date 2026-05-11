from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.trapped_guide_kernels import (
    backend_name,
    reduced_tg_grad_lap_local_batch,
    reduced_tg_log_batch,
    valid_batch,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ReducedTGHardRodGuide:
    """DMC guide for trapped hard rods in reduced coordinates.

    The guide uses the ordered-sector hard-rod mapping

        y_i = x_i - a * (i - (N-1)/2)

    and the positive TG/free-fermion-like amplitude

        Psi_T = exp[-alpha/2 * sum_i (y_i-x0)^2]
                * prod_{i<j} (y_j-y_i)^pair_power.

    This class is a DMC guide, not only a VMC trial: it owns log amplitude,
    first and second log derivatives, local energy, and ordered-sector validity.
    """

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    alpha: float
    pair_power: float = 1.0

    def __post_init__(self) -> None:
        if self.trap.center != self.system.center:
            raise ValueError("system and trap centers must match")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.pair_power <= 0.0:
            raise ValueError("pair_power must be positive")

    def is_valid(self, positions: FloatArray) -> bool:
        positions = np.asarray(positions, dtype=float)
        return bool(
            positions.shape == (self.system.n_particles,)
            and np.all(np.isfinite(positions))
            and np.all(np.diff(positions) >= self.system.rod_length)
        )

    def log_value(self, positions: FloatArray) -> float:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.system.n_particles,):
            return float("-inf")
        row = positions[np.newaxis, :]
        log_values, _finite = self.batch_log_value(row)
        return float(log_values[0])

    def grad_log_value(self, positions: FloatArray) -> FloatArray:
        grad, _lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return grad[0]

    def lap_log_value(self, positions: FloatArray) -> FloatArray:
        _grad, lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return lap[0]

    def local_energy(self, positions: FloatArray) -> float:
        _grad, _lap, local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod guide configuration")
        return float(local[0])

    def reduced_positions(self, positions: FloatArray) -> FloatArray:
        positions = np.asarray(positions, dtype=float)
        return positions - self._offsets()

    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]:
        positions = self._as_position_batch(positions)
        return valid_batch(positions, self.system.rod_length)

    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
        positions = self._as_position_batch(positions)
        return reduced_tg_log_batch(
            positions,
            self._offsets(),
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            center=self.system.center,
            pair_power=self.pair_power,
        )

    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        positions = self._as_position_batch(positions)
        return reduced_tg_grad_lap_local_batch(
            positions,
            self._offsets(),
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            center=self.system.center,
            omega2=self.trap.omega**2,
            pair_power=self.pair_power,
        )

    @property
    def batch_backend(self) -> str:
        return backend_name()

    def _valid_reduced_positions_or_raise(self, positions: FloatArray) -> FloatArray:
        if not self.is_valid(positions):
            raise ValueError("invalid ordered hard-rod guide configuration")
        y = self.reduced_positions(positions)
        if np.any(_upper_pair_gaps(y) <= 0.0):
            raise ValueError("invalid reduced hard-rod contact gap")
        return y

    def _offsets(self) -> FloatArray:
        return self.system.rod_length * (
            np.arange(self.system.n_particles, dtype=float) - 0.5 * (self.system.n_particles - 1)
        )

    def _as_position_batch(self, positions: FloatArray) -> FloatArray:
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != self.system.n_particles:
            raise ValueError("positions must have shape (n_walkers, n_particles)")
        return positions


def _upper_pair_gaps(y: FloatArray) -> FloatArray:
    i, j = np.triu_indices(y.size, k=1)
    return y[j] - y[i]
