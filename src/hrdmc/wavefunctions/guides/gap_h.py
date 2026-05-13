from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from hrdmc.systems.external_potential import HarmonicTrap
from hrdmc.systems.gap_h_pair_table import GapHPairGroundState, build_gap_h_pair_ground_state
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.kernels.gap_h import (
    gap_h_grad_lap_local_batch,
    gap_h_guide_backend_name,
    gap_h_log_batch,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GapHCorrectedHardRodGuide:
    """Fused trapped hard-rod guide matched to the gap h-transform proposal."""

    system: OpenLineHardRodSystem
    trap: HarmonicTrap
    alpha: float
    pair_power: float = 1.0
    pair_grid_points: int = 700
    pair_y_max: float | None = None
    _offsets: FloatArray = field(init=False, repr=False)
    _y_grid: FloatArray = field(init=False, repr=False)
    _log_correction: FloatArray = field(init=False, repr=False)
    _grad_correction: FloatArray = field(init=False, repr=False)
    _lap_correction: FloatArray = field(init=False, repr=False)
    _n2_total_energy: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.system.center != self.trap.center:
            raise ValueError("system and trap centers must match")
        if self.system.n_particles < 2:
            raise ValueError("gap-h-corrected guide requires at least two particles")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.pair_power != 1.0:
            raise ValueError("gap-h-corrected guide currently requires pair_power=1")
        table = build_gap_h_pair_ground_state(
            rod_length=self.system.rod_length,
            omega=self.trap.omega,
            grid_points=self.pair_grid_points,
            y_max=self.pair_y_max,
        )
        correction = _nearest_gap_correction(table, alpha=self.alpha, omega=self.trap.omega)
        object.__setattr__(self, "_offsets", _hard_rod_offsets(self.system))
        object.__setattr__(self, "_y_grid", correction.y_grid)
        object.__setattr__(self, "_log_correction", correction.log_correction)
        object.__setattr__(self, "_grad_correction", correction.grad_correction)
        object.__setattr__(self, "_lap_correction", correction.lap_correction)
        object.__setattr__(self, "_n2_total_energy", correction.n2_total_energy)

    def is_valid(self, positions: FloatArray) -> bool:
        row = np.asarray(positions, dtype=float)
        return bool(
            row.shape == (self.system.n_particles,)
            and np.all(np.isfinite(row))
            and np.all(np.diff(row) > self.system.rod_length)
            and np.all(np.diff(row) <= self._y_grid[-1])
        )

    def log_value(self, positions: FloatArray) -> float:
        values, finite = self.batch_log_value(np.asarray(positions, dtype=float)[np.newaxis, :])
        return float(values[0]) if finite[0] else float("-inf")

    def grad_log_value(self, positions: FloatArray) -> FloatArray:
        grad, _lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod gap-h guide configuration")
        return grad[0]

    def lap_log_value(self, positions: FloatArray) -> FloatArray:
        _grad, lap, _local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod gap-h guide configuration")
        return lap[0]

    def local_energy(self, positions: FloatArray) -> float:
        _grad, _lap, local, finite = self.batch_grad_lap_local(
            np.asarray(positions, dtype=float)[np.newaxis, :]
        )
        if not finite[0]:
            raise ValueError("invalid ordered hard-rod gap-h guide configuration")
        return float(local[0])

    def valid_batch(self, positions: FloatArray) -> NDArray[np.bool_]:
        _values, finite = self.batch_log_value(positions)
        return finite

    def batch_log_value(self, positions: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
        return gap_h_log_batch(
            self._as_position_batch(positions),
            self._offsets,
            self._y_grid,
            self._log_correction,
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            center=self.system.center,
            pair_power=self.pair_power,
        )

    def batch_grad_lap_local(
        self,
        positions: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        return gap_h_grad_lap_local_batch(
            self._as_position_batch(positions),
            self._offsets,
            self._y_grid,
            self._grad_correction,
            self._lap_correction,
            rod_length=self.system.rod_length,
            alpha=self.alpha,
            center=self.system.center,
            omega2=self.trap.omega**2,
            pair_power=self.pair_power,
            n2_total_energy=self._n2_total_energy,
        )

    @property
    def batch_backend(self) -> str:
        return f"gap_h_fused_{gap_h_guide_backend_name()}"

    def _as_position_batch(self, positions: FloatArray) -> FloatArray:
        x = np.asarray(positions, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.system.n_particles:
            raise ValueError("positions must have shape (n_walkers, n_particles)")
        return x


@dataclass(frozen=True)
class _NearestGapCorrection:
    y_grid: FloatArray
    log_correction: FloatArray
    grad_correction: FloatArray
    lap_correction: FloatArray
    n2_total_energy: float


def _nearest_gap_correction(
    table: GapHPairGroundState,
    *,
    alpha: float,
    omega: float,
) -> _NearestGapCorrection:
    y = np.asarray(table.y_grid, dtype=float)
    free_gap = y - float(table.rod_length)
    if np.any(free_gap <= 0.0):
        raise ValueError("gap correction grid must be strictly inside the hard-core domain")
    ground_state = np.asarray(table.ground_state, dtype=float)
    floor = max(float(np.max(np.abs(ground_state))) * 1.0e-14, np.finfo(float).tiny)
    log_h2 = np.log(np.maximum(ground_state, floor))
    log_tg = np.log(free_gap) - 0.25 * float(alpha) * free_gap * free_gap
    log_correction = log_h2 - log_tg
    grad_correction = np.gradient(log_correction, y, edge_order=2)
    lap_correction = np.gradient(grad_correction, y, edge_order=2)
    return _NearestGapCorrection(
        y_grid=np.asarray(y, dtype=float),
        log_correction=np.asarray(log_correction, dtype=float),
        grad_correction=np.asarray(grad_correction, dtype=float),
        lap_correction=np.asarray(lap_correction, dtype=float),
        n2_total_energy=float(table.relative_energy + omega / math.sqrt(2.0)),
    )


def _hard_rod_offsets(system: OpenLineHardRodSystem) -> FloatArray:
    return system.rod_length * (
        np.arange(system.n_particles, dtype=float) - 0.5 * (system.n_particles - 1)
    )
