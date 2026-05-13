from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.numerics.numba_backend import backend_label
from hrdmc.systems.kernels.gap_h_pair import sample_cdf_indices

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GapHPairGroundState:
    y_grid: FloatArray
    ground_state: FloatArray
    relative_energy: float
    rod_length: float
    dy: float


@dataclass(frozen=True)
class GapHTransformTable:
    y_grid: FloatArray
    ground_state: FloatArray
    relative_energy: float
    cdf: FloatArray
    log_density_table: FloatArray
    rod_length: float
    dy: float

    @property
    def sampling_backend(self) -> str:
        return backend_label("gap_h_cdf")

    def sample_gaps(self, rng: np.random.Generator, old_gaps: FloatArray) -> FloatArray:
        old = np.asarray(old_gaps, dtype=float)
        if old.ndim != 2:
            raise ValueError("old_gaps must be a two-dimensional array")
        if np.any(~np.isfinite(old)) or np.any(old < self.rod_length):
            raise ValueError("old_gaps must be finite and hard-core valid")
        row_idx = _nearest_indices(self.y_grid, old)
        uniforms = rng.random(size=old.shape)
        col_idx = sample_cdf_indices(self.cdf, row_idx, uniforms)
        return self.y_grid[col_idx]

    def log_density(self, old_gaps: FloatArray, new_gaps: FloatArray) -> FloatArray:
        old = np.asarray(old_gaps, dtype=float)
        new = np.asarray(new_gaps, dtype=float)
        if old.shape != new.shape:
            raise ValueError("old_gaps and new_gaps must have matching shapes")
        invalid = (
            np.any(~np.isfinite(old), axis=1)
            | np.any(~np.isfinite(new), axis=1)
            | np.any(old < self.rod_length, axis=1)
            | np.any(new < self.rod_length, axis=1)
            | np.any(old > self.y_grid[-1], axis=1)
            | np.any(new > self.y_grid[-1], axis=1)
        )
        row_idx = _nearest_indices(self.y_grid, old)
        col_idx = _nearest_indices(self.y_grid, new)
        values = self.log_density_table[row_idx, col_idx]
        out = np.sum(values, axis=1)
        out[invalid] = -np.inf
        return out


def build_gap_h_transform_table(
    *,
    rod_length: float,
    omega: float,
    tau: float,
    grid_points: int,
    y_max: float | None,
) -> GapHTransformTable:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    y, dy, eigenvalues, basis, psi0, energy0 = _solve_pair_basis(
        rod_length=rod_length,
        omega=omega,
        grid_points=grid_points,
        y_max=y_max,
    )
    weights = np.exp(-tau * eigenvalues)
    kernel = (basis * weights[np.newaxis, :]) @ basis.T
    h_density = kernel * psi0[np.newaxis, :]
    denom = math.exp(-tau * energy0) * psi0[:, np.newaxis]
    h_density = np.divide(
        h_density,
        denom,
        out=np.zeros_like(h_density),
        where=denom > 0.0,
    )
    h_mass = np.maximum(h_density, 0.0) * dy
    row_sums = np.sum(h_mass, axis=1)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(row_sums)):
        raise RuntimeError("invalid gap h-transform transition table")
    h_mass = h_mass / row_sums[:, np.newaxis]
    cdf = np.cumsum(h_mass, axis=1)
    cdf[:, -1] = 1.0
    log_density = np.log(np.maximum(h_mass / dy, np.finfo(float).tiny))
    return GapHTransformTable(
        y_grid=y,
        ground_state=psi0,
        relative_energy=energy0,
        cdf=cdf,
        log_density_table=log_density,
        rod_length=float(rod_length),
        dy=dy,
    )


def build_gap_h_pair_ground_state(
    *,
    rod_length: float,
    omega: float,
    grid_points: int,
    y_max: float | None,
) -> GapHPairGroundState:
    y, dy, _eigenvalues, _basis, psi0, energy0 = _solve_pair_basis(
        rod_length=rod_length,
        omega=omega,
        grid_points=grid_points,
        y_max=y_max,
    )
    return GapHPairGroundState(
        y_grid=y,
        ground_state=psi0,
        relative_energy=energy0,
        rod_length=float(rod_length),
        dy=dy,
    )


def _solve_pair_basis(
    *,
    rod_length: float,
    omega: float,
    grid_points: int,
    y_max: float | None,
) -> tuple[FloatArray, float, FloatArray, FloatArray, FloatArray, float]:
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    if grid_points < 64:
        raise ValueError("pair_grid_points must be at least 64")
    upper = _default_pair_y_max(rod_length, omega) if y_max is None else float(y_max)
    if upper <= rod_length:
        raise ValueError("pair_y_max must exceed rod_length")

    y_full = np.linspace(rod_length, upper, grid_points + 2, dtype=float)
    y = y_full[1:-1]
    dy = float(y[1] - y[0])
    hamiltonian = _relative_pair_hamiltonian(y, omega)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    basis = eigenvectors / math.sqrt(dy)
    psi0 = _positive_ground_state(basis[:, 0])
    energy0 = float(eigenvalues[0])
    return y, dy, eigenvalues, basis, psi0, energy0


def _relative_pair_hamiltonian(y: FloatArray, omega: float) -> FloatArray:
    dy = float(y[1] - y[0])
    n = y.size
    diagonal = np.full(n, 4.0 / (dy * dy), dtype=float)
    diagonal += 0.25 * omega * omega * y * y
    off_diagonal = np.full(n - 1, -2.0 / (dy * dy), dtype=float)
    return (
        np.diag(diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    )


def _default_pair_y_max(rod_length: float, omega: float) -> float:
    return float(max(rod_length + 20.0, rod_length + 18.0 / math.sqrt(omega)))


def _nearest_indices(grid: FloatArray, values: FloatArray) -> IntArray:
    arr = np.asarray(values, dtype=float)
    flat = arr.reshape(-1)
    idx = np.searchsorted(grid, flat)
    idx = np.clip(idx, 1, grid.size - 1)
    left = grid[idx - 1]
    right = grid[idx]
    nearest = np.where(np.abs(flat - left) <= np.abs(flat - right), idx - 1, idx)
    return nearest.astype(np.int64).reshape(arr.shape)


def _positive_ground_state(values: FloatArray) -> FloatArray:
    psi = np.asarray(values, dtype=float)
    floor = max(float(np.max(np.abs(psi))) * 1.0e-14, np.finfo(float).tiny)
    return np.maximum(np.abs(psi), floor)

