from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _GapHPairGroundState:
    y_grid: FloatArray
    ground_state: FloatArray
    relative_energy: float
    rod_length: float
    dy: float


def _build_gap_h_pair_ground_state(
    *,
    rod_length: float,
    omega: float,
    grid_points: int,
    y_max: float | None,
) -> _GapHPairGroundState:
    y, dy, _eigenvalues, _basis, psi0, energy0 = _solve_pair_basis(
        rod_length=rod_length,
        omega=omega,
        grid_points=grid_points,
        y_max=y_max,
    )
    return _GapHPairGroundState(
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
    diagonal = np.full(n, 2.0 / (dy * dy), dtype=float)
    diagonal += 0.25 * omega * omega * y * y
    off_diagonal = np.full(n - 1, -1.0 / (dy * dy), dtype=float)
    return np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)


def _default_pair_y_max(rod_length: float, omega: float) -> float:
    return float(max(rod_length + 20.0, rod_length + 18.0 / math.sqrt(omega)))


def _positive_ground_state(values: FloatArray) -> FloatArray:
    psi = np.asarray(values, dtype=float)
    floor = max(float(np.max(np.abs(psi))) * 1.0e-14, np.finfo(float).tiny)
    return np.maximum(np.abs(psi), floor)

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class TrappedN2FiniteAReference:
    """Deterministic N=2 trapped hard-rod reference from the relative coordinate."""
    rod_length: float
    omega: float
    y_grid: FloatArray
    relative_ground_state: FloatArray
    relative_probability_mass: FloatArray
    relative_energy: float
    com_energy: float
    total_energy: float
    com_variance: float
    relative_gap_mean_square: float
    r2_radius: float
    rms_radius: float
    dy: float
    center: float = 0.0
    @property
    def n_particles(self) -> int:
        return 2
    def density_profile(self, x: FloatArray) -> FloatArray:
        grid = np.asarray(x, dtype=float)
        if grid.ndim != 1:
            raise ValueError("x must be one-dimensional")
        if not np.all(np.isfinite(grid)):
            raise ValueError("x must be finite")
        sigma = math.sqrt(self.com_variance)
        prefactor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
        y = self.y_grid
        mass = self.relative_probability_mass
        shifted_left = grid[:, np.newaxis] + 0.5 * y[np.newaxis, :]
        shifted_right = grid[:, np.newaxis] - 0.5 * y[np.newaxis, :]
        left = _gaussian_density(
            shifted_left,
            center=self.center,
            prefactor=prefactor,
            sigma=sigma,
        )
        right = _gaussian_density(
            shifted_right,
            center=self.center,
            prefactor=prefactor,
            sigma=sigma,
        )
        return (left + right) @ mass
    def bin_averaged_density(
        self,
        bin_edges: FloatArray,
        *,
        quadrature_points: int = 24,
    ) -> FloatArray:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("bin_edges must be a one-dimensional edge array")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("bin_edges must be finite and strictly increasing")
        if quadrature_points < 4:
            raise ValueError("quadrature_points must be at least 4")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        values = np.empty(edges.size - 1, dtype=float)
        for index, (left, right) in enumerate(itertools.pairwise(edges)):
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            x = midpoint + half_width * nodes
            integral = half_width * float(np.sum(weights * self.density_profile(x)))
            values[index] = integral / (right - left)
        return values
    def to_metadata(self) -> dict[str, float | int]:
        return {
            "n_particles": 2,
            "rod_length": self.rod_length,
            "omega": self.omega,
            "relative_energy": self.relative_energy,
            "com_energy": self.com_energy,
            "total_energy": self.total_energy,
            "com_variance": self.com_variance,
            "relative_gap_mean_square": self.relative_gap_mean_square,
            "r2_radius": self.r2_radius,
            "rms_radius": self.rms_radius,
            "relative_grid_points": int(self.y_grid.size),
            "relative_y_min": float(self.y_grid[0]),
            "relative_y_max": float(self.y_grid[-1]),
            "relative_dy": self.dy,
            "center": self.center,
        }
def trapped_n2_finite_a_reference(
    *,
    rod_length: float,
    omega: float,
    grid_points: int = 1400,
    y_max: float | None = None,
    center: float = 0.0,
) -> TrappedN2FiniteAReference:
    """Build the trapped N=2 finite-a reference [GirardeauAstrakharchik2010]."""
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    ground = _build_gap_h_pair_ground_state(
        rod_length=rod_length,
        omega=omega,
        grid_points=grid_points,
        y_max=y_max,
    )
    psi = np.asarray(ground.ground_state, dtype=float)
    mass = psi * psi * ground.dy
    mass = mass / float(np.sum(mass))
    gap_mean_square = float(np.sum(ground.y_grid * ground.y_grid * mass))
    com_energy = float(0.5 * omega)
    com_variance = float(1.0 / (4.0 * omega))
    total_energy = float(ground.relative_energy + com_energy)
    r2_radius = float(center * center + com_variance + 0.25 * gap_mean_square)
    return TrappedN2FiniteAReference(
        rod_length=float(rod_length),
        omega=float(omega),
        y_grid=np.asarray(ground.y_grid, dtype=float),
        relative_ground_state=psi,
        relative_probability_mass=mass,
        relative_energy=float(ground.relative_energy),
        com_energy=com_energy,
        total_energy=total_energy,
        com_variance=com_variance,
        relative_gap_mean_square=gap_mean_square,
        r2_radius=r2_radius,
        rms_radius=float(math.sqrt(r2_radius)),
        dy=float(ground.dy),
        center=float(center),
    )
def _gaussian_density(
    x: FloatArray,
    *,
    center: float,
    prefactor: float,
    sigma: float,
) -> FloatArray:
    z = (x - center) / sigma
    return prefactor * np.exp(-0.5 * z * z)
