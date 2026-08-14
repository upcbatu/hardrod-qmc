from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
def lattice_spacing_for_target_rms(n_particles: int, target_rms: float) -> float:
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2")
    if target_rms <= 0.0:
        raise ValueError("target_rms must be positive")
    return float(target_rms * np.sqrt(12.0 / (n_particles * n_particles - 1.0)))
@dataclass(frozen=True)
class OpenLineHardRodSystem:
    """One-dimensional hard rods on an open line."""
    n_particles: int
    rod_length: float
    center: float = 0.0
    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.rod_length < 0:
            raise ValueError("rod_length must be non-negative")
    def sorted_positions(self, positions: FloatArray) -> FloatArray:
        return np.sort(np.asarray(positions, dtype=float))
    def nearest_neighbor_gaps(self, positions: FloatArray) -> FloatArray:
        x = self.sorted_positions(positions)
        return np.diff(x)
    def is_valid(self, positions: FloatArray, atol: float = 1e-12) -> bool:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.n_particles,):
            return False
        gaps = self.nearest_neighbor_gaps(positions)
        return bool(np.all(gaps + atol >= self.rod_length))
    def initial_lattice(
        self,
        jitter: float = 0.0,
        seed: int | None = None,
        spacing: float | None = None,
    ) -> FloatArray:
        if spacing is None:
            spacing = max(1.0, 2.0 * self.rod_length)
        return self.initial_lattice_with_spacing(spacing=spacing, jitter=jitter, seed=seed)
    def initial_lattice_with_spacing(
        self,
        spacing: float,
        jitter: float = 0.0,
        seed: int | None = None,
    ) -> FloatArray:
        if spacing <= self.rod_length:
            raise ValueError("spacing must be larger than rod_length")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")
        offsets = np.arange(self.n_particles, dtype=float) - 0.5 * (self.n_particles - 1)
        positions = self.center + spacing * offsets
        if jitter > 0.0:
            rng = np.random.default_rng(seed)
            max_jitter = min(jitter, 0.45 * (spacing - self.rod_length))
            positions = positions + rng.uniform(-max_jitter, max_jitter, size=self.n_particles)
        if not self.is_valid(positions):
            raise RuntimeError("failed to create a valid open-line hard-rod configuration")
        return positions
    def propose_single_particle(
        self,
        positions: FloatArray,
        particle_index: int,
        displacement: float,
    ) -> FloatArray:
        proposal = np.asarray(positions, dtype=float).copy()
        proposal[particle_index] = proposal[particle_index] + displacement
        return proposal
BASE_TRAP_QUADRATIC_COUPLING = 0.5
def lambda_from_relative_offset(
    relative_offset: float,
    *,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
) -> float:
    """Map a fractional trap-coupling change to its oscillator-unit coefficient."""
    if not np.isfinite(relative_offset):
        raise ValueError("relative_offset must be finite")
    if not np.isclose(lambda0, BASE_TRAP_QUADRATIC_COUPLING, rtol=0.0, atol=1e-15):
        raise ValueError("lambda0 must equal the oscillator-unit coupling 0.5")
    value = float(lambda0 * (1.0 + relative_offset))
    if value <= 0.0:
        raise ValueError("relative_offset must keep lambda positive")
    return value
@dataclass(frozen=True)
class HarmonicTrap:
    """Harmonic trap V(x)=0.5*omega^2*(x-center)^2."""
    omega: float
    center: float = 0.0
    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive")
    def values(self, positions: FloatArray) -> FloatArray:
        x = np.asarray(positions, dtype=float)
        return 0.5 * self.omega**2 * (x - self.center) ** 2
    def total(self, positions: FloatArray) -> float:
        return float(np.sum(self.values(positions)))
def harmonic_com_ground_variance(n_particles: int, omega: float) -> float:
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    return 1.0 / (2.0 * n_particles * omega)
