from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OpenLineHardRodSystem:
    """One-dimensional hard rods on an open line.

    This system owns only the non-periodic geometry and hard-core constraint.
    It does not own the trap potential, trial wavefunction, LDA, or benchmark
    comparison logic.
    """

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
