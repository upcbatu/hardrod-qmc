from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HardRodSystem:
    """One-dimensional hard rods on a periodic ring.

    Physical source
    ---------------
    The hard-rod Hamiltonian and the excluded-length mapping used here are
    taken from the 1D hard-rod benchmark system studied in:

        Mazzanti, Astrakharchik, Boronat, Casulleras,
        Phys. Rev. Lett. 100, 020401 (2008). [Mazzanti et. al. 2008HardRods]

    Parameters
    ----------
    n_particles:
        Number of rods.
    length:
        Ring length L.
    rod_length:
        Hard-core rod length a. Configurations with nearest-neighbor distance
        less than a are invalid.

    Notes
    -----
    The prototype works in units hbar^2 / (2m) = 1.
    """

    n_particles: int
    length: float
    rod_length: float

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.rod_length < 0:
            raise ValueError("rod_length must be non-negative")
        if self.n_particles * self.rod_length >= self.length:
            raise ValueError("excluded length N * a must be smaller than L")

    @property
    def density(self) -> float:
        return self.n_particles / self.length

    @property
    def packing_fraction(self) -> float:
        # eta = rho * a is the natural dimensionless density for hard rods.
        # Used as the density axis in [Mazzanti et. al. 2008HardRods].
        return self.density * self.rod_length

    @property
    def unexcluded_length(self) -> float:
        # Hard rods in an ordered 1D sector map to point-like particles on the
        # reduced length L' = L - N a. This is the excluded-volume mapping used
        # for the analytic hard-rod ground-state benchmark in [Mazzanti et. al. 2008HardRods].
        return self.length - self.n_particles * self.rod_length

    def wrap(self, positions: FloatArray) -> FloatArray:
        return np.mod(np.asarray(positions, dtype=float), self.length)

    def sorted_positions(self, positions: FloatArray) -> FloatArray:
        return np.sort(self.wrap(positions))

    def nearest_neighbor_gaps(self, positions: FloatArray) -> FloatArray:
        x = self.sorted_positions(positions)
        gaps = np.empty(self.n_particles, dtype=float)
        gaps[:-1] = x[1:] - x[:-1]
        gaps[-1] = self.length - x[-1] + x[0]
        return gaps

    def is_valid(self, positions: FloatArray, atol: float = 1e-12) -> bool:
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.n_particles,):
            return False
        gaps = self.nearest_neighbor_gaps(positions)
        # Hard-rod constraint: nearest-neighbor separation must be >= rod length a.
        # This encodes the infinite hard-core interaction V_HR(r<a)=infinity.
        # Source system: [Mazzanti et. al. 2008HardRods].
        return bool(np.all(gaps + atol >= self.rod_length))

    def initial_lattice(self, jitter: float = 0.0, seed: int | None = None) -> FloatArray:
        """Return a valid near-lattice configuration."""
        rng = np.random.default_rng(seed)
        positions = np.linspace(0.0, self.length, self.n_particles, endpoint=False)
        if jitter > 0:
            max_jitter = min(jitter, 0.45 * (self.length / self.n_particles - self.rod_length))
            positions = positions + rng.uniform(-max_jitter, max_jitter, size=self.n_particles)
        positions = self.wrap(positions)
        if not self.is_valid(positions):
            raise RuntimeError("failed to create a valid initial hard-rod configuration")
        return positions

    def pair_distances_min_image(self, positions: FloatArray) -> FloatArray:
        """All unique pair distances under minimum-image convention."""
        x = self.wrap(positions)
        distances: list[float] = []
        for i in range(self.n_particles - 1):
            dx = np.abs(x[i + 1 :] - x[i])
            dx = np.minimum(dx, self.length - dx)
            distances.extend(dx.tolist())
        return np.asarray(distances, dtype=float)

    def allowed_k_values(self, n_modes: int) -> FloatArray:
        if n_modes <= 0:
            raise ValueError("n_modes must be positive")
        modes = np.arange(1, n_modes + 1, dtype=float)
        # Periodic boundary condition on a ring: k_n = 2*pi*n/L.
        # This is standard for the hard-rod ring geometry used in [Mazzanti et. al. 2008HardRods].
        return 2.0 * np.pi * modes / self.length

    def exact_energy_per_particle_finite(self) -> float:
        """Finite-N hard-rod ground-state energy per particle.

        Source equation
        ---------------
        Uses the hard-rod mapping to free-fermion-like quasi-momenta on the
        reduced unexcluded length L' = L - N a. This is the analytic hard-rod
        benchmark behind the exact finite-system energy in [Mazzanti et. al. 2008HardRods].

        In units hbar^2/(2m)=1:
            E/N = (1/N) * sum_i k_i^2,
            k_i = 2*pi*I_i / (L - N*a).
        """
        n = self.n_particles
        quantum_numbers = np.arange(n, dtype=float) - (n - 1) / 2.0
        k = 2.0 * np.pi * quantum_numbers / self.unexcluded_length
        return float(np.sum(k**2) / n)

    def exact_energy_per_particle_thermodynamic(self) -> float:
        """Thermodynamic-limit hard-rod energy per particle.

        Source equation
        ---------------
        Hard-rod equation of state from [Mazzanti et. al. 2008HardRods]. In physical units:
            E/N = (pi^2 * hbar^2 * rho^2) / (6*m*(1-rho*a)^2).
        In repo units hbar^2/(2m)=1:
            E/N = pi^2 * rho^2 / (3*(1-rho*a)^2).
        """
        rho = self.density
        a = self.rod_length
        return float(np.pi**2 * rho**2 / (3.0 * (1.0 - rho * a) ** 2))
