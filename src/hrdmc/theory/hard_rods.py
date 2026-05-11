from __future__ import annotations

import numpy as np

from hrdmc.systems.reduced import excluded_length


def _validate_density(density: float, rod_length: float) -> None:
    if density < 0:
        raise ValueError("density must be non-negative")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    if density * rod_length >= 1.0:
        raise ValueError("packing fraction density * rod_length must be < 1")


def hard_rod_finite_ring_energy_per_particle(
    n_particles: int,
    length: float,
    rod_length: float,
) -> float:
    """Finite-N homogeneous ring benchmark energy in units hbar^2/(2m)=1."""
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2")
    free_length = excluded_length(n_particles, length, rod_length)
    quantum_numbers = np.arange(n_particles, dtype=float) - (n_particles - 1) / 2.0
    k = 2.0 * np.pi * quantum_numbers / free_length
    return float(np.sum(k**2) / n_particles)


def hard_rod_energy_per_particle(density: float, rod_length: float) -> float:
    """Thermodynamic homogeneous hard-rod EOS in units hbar^2/(2m)=1."""
    _validate_density(density, rod_length)
    if density == 0.0:
        return 0.0
    return float(np.pi**2 * density**2 / (3.0 * (1.0 - density * rod_length) ** 2))


def hard_rod_energy_density(density: float, rod_length: float) -> float:
    """Homogeneous energy density epsilon(rho)=rho e(rho)."""
    return float(density * hard_rod_energy_per_particle(density, rod_length))


def hard_rod_chemical_potential(density: float, rod_length: float) -> float:
    """Chemical potential d epsilon_HR / d rho for the homogeneous hard-rod EOS."""
    _validate_density(density, rod_length)
    if density == 0.0:
        return 0.0
    numerator = np.pi**2 * density**2 * (3.0 - rod_length * density)
    denominator = 3.0 * (1.0 - rod_length * density) ** 3
    return float(numerator / denominator)


def invert_hard_rod_chemical_potential(
    chemical_potential: float,
    rod_length: float,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Invert mu_HR(rho) for rho by monotone bisection."""
    if chemical_potential < 0:
        raise ValueError("chemical_potential must be non-negative")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if chemical_potential == 0.0:
        return 0.0
    if rod_length == 0.0:
        return float(np.sqrt(chemical_potential) / np.pi)

    low = 0.0
    high = (1.0 / rod_length) * (1.0 - 1e-14)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        mu_mid = hard_rod_chemical_potential(mid, rod_length)
        if abs(mu_mid - chemical_potential) <= tolerance * max(1.0, chemical_potential):
            return float(mid)
        if mu_mid < chemical_potential:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))
