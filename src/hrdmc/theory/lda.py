from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _validate_density(density: float, rod_length: float) -> None:
    if density < 0:
        raise ValueError("density must be non-negative")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    if density * rod_length >= 1.0:
        raise ValueError("packing fraction density * rod_length must be < 1")


def excluded_length(n_particles: int, length: float, rod_length: float) -> float:
    """Return the free ring length after removing the hard-rod excluded volume."""
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    reduced_length = length - n_particles * rod_length
    if reduced_length <= 0:
        raise ValueError("excluded length N * a must be smaller than L")
    return float(reduced_length)


def hard_rod_finite_ring_energy_per_particle(
    n_particles: int,
    length: float,
    rod_length: float,
) -> float:
    """Finite-N homogeneous ring benchmark energy in units hbar^2/(m)=1."""
    if n_particles < 2:
        raise ValueError("n_particles must be at least 2")
    free_length = excluded_length(n_particles, length, rod_length)
    quantum_numbers = np.arange(n_particles, dtype=float) - (n_particles - 1) / 2.0
    k = 2.0 * np.pi * quantum_numbers / free_length
    return float(0.5 * np.sum(k**2) / n_particles)


def hard_rod_energy_per_particle(density: float, rod_length: float) -> float:
    """Thermodynamic homogeneous hard-rod EOS in units hbar^2/(m)=1."""
    _validate_density(density, rod_length)
    if density == 0.0:
        return 0.0
    return float(np.pi**2 * density**2 / (6.0 * (1.0 - density * rod_length) ** 2))


def hard_rod_energy_density(density: float, rod_length: float) -> float:
    """Homogeneous energy density epsilon(rho)=rho e(rho)."""
    return float(density * hard_rod_energy_per_particle(density, rod_length))


def hard_rod_chemical_potential(density: float, rod_length: float) -> float:
    """Chemical potential d epsilon_HR / d rho for the homogeneous hard-rod EOS."""
    _validate_density(density, rod_length)
    if density == 0.0:
        return 0.0
    numerator = np.pi**2 * density**2 * (3.0 - rod_length * density)
    denominator = 6.0 * (1.0 - rod_length * density) ** 3
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
        return float(np.sqrt(2.0 * chemical_potential) / np.pi)
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


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LDADensityProfile:
    x: FloatArray
    n_x: FloatArray
    potential_x: FloatArray
    chemical_potential: float
    target_particles: float
    integrated_particles: float


def _hard_rod_lda_density_from_local_mu(
    local_mu: float | FloatArray,
    rod_length: float,
) -> float | FloatArray:
    """Invert the local hard-rod LDA equation in harmonic-oscillator units."""
    scalar_input = np.isscalar(local_mu)
    values = np.atleast_1d(np.asarray(local_mu, dtype=float))
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if np.any(values < 0.0):
        raise ValueError("local_mu must be non-negative")
    densities = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    densities[positive] = [
        invert_hard_rod_chemical_potential(float(mu), rod_length) for mu in values[positive]
    ]
    if scalar_input:
        return float(densities[0])
    return densities


def _integrate(x: FloatArray, y: FloatArray) -> float:
    return float(np.trapezoid(y, x))


def _validate_grid(x: FloatArray, values: FloatArray) -> None:
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if values.shape != x.shape:
        raise ValueError("values must have the same shape as x")
    if x.size < 2:
        raise ValueError("x must contain at least two points")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")


def lda_density_profile(
    x: FloatArray,
    potential_x: FloatArray,
    n_particles: float,
    rod_length: float,
    *,
    tolerance: float = 1e-10,
    boundary_density_tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> LDADensityProfile:
    """Solve the excluded-volume LDA normalization on a fixed spatial grid."""
    x = np.asarray(x, dtype=float)
    potential_x = np.asarray(potential_x, dtype=float)
    _validate_lda_request(x, potential_x, n_particles, rod_length, boundary_density_tolerance)
    v_min = float(np.min(potential_x))

    def density_for_mu(global_mu: float) -> FloatArray:
        local_mu = np.maximum(global_mu - potential_x, 0.0)
        return np.asarray(_hard_rod_lda_density_from_local_mu(local_mu, rod_length), dtype=float)

    def build_profile(global_mu: float, n_x: FloatArray, count: float) -> LDADensityProfile:
        if max(float(n_x[0]), float(n_x[-1])) > boundary_density_tolerance:
            raise ValueError(
                "LDA grid does not contain the density cloud; increase the spatial extent"
            )
        return LDADensityProfile(
            x=x,
            n_x=n_x,
            potential_x=potential_x,
            chemical_potential=float(global_mu),
            target_particles=float(n_particles),
            integrated_particles=float(count),
        )

    low, high = _bracket_chemical_potential(
        density_for_mu, x, n_particles, v_min, float(np.max(potential_x)), max_iterations
    )
    global_mu, n_x, count = _bisect_chemical_potential(
        density_for_mu,
        x,
        n_particles,
        low,
        high,
        tolerance,
        max_iterations,
    )
    return build_profile(global_mu, n_x, count)


def _validate_lda_request(
    x: FloatArray,
    potential_x: FloatArray,
    n_particles: float,
    rod_length: float,
    boundary_density_tolerance: float,
) -> None:
    _validate_grid(x, potential_x)
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    if boundary_density_tolerance < 0:
        raise ValueError("boundary_density_tolerance must be non-negative")
    if rod_length > 0 and n_particles >= (float(x[-1]) - float(x[0])) / rod_length:
        raise ValueError("grid is too small for the requested hard-rod excluded volume")


def _bracket_chemical_potential(
    density_for_mu: Callable[[float], FloatArray],
    x: FloatArray,
    n_particles: float,
    v_min: float,
    v_max: float,
    max_iterations: int,
) -> tuple[float, float]:
    high = max(v_min + 1.0, v_max + 1.0)
    for _ in range(max_iterations):
        if _integrate(x, density_for_mu(high)) >= n_particles:
            return v_min, high
        high = v_min + 2.0 * (high - v_min)
    raise RuntimeError("failed to bracket LDA chemical potential")


def _bisect_chemical_potential(
    density_for_mu: Callable[[float], FloatArray],
    x: FloatArray,
    n_particles: float,
    low: float,
    high: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[float, FloatArray, float]:
    n_mid = density_for_mu(0.5 * (low + high))
    count_mid = _integrate(x, n_mid)
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        n_mid = density_for_mu(mid)
        count_mid = _integrate(x, n_mid)
        if abs(count_mid - n_particles) <= tolerance * max(1.0, n_particles):
            return mid, n_mid, count_mid
        if count_mid < n_particles:
            low = mid
        else:
            high = mid
    mid = 0.5 * (low + high)
    n_mid = density_for_mu(mid)
    return mid, n_mid, _integrate(x, n_mid)


def lda_total_energy(profile: LDADensityProfile, rod_length: float) -> float:
    """Integrate epsilon_HR(n(x)) + V(x)n(x) over an LDA profile."""
    local_energy = np.asarray(
        [hard_rod_energy_density(float(density), rod_length) for density in profile.n_x],
        dtype=float,
    )
    potential_energy = profile.potential_x * profile.n_x
    return _integrate(profile.x, local_energy + potential_energy)


def lda_mean_square_radius(profile: LDADensityProfile, *, center: float = 0.0) -> float:
    """Return <(x-center)^2> from an LDA density profile."""
    if profile.integrated_particles <= 0:
        raise ValueError("profile must contain a positive particle count")
    moment = _integrate(profile.x, ((profile.x - center) ** 2) * profile.n_x)
    return float(moment / profile.integrated_particles)


def lda_rms_radius(profile: LDADensityProfile, *, center: float = 0.0) -> float:
    """Return sqrt(<(x-center)^2>) from an LDA density profile."""
    return float(np.sqrt(lda_mean_square_radius(profile, center=center)))


def lda_support_edges(
    profile: LDADensityProfile,
    *,
    density_threshold: float = 1e-8,
) -> tuple[float | None, float | None]:
    """Return the first and last LDA grid points above a density threshold."""
    if density_threshold < 0:
        raise ValueError("density_threshold must be non-negative")
    occupied = np.flatnonzero(profile.n_x > density_threshold)
    if occupied.size == 0:
        return None, None
    return float(profile.x[occupied[0]]), float(profile.x[occupied[-1]])
