from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.theory.hard_rods import (
    hard_rod_energy_density,
    invert_hard_rod_chemical_potential,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LDADensityProfile:
    x: FloatArray
    n_x: FloatArray
    potential_x: FloatArray
    chemical_potential: float
    target_particles: float
    integrated_particles: float


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
    _validate_grid(x, potential_x)
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    if rod_length < 0:
        raise ValueError("rod_length must be non-negative")
    if boundary_density_tolerance < 0:
        raise ValueError("boundary_density_tolerance must be non-negative")
    if rod_length > 0:
        max_particles = (float(x[-1]) - float(x[0])) / rod_length
        if n_particles >= max_particles:
            raise ValueError("grid is too small for the requested hard-rod excluded volume")

    v_min = float(np.min(potential_x))

    def density_for_mu(global_mu: float) -> FloatArray:
        local_mu = np.maximum(global_mu - potential_x, 0.0)
        densities = np.zeros_like(local_mu)
        positive = local_mu > 0.0
        densities[positive] = [
            invert_hard_rod_chemical_potential(mu, rod_length) for mu in local_mu[positive]
        ]
        return densities

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

    low = v_min
    high = max(v_min + 1.0, float(np.max(potential_x)) + 1.0)
    for _ in range(max_iterations):
        if _integrate(x, density_for_mu(high)) >= n_particles:
            break
        high = v_min + 2.0 * (high - v_min)
    else:
        raise RuntimeError("failed to bracket LDA chemical potential")

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        n_mid = density_for_mu(mid)
        count_mid = _integrate(x, n_mid)
        if abs(count_mid - n_particles) <= tolerance * max(1.0, n_particles):
            return build_profile(mid, n_mid, count_mid)
        if count_mid < n_particles:
            low = mid
        else:
            high = mid

    global_mu = 0.5 * (low + high)
    n_x = density_for_mu(global_mu)
    return build_profile(global_mu, n_x, _integrate(x, n_x))


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
