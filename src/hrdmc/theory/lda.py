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


def hard_rod_lda_density_from_local_mu(
    local_mu: float | FloatArray,
    rod_length: float,
) -> float | FloatArray:
    """Invert the local hard-rod LDA equation in repository energy units.

    The trapped LDA equation solved by this module is

        local_mu = mu_HR(n)

    where `local_mu = mu_0 - V(x)` and `mu_HR` is the homogeneous hard-rod
    chemical potential in the repository convention hbar^2/(2m)=1. This helper
    owns the local inversion used by the production LDA profile; it intentionally
    preserves the existing monotone bisection implementation.
    """

    scalar_input = np.isscalar(local_mu)
    values = np.atleast_1d(np.asarray(local_mu, dtype=float))
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if np.any(values < 0.0):
        raise ValueError("local_mu must be non-negative")
    densities = np.zeros_like(values, dtype=float)
    positive = values > 0.0
    densities[positive] = [
        invert_hard_rod_chemical_potential(float(mu), rod_length)
        for mu in values[positive]
    ]
    if scalar_input:
        return float(densities[0])
    return densities


def hard_rod_lda_density_from_local_mu_cubic(
    local_mu: float | FloatArray,
    rod_length: float,
) -> float | FloatArray:
    """Solve the same local LDA equation through its explicit cubic form.

    For `a = rod_length > 0`, set

        y = a n / (1 - a n).

    Then the hard-rod LDA equation becomes

        2 y^3 + 3 y^2 = 3 a^2 local_mu / pi^2.

    The physical density is recovered as `n = y / (a * (1 + y))`. This helper is
    kept as an analytic cross-check of the production bisection inversion, not
    as a replacement for it.
    """

    scalar_input = np.isscalar(local_mu)
    values = np.atleast_1d(np.asarray(local_mu, dtype=float))
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if np.any(values < 0.0):
        raise ValueError("local_mu must be non-negative")
    if rod_length == 0.0:
        densities = np.sqrt(values) / np.pi
        return float(densities[0]) if scalar_input else densities

    densities = np.zeros_like(values, dtype=float)
    positive_indices = np.flatnonzero(values > 0.0)
    for index in positive_indices:
        scaled_mu = 3.0 * rod_length * rod_length * float(values[index]) / (np.pi**2)
        roots = np.roots([2.0, 3.0, 0.0, -scaled_mu])
        real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-10)].real
        candidates = real_roots[real_roots >= 0.0]
        if candidates.size == 0:
            raise RuntimeError("failed to find physical hard-rod LDA cubic root")
        y = float(np.min(candidates))
        densities[index] = y / (rod_length * (1.0 + y))
    if scalar_input:
        return float(densities[0])
    return densities


def hard_rod_lda_density_small_a_expansion(
    local_mu: float | FloatArray,
    rod_length: float,
) -> float | FloatArray:
    """Return the fixed-chemical-potential small-`a` LDA density expansion.

    In repository energy units,

        n(local_mu, a) = sqrt(local_mu)/pi
            - 4 a local_mu / (3 pi^2) + O(a^2).

    In harmonic-oscillator energy units the same formula is obtained by passing
    `local_mu_code = 2 * local_mu_ho`. The helper is a diagnostic expression for
    the analytic small-rod limit; production profiles continue to use the exact
    local inversion above.
    """

    scalar_input = np.isscalar(local_mu)
    values = np.atleast_1d(np.asarray(local_mu, dtype=float))
    if rod_length < 0.0:
        raise ValueError("rod_length must be non-negative")
    if np.any(values < 0.0):
        raise ValueError("local_mu must be non-negative")
    densities = np.sqrt(values) / np.pi - (4.0 * rod_length * values) / (3.0 * np.pi**2)
    densities = np.maximum(densities, 0.0)
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
        return np.asarray(hard_rod_lda_density_from_local_mu(local_mu, rod_length), dtype=float)

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
