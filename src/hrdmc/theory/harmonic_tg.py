from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def trapped_tg_energy_total(n_particles: int, omega: float) -> float:
    """Exact trapped Tonks-Girardeau energy in units hbar^2/(2m)=1."""

    _validate_trapped_tg_inputs(n_particles, omega)
    return float(n_particles * n_particles * omega / np.sqrt(2.0))


def trapped_tg_r2_radius(n_particles: int, omega: float) -> float:
    """Exact per-particle <x^2> for zero-length hard rods in a harmonic trap."""

    _validate_trapped_tg_inputs(n_particles, omega)
    return float(n_particles / (np.sqrt(2.0) * omega))


def trapped_tg_rms_radius(n_particles: int, omega: float) -> float:
    """Exact RMS radius sqrt(<x^2>) for the trapped TG anchor."""

    return float(np.sqrt(trapped_tg_r2_radius(n_particles, omega)))


def trapped_tg_density_profile(
    x: FloatArray,
    *,
    n_particles: int,
    omega: float,
) -> FloatArray:
    """Exact one-body density of the trapped TG gas from occupied oscillator orbitals."""

    _validate_trapped_tg_inputs(n_particles, omega)
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(grid)):
        raise ValueError("x must be finite")

    q = omega / np.sqrt(2.0)
    z = np.sqrt(q) * grid
    phi0 = (q / np.pi) ** 0.25 * np.exp(-0.5 * z * z)
    density = phi0 * phi0
    if n_particles == 1:
        return density

    phi_nm1 = phi0
    phi_n = np.sqrt(2.0) * z * phi0
    density = density + phi_n * phi_n
    for n in range(1, n_particles - 1):
        phi_np1 = np.sqrt(2.0 / (n + 1.0)) * z * phi_n
        phi_np1 -= np.sqrt(n / (n + 1.0)) * phi_nm1
        density = density + phi_np1 * phi_np1
        phi_nm1, phi_n = phi_n, phi_np1
    return density


def trapped_tg_density_profile_semiclassical(
    x: FloatArray,
    *,
    n_particles: int,
    omega: float,
) -> FloatArray:
    """Large-N trapped TG density from the semiclassical Fermi sea."""

    _validate_trapped_tg_inputs(n_particles, omega)
    grid = np.asarray(x, dtype=float)
    if grid.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.all(np.isfinite(grid)):
        raise ValueError("x must be finite")

    q = omega / np.sqrt(2.0)
    chemical_potential = 2.0 * n_particles * q
    support = chemical_potential - q * q * grid * grid
    return np.sqrt(np.maximum(support, 0.0)) / np.pi


def _validate_trapped_tg_inputs(n_particles: int, omega: float) -> None:
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
