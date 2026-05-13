from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sample_harmonic_com_h_transform(
    rng: np.random.Generator,
    q_old: FloatArray,
    *,
    n_particles: int,
    omega: float,
    tau: float,
) -> FloatArray:
    rho, variance = harmonic_com_h_transform_moments(n_particles, omega, tau)
    return rho * q_old + math.sqrt(variance) * rng.normal(size=q_old.shape)


def harmonic_com_h_transform_log_density(
    q_old: FloatArray,
    q_new: FloatArray,
    *,
    n_particles: int,
    omega: float,
    tau: float,
) -> FloatArray:
    rho, variance = harmonic_com_h_transform_moments(n_particles, omega, tau)
    return -0.5 * (math.log(2.0 * math.pi * variance) + ((q_new - rho * q_old) ** 2) / variance)


def harmonic_com_h_transform_moments(
    n_particles: int,
    omega: float,
    tau: float,
) -> tuple[float, float]:
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    gamma = math.sqrt(2.0) * omega
    stationary_variance = 1.0 / (math.sqrt(2.0) * n_particles * omega)
    rho = math.exp(-gamma * tau)
    return rho, stationary_variance * (1.0 - rho * rho)


def harmonic_com_ground_variance(n_particles: int, omega: float) -> float:
    if n_particles < 1:
        raise ValueError("n_particles must be positive")
    if omega <= 0.0 or not math.isfinite(omega):
        raise ValueError("omega must be finite and positive")
    return 1.0 / (math.sqrt(2.0) * n_particles * omega)
