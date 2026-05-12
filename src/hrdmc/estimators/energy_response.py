from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ENERGY_RESPONSE_SCHEMA_VERSION = "energy_response_trap_r2_v2"


@dataclass(frozen=True)
class EnergyResponsePoint:
    """One ground-state energy estimate at one Hamiltonian coupling."""

    lambda_value: float
    energy: float
    energy_stderr: float | None = None
    omega: float | None = None
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnergyResponseFitResult:
    """Weighted centered-polynomial fit of E(lambda)."""

    lambda0: float
    degree: int
    point_count: int
    coefficients_centered: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    slope: float
    slope_stderr: float
    curvature: float
    curvature_stderr: float
    residual_rms: float
    chi2_dof: float
    status: str


@dataclass(frozen=True)
class TrapR2EnergyResponseResult:
    """Hellmann-Feynman trap-radius response result.

    For H(lambda)=H0+lambda*sum_i x_i^2,
    dE0/dlambda=<sum_i x_i^2> and pure_r2=(1/N)dE0/dlambda.
    """

    schema_version: str
    n_particles: int
    omega0: float
    lambda0: float
    fit: EnergyResponseFitResult
    pure_r2: float
    pure_r2_stderr: float
    paper_rms_radius: float
    paper_rms_radius_stderr: float
    bias_bracket: tuple[float, float]
    fit_response_status: str
    estimator_scope: str
    claim_boundary: str


def lambda_from_omega(omega: float) -> float:
    """Return lambda=0.5*omega^2 for the harmonic trap convention."""

    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be positive and finite")
    return float(0.5 * omega * omega)


def omega_from_lambda(lambda_value: float) -> float:
    """Return omega=sqrt(2*lambda) for the harmonic trap convention."""

    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("lambda_value must be positive and finite")
    return float(np.sqrt(2.0 * lambda_value))


def omega_ladder_from_relative_lambda_offsets(
    omega0: float,
    relative_offsets: tuple[float, ...],
) -> tuple[float, ...]:
    lambda0 = lambda_from_omega(omega0)
    offsets = np.asarray(relative_offsets, dtype=float)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("relative_offsets must be a non-empty one-dimensional sequence")
    lambda_values = lambda0 * (1.0 + offsets)
    if not np.all(np.isfinite(lambda_values)) or np.any(lambda_values <= 0.0):
        raise ValueError("relative lambda offsets must keep lambda positive")
    return tuple(float(omega_from_lambda(value)) for value in lambda_values)


def fit_energy_response(
    points: tuple[EnergyResponsePoint, ...],
    *,
    lambda0: float,
    degree: int = 2,
) -> EnergyResponseFitResult:
    """Fit E(lambda) around lambda0 with a centered polynomial."""

    if not np.isfinite(lambda0) or lambda0 <= 0.0:
        raise ValueError("lambda0 must be positive and finite")
    if degree < 1:
        raise ValueError("degree must be at least one")
    clean = _clean_points(points)
    if len(clean) < degree + 1:
        raise ValueError("not enough finite energy-response points for requested degree")

    lambdas = np.asarray([point.lambda_value for point in clean], dtype=float)
    energies = np.asarray([point.energy for point in clean], dtype=float)
    stderrs = np.asarray(
        [
            point.energy_stderr if point.energy_stderr is not None else np.nan
            for point in clean
        ],
        dtype=float,
    )
    x = lambdas - lambda0
    matrix = np.column_stack([x**power for power in range(degree + 1)])
    weights = _fit_weights(stderrs)
    sqrt_w = np.sqrt(weights)
    weighted_matrix = matrix * sqrt_w[:, np.newaxis]
    weighted_energy = energies * sqrt_w
    coefficients, *_ = np.linalg.lstsq(weighted_matrix, weighted_energy, rcond=None)
    residuals = energies - matrix @ coefficients
    normal = weighted_matrix.T @ weighted_matrix
    inv_normal = np.linalg.pinv(normal)
    dof = max(len(clean) - (degree + 1), 0)
    weighted_rss = float(np.sum(weights * residuals * residuals))
    if dof > 0:
        residual_scale = weighted_rss / dof
    else:
        residual_scale = 0.0
    if np.all(np.isfinite(stderrs) & (stderrs > 0.0)):
        residual_scale = max(residual_scale, 1.0) if dof > 0 else 1.0
    covariance = inv_normal * residual_scale
    slope = float(coefficients[1])
    slope_stderr = _sqrt_or_nan(float(covariance[1, 1]))
    curvature = float(2.0 * coefficients[2]) if degree >= 2 else 0.0
    curvature_stderr = _sqrt_or_nan(float(4.0 * covariance[2, 2])) if degree >= 2 else 0.0
    residual_rms = float(np.sqrt(np.mean(residuals * residuals)))
    chi2_dof = float(weighted_rss / dof) if dof > 0 else float("nan")
    status = "ENERGY_RESPONSE_FIT_GO" if np.isfinite(slope) else "ENERGY_RESPONSE_FIT_NO_GO"
    return EnergyResponseFitResult(
        lambda0=float(lambda0),
        degree=int(degree),
        point_count=len(clean),
        coefficients_centered=tuple(float(value) for value in coefficients),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        slope=slope,
        slope_stderr=slope_stderr,
        curvature=curvature,
        curvature_stderr=curvature_stderr,
        residual_rms=residual_rms,
        chi2_dof=chi2_dof,
        status=status,
    )


def trap_r2_from_energy_response(
    points: tuple[EnergyResponsePoint, ...],
    *,
    n_particles: int,
    omega0: float,
    degree: int = 2,
) -> TrapR2EnergyResponseResult:
    """Estimate pure trap R2/RMS through the Hellmann-Feynman energy response."""

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    lambda0 = lambda_from_omega(omega0)
    fit = fit_energy_response(points, lambda0=lambda0, degree=degree)
    pure_r2 = fit.slope / float(n_particles)
    pure_r2_stderr = fit.slope_stderr / float(n_particles)
    if pure_r2 >= 0.0 and np.isfinite(pure_r2):
        paper_rms = float(np.sqrt(pure_r2))
        paper_rms_stderr = (
            float(0.5 * pure_r2_stderr / paper_rms)
            if paper_rms > 0.0 and np.isfinite(pure_r2_stderr)
            else float("nan")
        )
        fit_response_status = (
            "ENERGY_RESPONSE_GO"
            if fit.status == "ENERGY_RESPONSE_FIT_GO"
            else "ENERGY_RESPONSE_FIT_NO_GO"
        )
    else:
        paper_rms = float("nan")
        paper_rms_stderr = float("nan")
        fit_response_status = "ENERGY_RESPONSE_NEGATIVE_R2_NO_GO"
    return TrapR2EnergyResponseResult(
        schema_version=ENERGY_RESPONSE_SCHEMA_VERSION,
        n_particles=int(n_particles),
        omega0=float(omega0),
        lambda0=lambda0,
        fit=fit,
        pure_r2=float(pure_r2),
        pure_r2_stderr=float(pure_r2_stderr),
        paper_rms_radius=paper_rms,
        paper_rms_radius_stderr=paper_rms_stderr,
        bias_bracket=_slope_bias_bracket(points, lambda0=lambda0, n_particles=n_particles),
        fit_response_status=fit_response_status,
        estimator_scope=(
            "Hellmann-Feynman energy-response estimator for trap R2/RMS; "
            "does not estimate density"
        ),
        claim_boundary=(
            "Paper RMS radius is sqrt(aggregated pure_r2). Each energy point "
            "must pass RN-DMC methodology gates and finite-difference stability checks."
        ),
    )


def _clean_points(points: tuple[EnergyResponsePoint, ...]) -> tuple[EnergyResponsePoint, ...]:
    if not points:
        raise ValueError("at least one energy response point is required")
    clean = tuple(
        point
        for point in points
        if np.isfinite(point.lambda_value)
        and point.lambda_value > 0.0
        and np.isfinite(point.energy)
    )
    if len(clean) != len(points):
        raise ValueError("all energy response points must have finite positive lambda and energy")
    lambdas = np.asarray([point.lambda_value for point in clean], dtype=float)
    if np.unique(lambdas).size != lambdas.size:
        raise ValueError("lambda values must be unique")
    return tuple(sorted(clean, key=lambda point: point.lambda_value))


def _fit_weights(stderrs: FloatArray) -> FloatArray:
    valid = np.isfinite(stderrs) & (stderrs > 0.0)
    if not np.any(valid):
        return np.ones(stderrs.shape, dtype=float)
    weights = np.ones(stderrs.shape, dtype=float)
    weights[valid] = 1.0 / (stderrs[valid] * stderrs[valid])
    if np.any(~valid):
        weights[~valid] = float(np.median(weights[valid]))
    return weights


def _slope_bias_bracket(
    points: tuple[EnergyResponsePoint, ...],
    *,
    lambda0: float,
    n_particles: int,
) -> tuple[float, float]:
    slopes: list[float] = []
    clean = _clean_points(points)
    if len(clean) >= 2:
        try:
            slopes.append(fit_energy_response(clean, lambda0=lambda0, degree=1).slope)
        except ValueError:
            pass
    if len(clean) >= 3:
        try:
            slopes.append(fit_energy_response(clean, lambda0=lambda0, degree=2).slope)
        except ValueError:
            pass
    lambdas = np.asarray([point.lambda_value for point in clean], dtype=float)
    energies = np.asarray([point.energy for point in clean], dtype=float)
    lower = np.flatnonzero(lambdas < lambda0)
    upper = np.flatnonzero(lambdas > lambda0)
    for i in lower:
        for j in upper:
            slopes.append(float((energies[j] - energies[i]) / (lambdas[j] - lambdas[i])))
    finite = np.asarray([value / float(n_particles) for value in slopes if np.isfinite(value)])
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.min(finite)), float(np.max(finite)))


def _sqrt_or_nan(value: float) -> float:
    return float(np.sqrt(value)) if value >= 0.0 and np.isfinite(value) else float("nan")
