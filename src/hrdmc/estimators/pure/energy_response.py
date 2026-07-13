from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from hrdmc.systems.external_potential import (
    BASE_TRAP_QUADRATIC_COUPLING,
    lambda_from_relative_offset,
)

ENERGY_RESPONSE_SCHEMA_VERSION = "energy_response_trap_r2_v3"


@dataclass(frozen=True)
class PairedEnergyResponsePoint:
    """One seed-level energy at one relative quadratic-coupling offset."""

    seed: int
    relative_lambda_offset: float
    energy: float
    lambda_value: float | None = None
    energy_stderr: float | None = None
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedEnergyResponseResult:
    """Symmetric two-scale response obtained from one complete seed ladder."""

    seed: int
    inner_relative_offset: float
    outer_relative_offset: float
    center_energy: float
    inner_r2: float
    outer_r2: float
    richardson_r2: float
    richardson_minus_inner_scale_shift: float
    absolute_richardson_scale_shift: float


@dataclass(frozen=True)
class TrapR2EnergyResponseResult:
    """Paired-seed Hellmann-Feynman trap-radius response.

    In fixed oscillator units,

        H(epsilon) = H(0) + 0.5 * epsilon * sum_i (q_i-q0)^2,
        lambda(epsilon) = 0.5 * (1 + epsilon).

    Therefore pure_r2 = (1/N) dE/dlambda at lambda=0.5.  Symmetric
    differences at epsilon=+/-h and +/-2h are Richardson-combined per seed
    before seed aggregation.  The reported finite-difference scale shift is
    ``|D_h-D_2h|/3``, equivalently the magnitude of the Richardson correction
    to ``D_h``.  It diagnoses resolved step-size sensitivity; it is not a
    rigorous bound on the remaining fourth-order Richardson error.
    """

    schema_version: str
    n_particles: int
    lambda0: float
    relative_lambda_offsets: tuple[float, ...]
    seed_count: int
    seed_results: tuple[SeedEnergyResponseResult, ...]
    inner_r2: float
    outer_r2: float
    pure_r2: float
    pure_r2_seed_stderr: float
    confidence_level: float
    pure_r2_confidence_interval: tuple[float, float]
    finite_difference_scale_shift: float
    finite_difference_relative_scale_shift: float
    finite_difference_scale_shift_to_confidence_half_width: float
    finite_difference_status: str
    rms_radius: float
    rms_radius_seed_stderr: float
    rms_radius_fd_scale_shift: float
    radius_status: str
    ladder_status: str
    method: str


def lambda_ladder_from_relative_offsets(
    relative_offsets: tuple[float, ...],
    *,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
) -> tuple[float, ...]:
    """Return fixed-unit quadratic couplings without introducing a trap frequency."""

    if not relative_offsets:
        raise ValueError("relative_offsets must not be empty")
    offsets = np.asarray(relative_offsets, dtype=float)
    if offsets.ndim != 1 or not np.all(np.isfinite(offsets)):
        raise ValueError("relative_offsets must be a finite one-dimensional sequence")
    if np.unique(offsets).size != offsets.size:
        raise ValueError("relative_offsets must be unique")
    return tuple(lambda_from_relative_offset(float(offset), lambda0=lambda0) for offset in offsets)


def paired_seed_trap_r2(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    n_particles: int,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
) -> SeedEnergyResponseResult:
    """Compute the symmetric Richardson response for one complete seed ladder."""

    if n_particles <= 0:
        raise ValueError("n_particles must be positive")
    _validate_lambda0(lambda0)
    clean = _validate_seed_points(points, lambda0=lambda0)
    seed = clean[0].seed
    offsets = np.asarray([point.relative_lambda_offset for point in clean], dtype=float)
    inner = float(offsets[3])
    outer = float(offsets[4])
    energies = np.asarray([point.energy for point in clean], dtype=float)

    inner_r2 = float((energies[3] - energies[1]) / (float(n_particles) * inner))
    outer_r2 = float((energies[4] - energies[0]) / (float(n_particles) * outer))
    richardson_r2 = float((4.0 * inner_r2 - outer_r2) / 3.0)
    correction = float(richardson_r2 - inner_r2)
    values = np.asarray([inner_r2, outer_r2, richardson_r2, correction])
    if not np.all(np.isfinite(values)):
        raise ValueError("seed energy response produced non-finite radius values")
    return SeedEnergyResponseResult(
        seed=int(seed),
        inner_relative_offset=inner,
        outer_relative_offset=outer,
        center_energy=float(energies[2]),
        inner_r2=inner_r2,
        outer_r2=outer_r2,
        richardson_r2=richardson_r2,
        richardson_minus_inner_scale_shift=correction,
        absolute_richardson_scale_shift=abs(correction),
    )


def paired_trap_r2_from_energy_response(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    n_particles: int,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
    confidence_level: float = 0.95,
    minimum_seed_count: int = 2,
) -> TrapR2EnergyResponseResult:
    """Aggregate complete paired seed ladders into pure R2 and RMS."""

    if not points:
        raise ValueError("at least one paired energy-response point is required")
    if minimum_seed_count < 2:
        raise ValueError("minimum_seed_count must be at least two")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    _validate_lambda0(lambda0)

    grouped: dict[int, list[PairedEnergyResponsePoint]] = {}
    for point in points:
        grouped.setdefault(int(point.seed), []).append(point)
    if len(grouped) < minimum_seed_count:
        raise ValueError(
            f"paired energy response requires at least {minimum_seed_count} complete seeds"
        )

    seed_results = tuple(
        paired_seed_trap_r2(
            tuple(grouped[seed]),
            n_particles=n_particles,
            lambda0=lambda0,
        )
        for seed in sorted(grouped)
    )
    first = seed_results[0]
    for result in seed_results[1:]:
        if not np.isclose(
            result.inner_relative_offset,
            first.inner_relative_offset,
            rtol=0.0,
            atol=1.0e-12,
        ) or not np.isclose(
            result.outer_relative_offset,
            first.outer_relative_offset,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("all seeds must use the same symmetric lambda ladder")

    inner_values = np.asarray([result.inner_r2 for result in seed_results], dtype=float)
    outer_values = np.asarray([result.outer_r2 for result in seed_results], dtype=float)
    richardson_values = np.asarray(
        [result.richardson_r2 for result in seed_results],
        dtype=float,
    )
    seed_count = len(seed_results)
    inner_r2 = float(np.mean(inner_values))
    outer_r2 = float(np.mean(outer_values))
    pure_r2 = float(np.mean(richardson_values))
    seed_stderr = float(np.std(richardson_values, ddof=1) / np.sqrt(seed_count))
    finite_difference_scale_shift = float(abs(inner_r2 - outer_r2) / 3.0)
    critical = float(student_t.ppf(0.5 * (1.0 + confidence_level), df=seed_count - 1))
    confidence_interval = (
        float(pure_r2 - critical * seed_stderr),
        float(pure_r2 + critical * seed_stderr),
    )
    confidence_half_width = float(critical * seed_stderr)
    relative_fd_scale_shift = (
        float(finite_difference_scale_shift / abs(pure_r2)) if pure_r2 != 0.0 else float("inf")
    )
    if confidence_half_width > 0.0:
        fd_to_confidence = float(finite_difference_scale_shift / confidence_half_width)
        finite_difference_status = (
            "scale_shift_within_r2_confidence_half_width"
            if fd_to_confidence <= 1.0
            else "scale_shift_exceeds_r2_confidence_half_width"
        )
    elif finite_difference_scale_shift == 0.0:
        fd_to_confidence = 0.0
        finite_difference_status = "difference_scales_identical"
    else:
        fd_to_confidence = float("inf")
        finite_difference_status = "scale_shift_resolved_with_zero_seed_error"

    if pure_r2 > 0.0 and np.isfinite(pure_r2) and np.isfinite(seed_stderr):
        rms_radius = float(np.sqrt(pure_r2))
        rms_radius_stderr = float(0.5 * seed_stderr / rms_radius)
        rms_radius_fd_scale_shift = float(0.5 * finite_difference_scale_shift / rms_radius)
        radius_status = "positive_r2"
    else:
        rms_radius = float("nan")
        rms_radius_stderr = float("nan")
        rms_radius_fd_scale_shift = float("nan")
        radius_status = "nonpositive_r2"

    inner = first.inner_relative_offset
    outer = first.outer_relative_offset
    return TrapR2EnergyResponseResult(
        schema_version=ENERGY_RESPONSE_SCHEMA_VERSION,
        n_particles=int(n_particles),
        lambda0=float(lambda0),
        relative_lambda_offsets=(-outer, -inner, 0.0, inner, outer),
        seed_count=seed_count,
        seed_results=seed_results,
        inner_r2=inner_r2,
        outer_r2=outer_r2,
        pure_r2=pure_r2,
        pure_r2_seed_stderr=seed_stderr,
        confidence_level=float(confidence_level),
        pure_r2_confidence_interval=confidence_interval,
        finite_difference_scale_shift=finite_difference_scale_shift,
        finite_difference_relative_scale_shift=relative_fd_scale_shift,
        finite_difference_scale_shift_to_confidence_half_width=fd_to_confidence,
        finite_difference_status=finite_difference_status,
        rms_radius=rms_radius,
        rms_radius_seed_stderr=rms_radius_stderr,
        rms_radius_fd_scale_shift=rms_radius_fd_scale_shift,
        radius_status=radius_status,
        ladder_status="complete_paired_symmetric_five_point_ladder",
        method="paired symmetric Hellmann-Feynman response with Richardson extrapolation",
    )


def _validate_seed_points(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    lambda0: float,
) -> tuple[PairedEnergyResponsePoint, ...]:
    if len(points) != 5:
        raise ValueError("each seed must contain exactly five energy-response points")
    seeds = {int(point.seed) for point in points}
    if len(seeds) != 1:
        raise ValueError("paired_seed_trap_r2 accepts points from exactly one seed")
    clean = tuple(sorted(points, key=lambda point: point.relative_lambda_offset))
    offsets = np.asarray([point.relative_lambda_offset for point in clean], dtype=float)
    energies = np.asarray([point.energy for point in clean], dtype=float)
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(energies)):
        raise ValueError("energy-response offsets and energies must be finite")
    if np.unique(offsets).size != offsets.size:
        raise ValueError("each seed must contain unique lambda offsets")
    if not np.isclose(offsets[2], 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("paired energy response requires the base lambda point")
    inner = float(offsets[3])
    outer = float(offsets[4])
    expected = np.asarray([-outer, -inner, 0.0, inner, outer], dtype=float)
    if (
        inner <= 0.0
        or outer <= inner
        or not np.allclose(
            offsets,
            expected,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("energy-response offsets must form two symmetric pairs and zero")
    if not np.isclose(outer, 2.0 * inner, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("outer relative lambda offset must equal twice the inner offset")

    for point in clean:
        if point.lambda_value is not None:
            expected_lambda = lambda_from_relative_offset(
                point.relative_lambda_offset,
                lambda0=lambda0,
            )
            if not np.isfinite(point.lambda_value) or not np.isclose(
                point.lambda_value,
                expected_lambda,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError("lambda_value is inconsistent with its relative offset")
        if point.energy_stderr is not None and (
            not np.isfinite(point.energy_stderr) or point.energy_stderr <= 0.0
        ):
            raise ValueError("energy_stderr must be positive and finite when provided")
    return clean


def _validate_lambda0(lambda0: float) -> None:
    lambda_from_relative_offset(0.0, lambda0=lambda0)
