from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from hrdmc.system.geometry import (
    BASE_TRAP_QUADRATIC_COUPLING,
    lambda_from_relative_offset,
)

ENERGY_RESPONSE_SCHEMA_VERSION = "energy_response_trap_r2_v3"
@dataclass(frozen=True)
class PairedEnergyResponsePoint:
    seed: int
    relative_lambda_offset: float
    energy: float
    lambda_value: float | None = None
@dataclass(frozen=True)
class SeedEnergyResponseResult:
    seed: int
    inner_relative_offset: float
    outer_relative_offset: float
    center_energy: float
    inner_r2: float
    outer_r2: float
    richardson_r2: float
    richardson_minus_inner_scale_shift: float
@dataclass(frozen=True)
class TrapR2EnergyResponseResult:
    n_particles: int
    seed_results: tuple[SeedEnergyResponseResult, ...]
    inner_r2: float
    outer_r2: float
    pure_r2: float
    pure_r2_seed_stderr: float
    pure_r2_confidence_interval: tuple[float, float]
    finite_difference_scale_shift: float
    finite_difference_status: str
    rms_radius: float
    rms_radius_seed_stderr: float
    rms_radius_fd_scale_shift: float
    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": ENERGY_RESPONSE_SCHEMA_VERSION,
                "lambda0": BASE_TRAP_QUADRATIC_COUPLING,
                "relative_lambda_offsets": [-0.005, -0.0025, 0.0, 0.0025, 0.005],
                "seed_count": len(self.seed_results),
                "confidence_level": 0.95,
                "radius_status": "positive_r2",
                "ladder_status": "complete_paired_symmetric_five_point_ladder",
                "method": (
                    "paired symmetric Hellmann-Feynman response with Richardson extrapolation"
                ),
            }
        )
        return payload
def paired_seed_trap_r2(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    n_particles: int,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
) -> SeedEnergyResponseResult:
    clean = _validate_seed_points(points, lambda0=lambda0)
    offsets = np.asarray([point.relative_lambda_offset for point in clean])
    energies = np.asarray([point.energy for point in clean])
    inner, outer = float(offsets[3]), float(offsets[4])
    inner_r2 = float((energies[3] - energies[1]) / (n_particles * inner))
    outer_r2 = float((energies[4] - energies[0]) / (n_particles * outer))
    richardson = float((4.0 * inner_r2 - outer_r2) / 3.0)
    return SeedEnergyResponseResult(
        seed=clean[0].seed,
        inner_relative_offset=inner,
        outer_relative_offset=outer,
        center_energy=float(energies[2]),
        inner_r2=inner_r2,
        outer_r2=outer_r2,
        richardson_r2=richardson,
        richardson_minus_inner_scale_shift=float(richardson - inner_r2),
    )
def paired_trap_r2_from_energy_response(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    n_particles: int,
    lambda0: float = BASE_TRAP_QUADRATIC_COUPLING,
    confidence_level: float = 0.95,
) -> TrapR2EnergyResponseResult:
    """Infer trap R2 by the Hellmann--Feynman theorem [Hellmann1937; Feynman1939]."""
    if n_particles <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid response aggregation controls")
    grouped: dict[int, list[PairedEnergyResponsePoint]] = {}
    for point in points:
        grouped.setdefault(point.seed, []).append(point)
    if len(grouped) < 2:
        raise ValueError("paired energy response requires at least two seeds")
    seeds = tuple(
        paired_seed_trap_r2(tuple(grouped[seed]), n_particles=n_particles, lambda0=lambda0)
        for seed in sorted(grouped)
    )
    inner = np.asarray([row.inner_r2 for row in seeds])
    outer = np.asarray([row.outer_r2 for row in seeds])
    richardson = np.asarray([row.richardson_r2 for row in seeds])
    mean = float(np.mean(richardson))
    stderr = float(np.std(richardson, ddof=1) / np.sqrt(len(seeds)))
    scale_shift = float(abs(np.mean(inner) - np.mean(outer)) / 3.0)
    critical = float(student_t.ppf(0.5 * (1.0 + confidence_level), df=len(seeds) - 1))
    half_width = critical * stderr
    rms = float(np.sqrt(mean))
    return TrapR2EnergyResponseResult(
        n_particles=n_particles,
        seed_results=seeds,
        inner_r2=float(np.mean(inner)),
        outer_r2=float(np.mean(outer)),
        pure_r2=mean,
        pure_r2_seed_stderr=stderr,
        pure_r2_confidence_interval=(mean - half_width, mean + half_width),
        finite_difference_scale_shift=scale_shift,
        finite_difference_status=(
            "scale_shift_within_r2_confidence_half_width"
            if scale_shift <= half_width
            else "scale_shift_exceeds_r2_confidence_half_width"
        ),
        rms_radius=rms,
        rms_radius_seed_stderr=float(0.5 * stderr / rms),
        rms_radius_fd_scale_shift=float(0.5 * scale_shift / rms),
    )
def _validate_seed_points(
    points: tuple[PairedEnergyResponsePoint, ...],
    *,
    lambda0: float,
) -> tuple[PairedEnergyResponsePoint, ...]:
    if len(points) != 5 or len({point.seed for point in points}) != 1:
        raise ValueError("each seed must provide exactly five response points")
    clean = tuple(sorted(points, key=lambda point: point.relative_lambda_offset))
    offsets = np.asarray([point.relative_lambda_offset for point in clean])
    energies = np.asarray([point.energy for point in clean])
    outer, inner = float(offsets[4]), float(offsets[3])
    if (
        not np.all(np.isfinite(energies))
        or not np.allclose(offsets, [-outer, -inner, 0.0, inner, outer], rtol=0.0, atol=1e-12)
        or not np.isclose(outer, 2.0 * inner, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("response offsets must be finite symmetric pairs and zero")
    for point in clean:
        if point.lambda_value is not None and not np.isclose(
            point.lambda_value,
            lambda_from_relative_offset(point.relative_lambda_offset, lambda0=lambda0),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("lambda_value does not match its relative offset")
    return clean
