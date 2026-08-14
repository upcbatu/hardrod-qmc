from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class PopulationEnergyPoint:
    """One mixed-energy estimate at a fixed walker population."""
    walkers: int
    energy: float
    conservative_stderr: float
    seed_ids: tuple[int, ...]
    seed_energies: FloatArray
    label: str | None = None
    def __post_init__(self) -> None:
        if (
            isinstance(self.walkers, bool)
            or not isinstance(self.walkers, (int, np.integer))
            or self.walkers <= 0
        ):
            raise ValueError("walkers must be a positive integer")
        if not math.isfinite(self.energy):
            raise ValueError("population-point energy must be finite")
        if not math.isfinite(self.conservative_stderr) or self.conservative_stderr <= 0.0:
            raise ValueError("population-point conservative stderr must be finite and positive")
        if (
            len(self.seed_ids) < 2
            or len(set(self.seed_ids)) != len(self.seed_ids)
            or any(
                isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                for seed in self.seed_ids
            )
        ):
            raise ValueError("population points require at least two unique seeds")
        values = np.asarray(self.seed_energies, dtype=np.float64)
        if values.shape != (len(self.seed_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("seed_energies must contain one finite value per seed")
        if not math.isclose(
            self.energy,
            float(np.mean(values)),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("population-point energy must equal the mean seed energy")
        object.__setattr__(self, "seed_energies", values.copy())
    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "seed_ids": list(self.seed_ids),
            "seed_energies": self.seed_energies.tolist(),
        }

@dataclass(frozen=True)
class PopulationDifferenceBound:
    first_walkers: int
    second_walkers: int
    mean_difference: float
    observed_absolute_difference: float
    paired_standard_error: float
    first_run_conservative_stderr: float
    second_run_conservative_stderr: float
    source_run_quadrature_standard_error: float
    worst_case_arbitrary_covariance_standard_error_envelope: float
    conservative_standard_error: float
    confidence_level: float
    degrees_of_freedom: int
    critical_value: float
    upper_allowance: float
    reporting_resolution: float
    bounded_below_reporting_resolution: bool
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def population_difference_bound(
    first: PopulationEnergyPoint,
    second: PopulationEnergyPoint,
    *,
    reporting_resolution: float,
    confidence_level: float = 0.95,
) -> PopulationDifferenceBound:
    """Bound a paired population-energy difference without hiding its error inputs."""
    _validate_reporting_controls(
        reporting_resolution=reporting_resolution,
        confidence_level=confidence_level,
        fit_alpha=0.05,
    )
    if first.seed_ids != second.seed_ids:
        raise ValueError("population differences require identical ordered seed ids")
    differences = second.seed_energies - first.seed_energies
    # Form the matched-seed contrast directly.  Reconstructing the same
    # quantity as ``second.energy - first.energy`` loses precision when the
    # common energy offset is large; each PopulationEnergyPoint already
    # validates that its aggregate energy is the mean of its seed energies.
    mean_difference = float(np.mean(differences))
    paired_standard_error = float(np.std(differences, ddof=1) / math.sqrt(float(differences.size)))
    source_run_quadrature_standard_error = math.hypot(
        first.conservative_stderr,
        second.conservative_stderr,
    )
    worst_case_arbitrary_covariance_standard_error_envelope = (
        first.conservative_stderr + second.conservative_stderr
    )
    conservative_standard_error = max(
        paired_standard_error,
        source_run_quadrature_standard_error,
    )
    degrees_of_freedom = differences.size - 1
    critical_value = _student_critical_value(
        confidence_level,
        degrees_of_freedom=degrees_of_freedom,
    )
    observed = abs(mean_difference)
    upper_allowance = float(observed + critical_value * conservative_standard_error)
    return PopulationDifferenceBound(
        first_walkers=first.walkers,
        second_walkers=second.walkers,
        mean_difference=mean_difference,
        observed_absolute_difference=observed,
        paired_standard_error=paired_standard_error,
        first_run_conservative_stderr=first.conservative_stderr,
        second_run_conservative_stderr=second.conservative_stderr,
        source_run_quadrature_standard_error=source_run_quadrature_standard_error,
        worst_case_arbitrary_covariance_standard_error_envelope=(
            worst_case_arbitrary_covariance_standard_error_envelope
        ),
        conservative_standard_error=conservative_standard_error,
        confidence_level=confidence_level,
        degrees_of_freedom=degrees_of_freedom,
        critical_value=critical_value,
        upper_allowance=upper_allowance,
        reporting_resolution=reporting_resolution,
        bounded_below_reporting_resolution=_bounded_at_resolution(
            upper_allowance,
            reporting_resolution,
        ),
    )

def _validate_reporting_controls(
    *,
    reporting_resolution: float,
    confidence_level: float,
    fit_alpha: float,
) -> None:
    if not math.isfinite(reporting_resolution) or reporting_resolution <= 0.0:
        raise ValueError("reporting_resolution must be finite and positive")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not math.isfinite(fit_alpha) or not 0.0 < fit_alpha < 1.0:
        raise ValueError("fit_alpha must lie strictly between zero and one")

def _student_critical_value(confidence_level: float, *, degrees_of_freedom: int) -> float:
    value = float(
        student_t.ppf(
            1.0 - (1.0 - confidence_level) / 2.0,
            df=degrees_of_freedom,
        )
    )
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("Student-t critical value is unavailable")
    return value

def _bounded_at_resolution(upper_allowance: float, reporting_resolution: float) -> bool:
    return upper_allowance <= reporting_resolution or math.isclose(
        upper_allowance,
        reporting_resolution,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )
