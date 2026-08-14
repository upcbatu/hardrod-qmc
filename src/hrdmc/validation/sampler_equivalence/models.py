from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.system.settings import TrappedCase

FloatArray = NDArray[np.float64]
VMC_VALIDATION_CASE_IDS = ("N10_A0", "N10_A1")
VMC_SAMPLERS = ("random_walk_metropolis", "branching_free_mala")
VMC_CUTOFF_EPSILONS = (0.01, 0.02, 0.04, 0.08)
@dataclass(frozen=True)
class VMCValidationPolicy:
    """Prospective numerical qualifications for the thesis VMC packet."""
    confidence_level: float = 0.95
    rhat_limit: float = 1.01
    total_bulk_ess_minimum: float = 400.0
    per_seed_ess_minimum: float = 50.0
    histogram_normalization_tolerance: float = 1.0e-8
    histogram_out_of_grid_mass_limit: float = 1.0e-6
    kinetic_relative_margin: float = 0.01
    energy_relative_margin: float = 0.005
    trap_relative_margin: float = 0.01
    r2_relative_margin: float = 0.01
    weighted_free_gap_relative_margin: float = 0.02
    density_relative_l2_margin: float = 0.05
    gap_distribution_relative_l2_margin: float = 0.05
    def validate(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not 1.0 < self.rhat_limit < 2.0:
            raise ValueError("rhat_limit must lie between one and two")
        positive = (
            self.total_bulk_ess_minimum,
            self.per_seed_ess_minimum,
            self.histogram_normalization_tolerance,
            self.histogram_out_of_grid_mass_limit,
            self.kinetic_relative_margin,
            self.energy_relative_margin,
            self.trap_relative_margin,
            self.r2_relative_margin,
            self.weighted_free_gap_relative_margin,
            self.density_relative_l2_margin,
            self.gap_distribution_relative_l2_margin,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("VMC validation thresholds must be finite and positive")
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)
@dataclass(frozen=True)
class VMCSamplingControls:
    walkers: int = 64
    burn_in_steps: int = 5_000
    production_steps: int = 20_000
    block_steps: int = 20
    density_bins: int = 840
    free_gap_bins: int = 640
    cutoff_epsilons: tuple[float, ...] = VMC_CUTOFF_EPSILONS
    def validate(self) -> None:
        if self.walkers <= 0:
            raise ValueError("walkers must be positive")
        if self.burn_in_steps < 0 or self.production_steps <= 0:
            raise ValueError("burn-in must be non-negative and production positive")
        if self.block_steps <= 0:
            raise ValueError("block_steps must be positive")
        if self.density_bins < 80 or self.free_gap_bins < 80:
            raise ValueError("VMC histogram grids must contain at least 80 bins")
        _validated_cutoffs(self.cutoff_epsilons)
    @property
    def maximum_records(self) -> int:
        return (self.production_steps + self.block_steps - 1) // self.block_steps
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "maximum_records": self.maximum_records,
            "production_sample_count_per_seed": self.walkers * self.production_steps,
        }
@dataclass(frozen=True)
class VMCSamplerChoice:
    method: str
    proposal_scale: float
    drift_limiter: str = "none"
    def validate(self) -> None:
        if self.method not in VMC_SAMPLERS:
            raise ValueError(f"unsupported VMC sampler: {self.method}")
        if not np.isfinite(self.proposal_scale) or self.proposal_scale <= 0.0:
            raise ValueError("proposal_scale must be finite and positive")
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")
        if self.method == "random_walk_metropolis" and self.drift_limiter != "none":
            raise ValueError("RWM does not use a drift limiter")
    @property
    def engine_method(self) -> str:
        return "rwm" if self.method == "random_walk_metropolis" else "mala"
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "engine_method": self.engine_method,
            "proposal_parameter": "step_size" if self.engine_method == "rwm" else "dt",
        }
def density_bin_edges(case: TrappedCase, bins: int = 840) -> FloatArray:
    if bins < 2:
        raise ValueError("density grid requires at least two bins")
    # The excluded-volume span is explicit and the remaining eight oscillator
    # lengths cover the Gaussian tail of every prospectively selected case.
    extent = 8.0 + 0.75 * (case.n_particles - 1) * case.rod_length
    return np.linspace(-extent, extent, bins + 1, dtype=float)
def free_gap_bin_edges(case: TrappedCase, bins: int = 640) -> FloatArray:
    if bins < 2:
        raise ValueError("free-gap grid requires at least two bins")
    # Free gaps are reduced coordinates.  Sixteen oscillator lengths is far
    # beyond the occupied support for N=10 at A=0 or A=1, while the explicit
    # out-of-grid counter still makes any bad assumption fail closed.
    extent = max(16.0, 8.0 + 0.25 * case.n_particles * case.rod_length)
    return np.linspace(0.0, extent, bins + 1, dtype=float)
def _validated_cutoffs(values: tuple[float, ...]) -> None:
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or array.size < 3
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
        or not np.all(np.diff(array) > 0.0)
    ):
        raise ValueError("cutoff_epsilons must be a finite increasing positive ladder")
