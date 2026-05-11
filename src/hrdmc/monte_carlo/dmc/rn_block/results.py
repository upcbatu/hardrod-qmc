from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.analysis import RunningStats
from hrdmc.monte_carlo.dmc.common.results import WeightedDMCResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RNBlockDMCResult(WeightedDMCResult):
    """Result contract for RN-block DMC runs."""


@dataclass(frozen=True)
class RNBlockStreamingSummary:
    """Compact RN-block DMC result that does not retain raw snapshots."""

    stored_batch_count: int
    sample_count: int
    mixed_energy: float
    mixed_energy_batch_stats: RunningStats
    r2_radius: float
    r2_radius_batch_stats: RunningStats
    density_bin_edges: FloatArray
    density_counts: FloatArray
    density_integral: float
    lost_out_of_grid_sample_count: int
    lost_out_of_grid_weight: float
    metadata: dict
    trace_times: FloatArray | None = None
    mixed_energy_trace: FloatArray | None = None
    rms_radius_trace: FloatArray | None = None
    r2_radius_trace: FloatArray | None = None
    local_energy_variance_trace: FloatArray | None = None
    log_weight_span_trace: FloatArray | None = None
    ess_fraction_trace: FloatArray | None = None
    invalid_proposal_fraction_trace: FloatArray | None = None
    hard_wall_kill_fraction_trace: FloatArray | None = None
    zero_weight_excluded_fraction_trace: FloatArray | None = None
    rn_logk_mean_trace: FloatArray | None = None
    rn_logq_mean_trace: FloatArray | None = None
    rn_logw_increment_mean_trace: FloatArray | None = None
    rn_logw_increment_variance_trace: FloatArray | None = None
    retained_fraction_trace: FloatArray | None = None

    @property
    def rms_radius(self) -> float:
        return float(np.sqrt(self.r2_radius))

    @property
    def density(self) -> FloatArray:
        widths = np.diff(self.density_bin_edges)
        return self.density_counts / widths

    def to_dict(self) -> dict:
        return {
            "stored_batch_count": self.stored_batch_count,
            "sample_count": self.sample_count,
            "mixed_energy": self.mixed_energy,
            "mixed_energy_batch_stats": self.mixed_energy_batch_stats.to_dict(),
            "r2_radius": self.r2_radius,
            "r2_radius_batch_stats": self.r2_radius_batch_stats.to_dict(),
            "rms_radius": self.rms_radius,
            "density_integral": self.density_integral,
            "lost_out_of_grid_sample_count": self.lost_out_of_grid_sample_count,
            "lost_out_of_grid_weight": self.lost_out_of_grid_weight,
            "metadata": self.metadata,
            "trace_times": _array_or_none(self.trace_times),
            "mixed_energy_trace": _array_or_none(self.mixed_energy_trace),
            "r2_radius_trace": _array_or_none(self.r2_radius_trace),
            "rms_radius_trace": _array_or_none(self.rms_radius_trace),
            "local_energy_variance_trace": _array_or_none(self.local_energy_variance_trace),
            "log_weight_span_trace": _array_or_none(self.log_weight_span_trace),
            "ess_fraction_trace": _array_or_none(self.ess_fraction_trace),
            "invalid_proposal_fraction_trace": _array_or_none(
                self.invalid_proposal_fraction_trace
            ),
            "hard_wall_kill_fraction_trace": _array_or_none(
                self.hard_wall_kill_fraction_trace
            ),
            "zero_weight_excluded_fraction_trace": _array_or_none(
                self.zero_weight_excluded_fraction_trace
            ),
            "rn_logk_mean_trace": _array_or_none(self.rn_logk_mean_trace),
            "rn_logq_mean_trace": _array_or_none(self.rn_logq_mean_trace),
            "rn_logw_increment_mean_trace": _array_or_none(
                self.rn_logw_increment_mean_trace
            ),
            "rn_logw_increment_variance_trace": _array_or_none(
                self.rn_logw_increment_variance_trace
            ),
            "retained_fraction_trace": _array_or_none(self.retained_fraction_trace),
        }


def _array_or_none(values: FloatArray | None) -> list[float] | None:
    if values is None:
        return None
    return values.tolist()
