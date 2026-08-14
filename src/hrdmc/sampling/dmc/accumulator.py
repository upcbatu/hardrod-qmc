from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.estimators.mixed import (
    StreamingBatchObservables,
    streaming_batch_observables,
)
from hrdmc.sampling.dmc.checkpoint import DMCCheckpointMixin
from hrdmc.sampling.dmc.guide_api import evaluate_guide, guide_batch_backend, valid_rows
from hrdmc.sampling.dmc.population import (
    finite_max,
    finite_mean,
    finite_min,
    log_weight_span,
    normalize_log_weights,
    safe_fraction,
)
from hrdmc.sampling.dmc.results import DMCStreamingSummary
from hrdmc.sampling.dmc.telemetry import DMCStepTelemetry, TraceAccumulator
from hrdmc.sampling.mobility import free_gap_batch_diagnostics
from hrdmc.statistics.streaming import RunningHistogram, RunningStats
from hrdmc.system.geometry import OpenLineHardRodSystem
from hrdmc.trial.guide import DMCGuide

FloatArray = NDArray[np.float64]
@dataclass
class DMCStreamingState(DMCCheckpointMixin):
    step_start: int
    positions: FloatArray
    local_energies: FloatArray
    log_weights: FloatArray
    density_grid: FloatArray
    density_histogram: RunningHistogram
    energy_stats: RunningStats = field(default_factory=RunningStats.empty)
    r2_stats: RunningStats = field(default_factory=RunningStats.empty)
    energy_numerator: float = 0.0
    r2_numerator: float = 0.0
    weight_denominator: float = 0.0
    stored_batch_count: int = 0
    local_step_count: int = 0
    killed_count: int = 0
    resample_count: int = 0
    finite_sample_count: int = 0
    valid_sample_count: int = 0
    included_sample_count: int = 0
    total_sample_count: int = 0
    ess_values: list[float] = field(default_factory=list)
    trace_times: list[float] = field(default_factory=list)
    mixed_energy_trace: list[float] = field(default_factory=list)
    rms_radius_trace: list[float] = field(default_factory=list)
    r2_radius_trace: list[float] = field(default_factory=list)
    local_energy_variance_trace: list[float] = field(default_factory=list)
    local_energy_median_trace: list[float] = field(default_factory=list)
    local_energy_mad_trace: list[float] = field(default_factory=list)
    local_energy_p001_trace: list[float] = field(default_factory=list)
    local_energy_p01_trace: list[float] = field(default_factory=list)
    local_energy_p99_trace: list[float] = field(default_factory=list)
    local_energy_p999_trace: list[float] = field(default_factory=list)
    log_weight_span_trace: list[float] = field(default_factory=list)
    ess_fraction_trace: list[float] = field(default_factory=list)
    invalid_proposal_fraction_trace: list[float] = field(default_factory=list)
    hard_wall_kill_fraction_trace: list[float] = field(default_factory=list)
    local_acceptance_fraction_trace: list[float] = field(default_factory=list)
    metropolis_rejection_fraction_trace: list[float] = field(default_factory=list)
    drift_norm_max_trace: list[float] = field(default_factory=list)
    configuration_esjd_trace: list[float] = field(default_factory=list)
    r2_esjd_trace: list[float] = field(default_factory=list)
    weighted_free_gap_esjd_trace: list[float] = field(default_factory=list)
    weighted_free_gap_mean_trace: list[float] = field(default_factory=list)
    free_gap_min_trace: list[float] = field(default_factory=list)
    free_gap_p01_trace: list[float] = field(default_factory=list)
    zero_weight_excluded_fraction_trace: list[float] = field(default_factory=list)
    retained_fraction_trace: list[float] = field(default_factory=list)
    interval_trace: TraceAccumulator = field(default_factory=TraceAccumulator)
    @classmethod
    def from_initial(
        cls,
        *,
        initial_walkers: FloatArray,
        guide: DMCGuide,
        system: OpenLineHardRodSystem,
        density_grid: FloatArray,
    ) -> DMCStreamingState:
        positions = np.asarray(initial_walkers, dtype=float).copy()
        if positions.ndim != 2:
            raise ValueError("initial_walkers must have shape (n_walkers, n_particles)")
        if positions.shape[1] != system.n_particles:
            raise ValueError("initial walker particle count must match system")
        local_energies, valid = evaluate_guide(guide, positions)
        if not np.all(valid):
            raise ValueError("initial_walkers must all be valid finite guide configurations")
        return cls(
            step_start=1,
            positions=positions,
            local_energies=local_energies,
            log_weights=np.zeros(positions.shape[0], dtype=float),
            density_grid=np.asarray(density_grid, dtype=float),
            density_histogram=RunningHistogram.from_centers(density_grid),
        )
    def record_step(
        self,
        *,
        killed: NDArray[np.bool_],
        ess: float,
        telemetry: DMCStepTelemetry,
    ) -> None:
        killed_count = int(np.count_nonzero(killed))
        self.killed_count += killed_count
        self.ess_values.append(ess)
        self.interval_trace.update(
            killed_count=killed_count,
            walker_count=self.positions.shape[0],
            ess=ess,
            log_weight_span=log_weight_span(self.log_weights),
            telemetry=telemetry,
        )
    def record_resample(self, resampled: bool) -> None:
        if resampled:
            self.resample_count += 1
    def reset_interval_trace(self) -> None:
        """Start production telemetry without carrying burn-in steps forward."""
        self.interval_trace = TraceAccumulator()
    def record_production_if_due(
        self,
        *,
        step_index: int,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        dt: float,
        system: OpenLineHardRodSystem,
        guide: DMCGuide,
    ) -> None:
        if step_index <= burn_in_steps:
            return
        production_index = step_index - burn_in_steps
        if production_index % store_every != 0 and production_index != production_steps:
            return
        normalized_weights = normalize_log_weights(self.log_weights)
        valid_mask = valid_rows(system, guide, self.positions)
        batch = streaming_batch_observables(
            self.positions,
            self.local_energies,
            normalized_weights,
            valid_mask,
            center=system.center,
        )
        self._record_batch(batch, production_index=production_index, dt=dt, system=system)
    def to_summary(
        self,
        *,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        ess_resample_fraction: float,
        guide: DMCGuide,
    ) -> DMCStreamingSummary:
        if self.weight_denominator <= 0.0:
            raise RuntimeError("no positive-weight production samples were accumulated")
        finite_fraction = safe_fraction(self.finite_sample_count, self.total_sample_count)
        valid_fraction = safe_fraction(self.valid_sample_count, self.total_sample_count)
        included_fraction = safe_fraction(self.included_sample_count, self.total_sample_count)
        density_counts = self.density_histogram.counts / self.weight_denominator
        return DMCStreamingSummary(
            stored_batch_count=self.stored_batch_count,
            sample_count=int(self.stored_batch_count * self.positions.shape[0]),
            mixed_energy=float(self.energy_numerator / self.weight_denominator),
            mixed_energy_batch_stats=self.energy_stats,
            r2_radius=float(self.r2_numerator / self.weight_denominator),
            r2_radius_batch_stats=self.r2_stats,
            density_bin_edges=self.density_histogram.bin_edges,
            density_counts=density_counts,
            density_integral=float(np.sum(density_counts)),
            lost_out_of_grid_sample_count=self.density_histogram.lost_sample_count,
            lost_out_of_grid_weight=self.density_histogram.lost_weight / self.weight_denominator,
            metadata=self._summary_metadata(
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                ess_resample_fraction=ess_resample_fraction,
                guide=guide,
                finite_fraction=finite_fraction,
                valid_fraction=valid_fraction,
                included_fraction=included_fraction,
            ),
            trace_times=_array(self.trace_times),
            mixed_energy_trace=_array(self.mixed_energy_trace),
            rms_radius_trace=_array(self.rms_radius_trace),
            r2_radius_trace=_array(self.r2_radius_trace),
            local_energy_variance_trace=_array(self.local_energy_variance_trace),
            local_energy_median_trace=_array(self.local_energy_median_trace),
            local_energy_mad_trace=_array(self.local_energy_mad_trace),
            local_energy_p001_trace=_array(self.local_energy_p001_trace),
            local_energy_p01_trace=_array(self.local_energy_p01_trace),
            local_energy_p99_trace=_array(self.local_energy_p99_trace),
            local_energy_p999_trace=_array(self.local_energy_p999_trace),
            log_weight_span_trace=_array(self.log_weight_span_trace),
            ess_fraction_trace=_array(self.ess_fraction_trace),
            invalid_proposal_fraction_trace=_array(self.invalid_proposal_fraction_trace),
            hard_wall_kill_fraction_trace=_array(self.hard_wall_kill_fraction_trace),
            local_acceptance_fraction_trace=_array(self.local_acceptance_fraction_trace),
            metropolis_rejection_fraction_trace=_array(self.metropolis_rejection_fraction_trace),
            drift_norm_max_trace=_array(self.drift_norm_max_trace),
            configuration_esjd_trace=_array(self.configuration_esjd_trace),
            r2_esjd_trace=_array(self.r2_esjd_trace),
            weighted_free_gap_esjd_trace=_array(self.weighted_free_gap_esjd_trace),
            weighted_free_gap_mean_trace=_array(self.weighted_free_gap_mean_trace),
            free_gap_min_trace=_array(self.free_gap_min_trace),
            free_gap_p01_trace=_array(self.free_gap_p01_trace),
            zero_weight_excluded_fraction_trace=_array(self.zero_weight_excluded_fraction_trace),
            retained_fraction_trace=_array(self.retained_fraction_trace),
        )
    def _record_batch(
        self,
        batch: StreamingBatchObservables,
        *,
        production_index: int,
        dt: float,
        system: OpenLineHardRodSystem,
    ) -> None:
        self.stored_batch_count += 1
        self.weight_denominator += batch["weight_sum"]
        self.energy_numerator += batch["energy_numerator"]
        self.r2_numerator += batch["r2_numerator"]
        self.finite_sample_count += batch["finite_sample_count"]
        self.valid_sample_count += batch["valid_sample_count"]
        self.included_sample_count += batch["included_sample_count"]
        self.total_sample_count += batch["total_sample_count"]
        self.energy_stats = self.energy_stats.update(batch["mixed_energy"])
        self.r2_stats = self.r2_stats.update(batch["r2_radius"])
        self.trace_times.append(production_index * dt)
        self.mixed_energy_trace.append(batch["mixed_energy"])
        self.rms_radius_trace.append(float(np.sqrt(batch["r2_radius"])))
        self.r2_radius_trace.append(batch["r2_radius"])
        self.local_energy_variance_trace.append(batch["local_energy_variance"])
        self.local_energy_median_trace.append(batch["local_energy_median"])
        self.local_energy_mad_trace.append(batch["local_energy_mad"])
        self.local_energy_p001_trace.append(batch["local_energy_p001"])
        self.local_energy_p01_trace.append(batch["local_energy_p01"])
        self.local_energy_p99_trace.append(batch["local_energy_p99"])
        self.local_energy_p999_trace.append(batch["local_energy_p999"])
        trace_values = self.interval_trace.to_trace_values(walker_count=self.positions.shape[0])
        self.log_weight_span_trace.append(trace_values["log_weight_span"])
        self.ess_fraction_trace.append(trace_values["ess_fraction"])
        self.invalid_proposal_fraction_trace.append(trace_values["invalid_proposal_fraction"])
        self.hard_wall_kill_fraction_trace.append(trace_values["hard_wall_kill_fraction"])
        self.local_acceptance_fraction_trace.append(trace_values["local_acceptance_fraction"])
        self.metropolis_rejection_fraction_trace.append(
            trace_values["metropolis_rejection_fraction"]
        )
        self.drift_norm_max_trace.append(trace_values["drift_norm_max"])
        self.configuration_esjd_trace.append(trace_values["configuration_esjd"])
        self.r2_esjd_trace.append(trace_values["r2_esjd"])
        self.weighted_free_gap_esjd_trace.append(trace_values["weighted_free_gap_esjd"])
        gap_diagnostics = free_gap_batch_diagnostics(
            batch["samples"],
            batch["normalized_weights"],
            rod_length=system.rod_length,
        )
        self.weighted_free_gap_mean_trace.append(gap_diagnostics["weighted_free_gap_mean"])
        self.free_gap_min_trace.append(gap_diagnostics["free_gap_min"])
        self.free_gap_p01_trace.append(gap_diagnostics["free_gap_p01"])
        self.zero_weight_excluded_fraction_trace.append(
            safe_fraction(
                batch["valid_sample_count"] - batch["included_sample_count"],
                batch["total_sample_count"],
            )
        )
        self.retained_fraction_trace.append(
            safe_fraction(batch["included_sample_count"], batch["total_sample_count"])
        )
        self.interval_trace = TraceAccumulator()
        self.density_histogram = self.density_histogram.update(
            batch["samples"].ravel(),
            np.repeat(batch["weights"], system.n_particles),
        )
    def _summary_metadata(
        self,
        *,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        ess_resample_fraction: float,
        guide: DMCGuide,
        finite_fraction: float,
        valid_fraction: float,
        included_fraction: float,
    ) -> dict[str, Any]:
        return {
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "local_step_count": self.local_step_count,
            "killed_count": self.killed_count,
            "resample_count": self.resample_count,
            "ess_min": float(np.min(self.ess_values)) if self.ess_values else float("nan"),
            "ess_mean": float(np.mean(self.ess_values)) if self.ess_values else float("nan"),
            "ess_resample_fraction": ess_resample_fraction,
            "ess_fraction_min": finite_min(self.ess_fraction_trace),
            "log_weight_span_max": finite_max(self.log_weight_span_trace),
            "invalid_proposal_fraction_max": finite_max(self.invalid_proposal_fraction_trace),
            "hard_wall_kill_fraction_max": finite_max(self.hard_wall_kill_fraction_trace),
            "local_acceptance_fraction_mean": finite_mean(self.local_acceptance_fraction_trace),
            "metropolis_rejection_fraction_max": finite_max(
                self.metropolis_rejection_fraction_trace
            ),
            "local_energy_median_mean": finite_mean(self.local_energy_median_trace),
            "local_energy_mad_mean": finite_mean(self.local_energy_mad_trace),
            "local_energy_p001_min": finite_min(self.local_energy_p001_trace),
            "local_energy_p01_min": finite_min(self.local_energy_p01_trace),
            "local_energy_p99_max": finite_max(self.local_energy_p99_trace),
            "local_energy_p999_max": finite_max(self.local_energy_p999_trace),
            "drift_norm_max": finite_max(self.drift_norm_max_trace),
            "configuration_esjd_mean": finite_mean(self.configuration_esjd_trace),
            "r2_esjd_mean": finite_mean(self.r2_esjd_trace),
            "weighted_free_gap_esjd_mean": finite_mean(self.weighted_free_gap_esjd_trace),
            "weighted_free_gap_mean_min": finite_min(self.weighted_free_gap_mean_trace),
            "weighted_free_gap_mean_max": finite_max(self.weighted_free_gap_mean_trace),
            "free_gap_min": finite_min(self.free_gap_min_trace),
            "free_gap_p01_min": finite_min(self.free_gap_p01_trace),
            "zero_weight_excluded_fraction_max": finite_max(
                self.zero_weight_excluded_fraction_trace
            ),
            "finite_local_energy_fraction": finite_fraction,
            "valid_snapshot_fraction": valid_fraction,
            "included_sample_fraction": included_fraction,
            "summary_mode": "streaming",
            "guide_batch_backend": guide_batch_backend(guide),
        }


def _array(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=float)
