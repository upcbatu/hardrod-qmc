from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.artifacts import config_fingerprint
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.estimators.mixed.streaming import (
    StreamingBatchObservables,
    streaming_batch_observables,
)
from hrdmc.monte_carlo.dmc.common.guide_api import evaluate_guide, guide_batch_backend, valid_rows
from hrdmc.monte_carlo.dmc.common.numeric import (
    finite_max,
    finite_mean,
    finite_min,
    log_weight_span,
    safe_fraction,
)
from hrdmc.monte_carlo.dmc.common.population import normalize_log_weights
from hrdmc.monte_carlo.dmc.local.checkpoint import (
    load_streaming_checkpoint,
    save_streaming_checkpoint,
)
from hrdmc.monte_carlo.dmc.local.mobility import free_gap_batch_diagnostics
from hrdmc.monte_carlo.dmc.local.results import DMCStreamingSummary
from hrdmc.monte_carlo.dmc.local.telemetry import DMCStepTelemetry, TraceAccumulator
from hrdmc.numerics import RunningHistogram, RunningStats
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.api import DMCGuide

FloatArray = NDArray[np.float64]


@dataclass
class DMCStreamingState:
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
    scheduled_move_count: int = 0
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
    scheduled_log_target_mean_trace: list[float] = field(default_factory=list)
    scheduled_log_proposal_mean_trace: list[float] = field(default_factory=list)
    scheduled_log_weight_increment_mean_trace: list[float] = field(default_factory=list)
    scheduled_log_weight_increment_variance_trace: list[float] = field(default_factory=list)
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

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        rng: np.random.Generator,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        scheduled_move_interval_steps: int,
        scheduled_move_enabled: bool,
        scheduled_move_name: str | None,
        system: OpenLineHardRodSystem,
        density_grid: FloatArray,
        resume_identity: dict[str, Any],
    ) -> DMCStreamingState:
        metadata, arrays = load_streaming_checkpoint(path)
        _validate_checkpoint(
            metadata,
            arrays,
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            scheduled_move_interval_steps=scheduled_move_interval_steps,
            scheduled_move_enabled=scheduled_move_enabled,
            scheduled_move_name=scheduled_move_name,
            system=system,
            density_grid=density_grid,
            resume_identity=resume_identity,
        )
        rng.bit_generator.state = metadata["rng_state"]
        return cls(
            step_start=int(metadata["step_index"]) + 1,
            positions=arrays["positions"].copy(),
            local_energies=arrays["local_energies"].copy(),
            log_weights=arrays["log_weights"].copy(),
            density_grid=np.asarray(density_grid, dtype=float),
            density_histogram=RunningHistogram(
                bin_edges=arrays["density_bin_edges"].copy(),
                counts=arrays["density_counts"].copy(),
                sample_count=int(metadata["density_sample_count"]),
                total_weight=float(metadata["density_total_weight"]),
                lost_sample_count=int(metadata["density_lost_sample_count"]),
                lost_weight=float(metadata["density_lost_weight"]),
            ),
            energy_stats=RunningStats(
                count=int(metadata["energy_stats_count"]),
                mean=float(metadata["energy_stats_mean"]),
                m2=float(metadata["energy_stats_m2"]),
            ),
            r2_stats=RunningStats(
                count=int(metadata["r2_stats_count"]),
                mean=float(metadata["r2_stats_mean"]),
                m2=float(metadata["r2_stats_m2"]),
            ),
            energy_numerator=float(metadata["energy_numerator"]),
            r2_numerator=float(metadata["r2_numerator"]),
            weight_denominator=float(metadata["weight_denominator"]),
            stored_batch_count=int(metadata["stored_batch_count"]),
            scheduled_move_count=int(metadata["scheduled_move_count"]),
            local_step_count=int(metadata["local_step_count"]),
            killed_count=int(metadata["killed_count"]),
            resample_count=int(metadata["resample_count"]),
            finite_sample_count=int(metadata["finite_sample_count"]),
            valid_sample_count=int(metadata["valid_sample_count"]),
            included_sample_count=int(metadata["included_sample_count"]),
            total_sample_count=int(metadata["total_sample_count"]),
            ess_values=_list(arrays, "ess_values"),
            trace_times=_list(arrays, "trace_times"),
            mixed_energy_trace=_list(arrays, "mixed_energy_trace"),
            rms_radius_trace=_list(arrays, "rms_radius_trace"),
            r2_radius_trace=_list(arrays, "r2_radius_trace"),
            local_energy_variance_trace=_list(arrays, "local_energy_variance_trace"),
            local_energy_median_trace=_list(arrays, "local_energy_median_trace"),
            local_energy_mad_trace=_list(arrays, "local_energy_mad_trace"),
            local_energy_p001_trace=_list(arrays, "local_energy_p001_trace"),
            local_energy_p01_trace=_list(arrays, "local_energy_p01_trace"),
            local_energy_p99_trace=_list(arrays, "local_energy_p99_trace"),
            local_energy_p999_trace=_list(arrays, "local_energy_p999_trace"),
            log_weight_span_trace=_list(arrays, "log_weight_span_trace"),
            ess_fraction_trace=_list(arrays, "ess_fraction_trace"),
            invalid_proposal_fraction_trace=_list(arrays, "invalid_proposal_fraction_trace"),
            hard_wall_kill_fraction_trace=_list(arrays, "hard_wall_kill_fraction_trace"),
            local_acceptance_fraction_trace=_list(arrays, "local_acceptance_fraction_trace"),
            metropolis_rejection_fraction_trace=_list(
                arrays,
                "metropolis_rejection_fraction_trace",
            ),
            drift_norm_max_trace=_list(arrays, "drift_norm_max_trace"),
            configuration_esjd_trace=_list(arrays, "configuration_esjd_trace"),
            r2_esjd_trace=_list(arrays, "r2_esjd_trace"),
            weighted_free_gap_esjd_trace=_list(arrays, "weighted_free_gap_esjd_trace"),
            weighted_free_gap_mean_trace=_list(arrays, "weighted_free_gap_mean_trace"),
            free_gap_min_trace=_list(arrays, "free_gap_min_trace"),
            free_gap_p01_trace=_list(arrays, "free_gap_p01_trace"),
            zero_weight_excluded_fraction_trace=_list(
                arrays,
                "zero_weight_excluded_fraction_trace",
            ),
            scheduled_log_target_mean_trace=_list(
                arrays,
                "scheduled_log_target_mean_trace",
            ),
            scheduled_log_proposal_mean_trace=_list(
                arrays,
                "scheduled_log_proposal_mean_trace",
            ),
            scheduled_log_weight_increment_mean_trace=_list(
                arrays,
                "scheduled_log_weight_increment_mean_trace",
            ),
            scheduled_log_weight_increment_variance_trace=_list(
                arrays,
                "scheduled_log_weight_increment_variance_trace",
            ),
            retained_fraction_trace=_list(arrays, "retained_fraction_trace"),
            interval_trace=TraceAccumulator(
                step_count=int(metadata["interval_trace_step_count"]),
                killed_count=int(metadata["interval_trace_killed_count"]),
                ess_fraction_sum=float(metadata["interval_trace_ess_fraction_sum"]),
                log_weight_span_sum=float(metadata["interval_trace_log_weight_span_sum"]),
                scheduled_log_target_values=_list(
                    arrays,
                    "interval_trace_scheduled_log_target_values",
                ),
                scheduled_log_proposal_values=_list(
                    arrays,
                    "interval_trace_scheduled_log_proposal_values",
                ),
                scheduled_log_weight_increment_values=_list(
                    arrays,
                    "interval_trace_scheduled_log_weight_increment_values",
                ),
                scheduled_log_weight_increment_variance_values=_list(
                    arrays,
                    "interval_trace_scheduled_log_weight_increment_variance_values",
                ),
                local_acceptance_values=_list(arrays, "interval_trace_local_acceptance_values"),
                invalid_proposal_values=_list(arrays, "interval_trace_invalid_proposal_values"),
                metropolis_rejection_values=_list(
                    arrays,
                    "interval_trace_metropolis_rejection_values",
                ),
                drift_norm_max_values=_list(
                    arrays,
                    "interval_trace_drift_norm_max_values",
                ),
                configuration_esjd_values=_list(
                    arrays,
                    "interval_trace_configuration_esjd_values",
                ),
                r2_esjd_values=_list(arrays, "interval_trace_r2_esjd_values"),
                weighted_free_gap_esjd_values=_list(
                    arrays,
                    "interval_trace_weighted_free_gap_esjd_values",
                ),
            ),
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

    def save_checkpoint(
        self,
        path: Path,
        *,
        step_index: int,
        rng: np.random.Generator,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        scheduled_move_interval_steps: int,
        scheduled_move_enabled: bool,
        scheduled_move_name: str | None,
        system: OpenLineHardRodSystem,
        resume_identity: dict[str, Any],
    ) -> None:
        save_streaming_checkpoint(
            path,
            metadata=self._checkpoint_metadata(
                step_index=step_index,
                rng=rng,
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                scheduled_move_interval_steps=scheduled_move_interval_steps,
                scheduled_move_enabled=scheduled_move_enabled,
                scheduled_move_name=scheduled_move_name,
                system=system,
                resume_identity=resume_identity,
            ),
            arrays=self._checkpoint_arrays(),
        )

    def to_summary(
        self,
        *,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        scheduled_move_interval_steps: int,
        scheduled_move_enabled: bool,
        ess_resample_fraction: float,
        guide: DMCGuide,
        scheduled_move_metadata: dict[str, Any],
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
                scheduled_move_interval_steps=scheduled_move_interval_steps,
                scheduled_move_enabled=scheduled_move_enabled,
                ess_resample_fraction=ess_resample_fraction,
                guide=guide,
                scheduled_move_metadata=scheduled_move_metadata,
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
            scheduled_log_target_mean_trace=_array(self.scheduled_log_target_mean_trace),
            scheduled_log_proposal_mean_trace=_array(self.scheduled_log_proposal_mean_trace),
            scheduled_log_weight_increment_mean_trace=_array(
                self.scheduled_log_weight_increment_mean_trace
            ),
            scheduled_log_weight_increment_variance_trace=_array(
                self.scheduled_log_weight_increment_variance_trace
            ),
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
        self.scheduled_log_target_mean_trace.append(trace_values["scheduled_log_target_mean"])
        self.scheduled_log_proposal_mean_trace.append(trace_values["scheduled_log_proposal_mean"])
        self.scheduled_log_weight_increment_mean_trace.append(
            trace_values["scheduled_log_weight_increment_mean"]
        )
        self.scheduled_log_weight_increment_variance_trace.append(
            trace_values["scheduled_log_weight_increment_variance"]
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
        scheduled_move_interval_steps: int,
        scheduled_move_enabled: bool,
        ess_resample_fraction: float,
        guide: DMCGuide,
        scheduled_move_metadata: dict[str, Any],
        finite_fraction: float,
        valid_fraction: float,
        included_fraction: float,
    ) -> dict[str, Any]:
        return {
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "scheduled_move_interval_steps": scheduled_move_interval_steps,
            "scheduled_move_enabled": scheduled_move_enabled,
            "scheduled_move_count": self.scheduled_move_count,
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
            "scheduled_log_weight_increment_variance_max": finite_max(
                self.scheduled_log_weight_increment_variance_trace
            ),
            "finite_local_energy_fraction": finite_fraction,
            "valid_snapshot_fraction": valid_fraction,
            "included_sample_fraction": included_fraction,
            "summary_mode": "streaming",
            "guide_batch_backend": guide_batch_backend(guide),
            **scheduled_move_metadata,
        }

    def _checkpoint_metadata(
        self,
        *,
        step_index: int,
        rng: np.random.Generator,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        scheduled_move_interval_steps: int,
        scheduled_move_enabled: bool,
        scheduled_move_name: str | None,
        system: OpenLineHardRodSystem,
        resume_identity: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_identity = to_jsonable(resume_identity)
        if not isinstance(normalized_identity, dict):
            raise TypeError("resume identity must normalize to a mapping")
        return {
            "step_index": step_index,
            "rng_state": rng.bit_generator.state,
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "scheduled_move_interval_steps": scheduled_move_interval_steps,
            "scheduled_move_enabled": scheduled_move_enabled,
            "scheduled_move_name": scheduled_move_name,
            "walker_count": int(self.positions.shape[0]),
            "n_particles": int(system.n_particles),
            "resume_identity": normalized_identity,
            "resume_identity_sha256": config_fingerprint(normalized_identity),
            "energy_numerator": self.energy_numerator,
            "r2_numerator": self.r2_numerator,
            "weight_denominator": self.weight_denominator,
            "stored_batch_count": self.stored_batch_count,
            "scheduled_move_count": self.scheduled_move_count,
            "local_step_count": self.local_step_count,
            "killed_count": self.killed_count,
            "resample_count": self.resample_count,
            "finite_sample_count": self.finite_sample_count,
            "valid_sample_count": self.valid_sample_count,
            "included_sample_count": self.included_sample_count,
            "total_sample_count": self.total_sample_count,
            "density_sample_count": self.density_histogram.sample_count,
            "density_total_weight": self.density_histogram.total_weight,
            "density_lost_sample_count": self.density_histogram.lost_sample_count,
            "density_lost_weight": self.density_histogram.lost_weight,
            "energy_stats_count": self.energy_stats.count,
            "energy_stats_mean": self.energy_stats.mean,
            "energy_stats_m2": self.energy_stats.m2,
            "r2_stats_count": self.r2_stats.count,
            "r2_stats_mean": self.r2_stats.mean,
            "r2_stats_m2": self.r2_stats.m2,
            "interval_trace_step_count": self.interval_trace.step_count,
            "interval_trace_killed_count": self.interval_trace.killed_count,
            "interval_trace_ess_fraction_sum": self.interval_trace.ess_fraction_sum,
            "interval_trace_log_weight_span_sum": self.interval_trace.log_weight_span_sum,
        }

    def _checkpoint_arrays(self) -> dict[str, FloatArray]:
        return {
            "density_grid": self.density_grid,
            "positions": self.positions,
            "local_energies": self.local_energies,
            "log_weights": self.log_weights,
            "density_bin_edges": self.density_histogram.bin_edges,
            "density_counts": self.density_histogram.counts,
            "ess_values": _array(self.ess_values),
            "trace_times": _array(self.trace_times),
            "mixed_energy_trace": _array(self.mixed_energy_trace),
            "rms_radius_trace": _array(self.rms_radius_trace),
            "r2_radius_trace": _array(self.r2_radius_trace),
            "local_energy_variance_trace": _array(self.local_energy_variance_trace),
            "local_energy_median_trace": _array(self.local_energy_median_trace),
            "local_energy_mad_trace": _array(self.local_energy_mad_trace),
            "local_energy_p001_trace": _array(self.local_energy_p001_trace),
            "local_energy_p01_trace": _array(self.local_energy_p01_trace),
            "local_energy_p99_trace": _array(self.local_energy_p99_trace),
            "local_energy_p999_trace": _array(self.local_energy_p999_trace),
            "log_weight_span_trace": _array(self.log_weight_span_trace),
            "ess_fraction_trace": _array(self.ess_fraction_trace),
            "invalid_proposal_fraction_trace": _array(self.invalid_proposal_fraction_trace),
            "hard_wall_kill_fraction_trace": _array(self.hard_wall_kill_fraction_trace),
            "local_acceptance_fraction_trace": _array(self.local_acceptance_fraction_trace),
            "metropolis_rejection_fraction_trace": _array(self.metropolis_rejection_fraction_trace),
            "drift_norm_max_trace": _array(self.drift_norm_max_trace),
            "configuration_esjd_trace": _array(self.configuration_esjd_trace),
            "r2_esjd_trace": _array(self.r2_esjd_trace),
            "weighted_free_gap_esjd_trace": _array(self.weighted_free_gap_esjd_trace),
            "weighted_free_gap_mean_trace": _array(self.weighted_free_gap_mean_trace),
            "free_gap_min_trace": _array(self.free_gap_min_trace),
            "free_gap_p01_trace": _array(self.free_gap_p01_trace),
            "zero_weight_excluded_fraction_trace": _array(self.zero_weight_excluded_fraction_trace),
            "scheduled_log_target_mean_trace": _array(self.scheduled_log_target_mean_trace),
            "scheduled_log_proposal_mean_trace": _array(self.scheduled_log_proposal_mean_trace),
            "scheduled_log_weight_increment_mean_trace": _array(
                self.scheduled_log_weight_increment_mean_trace
            ),
            "scheduled_log_weight_increment_variance_trace": _array(
                self.scheduled_log_weight_increment_variance_trace
            ),
            "retained_fraction_trace": _array(self.retained_fraction_trace),
            "interval_trace_scheduled_log_target_values": _array(
                self.interval_trace.scheduled_log_target_values
            ),
            "interval_trace_scheduled_log_proposal_values": _array(
                self.interval_trace.scheduled_log_proposal_values
            ),
            "interval_trace_scheduled_log_weight_increment_values": _array(
                self.interval_trace.scheduled_log_weight_increment_values
            ),
            "interval_trace_scheduled_log_weight_increment_variance_values": _array(
                self.interval_trace.scheduled_log_weight_increment_variance_values
            ),
            "interval_trace_local_acceptance_values": _array(
                self.interval_trace.local_acceptance_values
            ),
            "interval_trace_invalid_proposal_values": _array(
                self.interval_trace.invalid_proposal_values
            ),
            "interval_trace_metropolis_rejection_values": _array(
                self.interval_trace.metropolis_rejection_values
            ),
            "interval_trace_drift_norm_max_values": _array(
                self.interval_trace.drift_norm_max_values
            ),
            "interval_trace_configuration_esjd_values": _array(
                self.interval_trace.configuration_esjd_values
            ),
            "interval_trace_r2_esjd_values": _array(self.interval_trace.r2_esjd_values),
            "interval_trace_weighted_free_gap_esjd_values": _array(
                self.interval_trace.weighted_free_gap_esjd_values
            ),
        }


def _array(values: list[float]) -> FloatArray:
    return np.asarray(values, dtype=float)


def _list(
    arrays: dict[str, np.ndarray],
    key: str,
) -> list[float]:
    if key not in arrays:
        raise KeyError(f"checkpoint array is missing: {key}")
    return arrays[key].astype(float).tolist()


def _validate_checkpoint(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
    scheduled_move_interval_steps: int,
    scheduled_move_enabled: bool,
    scheduled_move_name: str | None,
    system: OpenLineHardRodSystem,
    density_grid: FloatArray,
    resume_identity: dict[str, Any],
) -> None:
    observed_identity = metadata.get("resume_identity")
    observed_fingerprint = metadata.get("resume_identity_sha256")
    if not isinstance(observed_identity, dict) or not isinstance(observed_fingerprint, str):
        source_schema = metadata.get("source_schema_version", metadata.get("schema_version"))
        raise ValueError(
            "checkpoint lacks a canonical resume identity and cannot be resumed safely: "
            f"{source_schema}"
        )
    if config_fingerprint(observed_identity) != observed_fingerprint:
        raise ValueError("checkpoint resume identity fingerprint is invalid")
    normalized_expected = to_jsonable(resume_identity)
    if not isinstance(normalized_expected, dict):
        raise TypeError("resume identity must normalize to a mapping")
    if observed_fingerprint != config_fingerprint(normalized_expected):
        raise ValueError("checkpoint resume identity does not match the requested run")
    expected_scalars = {
        "dt": dt,
        "burn_in_steps": burn_in_steps,
        "production_steps": production_steps,
        "store_every": store_every,
        "n_particles": system.n_particles,
    }
    for key, expected in expected_scalars.items():
        observed = metadata.get(key)
        if observed != expected:
            raise ValueError(f"checkpoint {key}={observed!r} does not match {expected!r}")
    observed_move_enabled = bool(metadata["scheduled_move_enabled"])
    if observed_move_enabled != scheduled_move_enabled:
        raise ValueError("checkpoint scheduled-move setting does not match the requested run")
    if scheduled_move_enabled:
        observed_interval = int(metadata["scheduled_move_interval_steps"])
        if observed_interval != scheduled_move_interval_steps:
            raise ValueError("checkpoint scheduled-move interval does not match the requested run")
    observed_move_name = metadata.get("scheduled_move_name")
    if observed_move_name != scheduled_move_name:
        raise ValueError("checkpoint scheduled-move name does not match the requested run")
    positions = arrays.get("positions")
    local_energies = arrays.get("local_energies")
    log_weights = arrays.get("log_weights")
    saved_grid = arrays.get("density_grid")
    if positions is None or positions.ndim != 2 or positions.shape[1] != system.n_particles:
        raise ValueError("checkpoint positions have incompatible shape")
    if local_energies is None or local_energies.shape != (positions.shape[0],):
        raise ValueError("checkpoint local energies have incompatible shape")
    if log_weights is None or log_weights.shape != (positions.shape[0],):
        raise ValueError("checkpoint log weights have incompatible shape")
    if saved_grid is None or not np.array_equal(saved_grid, density_grid):
        raise ValueError("checkpoint density grid does not match requested run")
    step_index = int(metadata.get("step_index", -1))
    total_steps = burn_in_steps + production_steps
    if step_index < 0 or step_index > total_steps:
        raise ValueError("checkpoint step index is outside this run")
