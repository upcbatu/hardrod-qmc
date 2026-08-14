from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from hrdmc.artifacts.manifest import config_fingerprint, ensure_dir
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.sampling.dmc.telemetry import TraceAccumulator
from hrdmc.statistics.streaming import RunningHistogram, RunningStats
from hrdmc.system.geometry import OpenLineHardRodSystem

FloatArray = NDArray[np.float64]


def _save_streaming_checkpoint(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    tmp = target.with_name(f".{target.name}.tmp")
    savez_compressed = cast(Any, np.savez_compressed)
    archive_arrays: dict[str, Any] = {
        "checkpoint_metadata": np.asarray(json.dumps(to_jsonable(metadata), allow_nan=True)),
        **arrays,
    }
    savez_compressed(tmp, **archive_arrays)
    npz_tmp = tmp.with_suffix(tmp.suffix + ".npz")
    npz_tmp.replace(target)
    return target


def _load_streaming_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["checkpoint_metadata"].item()))
        arrays = {
            key: np.asarray(archive[key]) for key in archive.files if key != "checkpoint_metadata"
        }
    return metadata, arrays


class DMCCheckpointMixin:
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
        system: OpenLineHardRodSystem,
        density_grid: FloatArray,
        resume_identity: dict[str, Any],
    ) -> Any:
        metadata, arrays = _load_streaming_checkpoint(path)
        _validate_checkpoint(
            metadata,
            arrays,
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            system=system,
            density_grid=density_grid,
            resume_identity=resume_identity,
        )
        rng.bit_generator.state = metadata["rng_state"]
        constructor = cast(Any, cls)
        return constructor(
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
            retained_fraction_trace=_list(arrays, "retained_fraction_trace"),
            interval_trace=TraceAccumulator(
                step_count=int(metadata["interval_trace_step_count"]),
                killed_count=int(metadata["interval_trace_killed_count"]),
                ess_fraction_sum=float(metadata["interval_trace_ess_fraction_sum"]),
                log_weight_span_sum=float(metadata["interval_trace_log_weight_span_sum"]),
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

    def save_checkpoint(
        self: Any,
        path: Path,
        *,
        step_index: int,
        rng: np.random.Generator,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
        system: OpenLineHardRodSystem,
        resume_identity: dict[str, Any],
    ) -> None:
        _save_streaming_checkpoint(
            path,
            metadata=self._checkpoint_metadata(
                step_index=step_index,
                rng=rng,
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                system=system,
                resume_identity=resume_identity,
            ),
            arrays=self._checkpoint_arrays(),
        )

    def _checkpoint_metadata(
        self: Any,
        *,
        step_index: int,
        rng: np.random.Generator,
        dt: float,
        burn_in_steps: int,
        production_steps: int,
        store_every: int,
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
            "walker_count": int(self.positions.shape[0]),
            "n_particles": int(system.n_particles),
            "resume_identity": normalized_identity,
            "resume_identity_sha256": config_fingerprint(normalized_identity),
            "energy_numerator": self.energy_numerator,
            "r2_numerator": self.r2_numerator,
            "weight_denominator": self.weight_denominator,
            "stored_batch_count": self.stored_batch_count,
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

    def _checkpoint_arrays(self: Any) -> dict[str, FloatArray]:
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
            "retained_fraction_trace": _array(self.retained_fraction_trace),
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
    system: OpenLineHardRodSystem,
    density_grid: FloatArray,
    resume_identity: dict[str, Any],
) -> None:
    _validate_checkpoint_identity(metadata, resume_identity)
    _validate_checkpoint_scalars(
        metadata,
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        n_particles=system.n_particles,
    )
    _validate_checkpoint_arrays(arrays, system, density_grid)
    step_index = int(metadata.get("step_index", -1))
    total_steps = burn_in_steps + production_steps
    if step_index < 0 or step_index > total_steps:
        raise ValueError("checkpoint step index is outside this run")


def _validate_checkpoint_identity(
    metadata: dict[str, Any], resume_identity: dict[str, Any]
) -> None:
    observed_identity = metadata.get("resume_identity")
    observed_fingerprint = metadata.get("resume_identity_sha256")
    if not isinstance(observed_identity, dict) or not isinstance(observed_fingerprint, str):
        raise ValueError("checkpoint lacks a resume identity and cannot be resumed safely")
    if config_fingerprint(observed_identity) != observed_fingerprint:
        raise ValueError("checkpoint resume identity fingerprint is invalid")
    normalized_expected = to_jsonable(resume_identity)
    if not isinstance(normalized_expected, dict):
        raise TypeError("resume identity must normalize to a mapping")
    if observed_fingerprint != config_fingerprint(normalized_expected):
        raise ValueError("checkpoint resume identity does not match the requested run")


def _validate_checkpoint_scalars(
    metadata: dict[str, Any],
    *,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
    n_particles: int,
) -> None:
    expected_scalars = {
        "dt": dt,
        "burn_in_steps": burn_in_steps,
        "production_steps": production_steps,
        "store_every": store_every,
        "n_particles": n_particles,
    }
    for key, expected in expected_scalars.items():
        observed = metadata.get(key)
        if observed != expected:
            raise ValueError(f"checkpoint {key}={observed!r} does not match {expected!r}")


def _validate_checkpoint_arrays(
    arrays: dict[str, np.ndarray],
    system: OpenLineHardRodSystem,
    density_grid: FloatArray,
) -> np.ndarray:
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
    return positions
