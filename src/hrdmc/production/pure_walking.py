from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from hrdmc.artifacts.progress import ProgressBar
from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.forward_walking.results import PureWalkingResult
from hrdmc.estimators.forward_walking.transported import TransportedAuxiliaryForwardWalking
from hrdmc.sampling.dmc.results import DMCStreamingSummary
from hrdmc.sampling.dmc.run import run_streaming_seed
from hrdmc.sampling.initial_conditions import InitializationControls
from hrdmc.system.geometry import harmonic_com_ground_variance
from hrdmc.system.settings import DMCRunControls, TrappedCase
from hrdmc.trial.guide import DEFAULT_GUIDE_FAMILY


def pure_config_metadata(config: PureWalkingConfig) -> dict[str, Any]:
    return {
        "lag_steps": list(config.lag_steps),
        "density_lag_steps": (
            None if config.density_lag_steps is None else list(config.density_lag_steps)
        ),
        "lag_unit": config.lag_unit,
        "observables": list(config.observables),
        "observable_source": config.observable_source,
        "r2_rb_com_variance": config.r2_rb_com_variance,
        "density_source": config.density_source,
        "density_com_variance": config.density_com_variance,
        "density_parity_average": config.density_parity_average,
        "density_expected_particles": config.density_expected_particles,
        "density_accounting_abs_tolerance": config.density_accounting_abs_tolerance,
        "min_block_count": config.min_block_count,
        "min_walker_weight_ess": config.min_walker_weight_ess,
        "min_source_ancestor_ess": config.min_source_ancestor_ess,
        "max_source_family_fraction": config.max_source_family_fraction,
        "block_size_steps": config.block_size_steps,
        "collection_stride_steps": config.collection_stride_steps,
        "density_collection_stride_steps": config.density_collection_stride_steps,
        "transport_mode": config.transport_mode,
        "collection_mode": config.collection_mode,
        "center": config.center,
        "plateau_sigma_threshold": config.plateau_sigma_threshold,
        "rms_plateau_relative_tolerance": config.rms_plateau_relative_tolerance,
        "plateau_equivalence_confidence_level": (config.plateau_equivalence_confidence_level),
        "plateau_window_lag_count": config.plateau_window_lag_count,
        "density_plateau_window_lag_count": config.density_plateau_window_lag_count,
        "density_plateau_relative_l2_tolerance": (config.density_plateau_relative_l2_tolerance),
        "transport_invariant_tests_passed": list(config.transport_invariant_tests_passed),
    }
def pure_config_for_case(
    config: PureWalkingConfig,
    *,
    grid: np.ndarray,
    case: TrappedCase,
) -> PureWalkingConfig:
    resolved = config
    if "density" in resolved.observables and resolved.density_bin_edges is None:
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError("density grid must contain at least two centers")
        dx = float(grid[1] - grid[0])
        edges = np.concatenate(
            ([grid[0] - 0.5 * dx], 0.5 * (grid[:-1] + grid[1:]), [grid[-1] + 0.5 * dx])
        )
        resolved = replace(resolved, density_bin_edges=edges)
    if "density" in resolved.observables:
        expected_particles = float(case.n_particles)
        if resolved.density_expected_particles is not None and not np.isclose(
            resolved.density_expected_particles,
            expected_particles,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("density_expected_particles does not match the trapped case")
        resolved = replace(resolved, density_expected_particles=expected_particles)
    if resolved.observable_source == "r2_rb":
        expected_variance = harmonic_com_ground_variance(case.n_particles, case.omega)
        if resolved.r2_rb_com_variance is not None and not np.isclose(
            resolved.r2_rb_com_variance,
            expected_variance,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("r2_rb_com_variance does not match the trapped COM ground state")
        resolved = replace(resolved, r2_rb_com_variance=expected_variance)
    if resolved.density_source == "com_rao_blackwell":
        expected_variance = harmonic_com_ground_variance(case.n_particles, case.omega)
        if resolved.density_com_variance is not None and not np.isclose(
            resolved.density_com_variance,
            expected_variance,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("density_com_variance does not match the trapped COM ground state")
        resolved = replace(resolved, density_com_variance=expected_variance)
    resolved.validate()
    return resolved

@dataclass(frozen=True)
class PureWalkingSeedRun:
    seed: int
    dmc_summary: DMCStreamingSummary
    pure_result: PureWalkingResult
    schema_reference: str
    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.pure_result.status,
            "dmc_summary": compact_dmc_seed_summary(self.dmc_summary),
            "pure_walking": self.pure_result.to_summary_dict(),
            "schema_reference": self.schema_reference,
        }
def run_pure_walking_seed_run(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    initialization: InitializationControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> PureWalkingSeedRun:
    """Run one DMC seed and keep both the DMC summary and FW result."""
    observer = TransportedAuxiliaryForwardWalking(pure_config)
    summary = run_streaming_seed(
        case,
        controls,
        seed,
        density_grid=density_grid,
        progress=progress,
        initialization=initialization,
        guide_family=guide_family,
        transport_observer=observer,
    )
    use_stored_raw_reference = (
        controls.store_every == 1 and pure_config.observable_source == "raw_r2"
    )
    mixed_r2_reference = summary.r2_radius if use_stored_raw_reference else None
    mixed_rms_reference = summary.rms_radius if use_stored_raw_reference else None
    pure_result = observer.result(
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_reference,
    )
    return PureWalkingSeedRun(
        seed=seed,
        dmc_summary=summary,
        pure_result=pure_result,
        schema_reference=(
            "dmc_summary_store_every_1_raw_r2"
            if use_stored_raw_reference
            else (
                "internal_fw_event_stream_r2_rb"
                if pure_config.observable_source == "r2_rb"
                else "internal_fw_event_stream; dmc_summary cadence differs"
            )
        ),
    )
def compact_dmc_seed_summary(summary: DMCStreamingSummary) -> dict[str, Any]:
    return {
        "mixed_energy": summary.mixed_energy,
        "r2_radius": summary.r2_radius,
        "rms_radius": summary.rms_radius,
        "density_integral": summary.density_integral,
        "lost_out_of_grid_sample_count": summary.lost_out_of_grid_sample_count,
        "metadata": {
            "stored_batch_count": summary.stored_batch_count,
            "sample_count": summary.sample_count,
            "local_step_count": summary.metadata.get("local_step_count"),
            "killed_count": summary.metadata.get("killed_count"),
            "resample_count": summary.metadata.get("resample_count"),
            "ess_fraction_min": summary.metadata.get("ess_fraction_min"),
            "log_weight_span_max": summary.metadata.get("log_weight_span_max"),
            "drift_limiter": summary.metadata.get("drift_limiter"),
            "local_acceptance_fraction_mean": summary.metadata.get(
                "local_acceptance_fraction_mean"
            ),
            "invalid_proposal_fraction_max": summary.metadata.get("invalid_proposal_fraction_max"),
            "metropolis_rejection_fraction_max": summary.metadata.get(
                "metropolis_rejection_fraction_max"
            ),
            "local_energy_median_mean": summary.metadata.get("local_energy_median_mean"),
            "local_energy_mad_mean": summary.metadata.get("local_energy_mad_mean"),
            "local_energy_p001_min": summary.metadata.get("local_energy_p001_min"),
            "local_energy_p01_min": summary.metadata.get("local_energy_p01_min"),
            "local_energy_p99_max": summary.metadata.get("local_energy_p99_max"),
            "local_energy_p999_max": summary.metadata.get("local_energy_p999_max"),
            "drift_norm_max": summary.metadata.get("drift_norm_max"),
            "configuration_esjd_mean": summary.metadata.get("configuration_esjd_mean"),
            "r2_esjd_mean": summary.metadata.get("r2_esjd_mean"),
            "weighted_free_gap_esjd_mean": summary.metadata.get("weighted_free_gap_esjd_mean"),
            "weighted_free_gap_mean_min": summary.metadata.get("weighted_free_gap_mean_min"),
            "weighted_free_gap_mean_max": summary.metadata.get("weighted_free_gap_mean_max"),
            "free_gap_min": summary.metadata.get("free_gap_min"),
            "free_gap_p01_min": summary.metadata.get("free_gap_p01_min"),
            "guide_batch_backend": summary.metadata.get("guide_batch_backend"),
            "target_backend": summary.metadata.get("target_backend"),
            "proposal_backend": summary.metadata.get("proposal_backend"),
            "resolved_guide_family": summary.metadata.get("resolved_guide_family"),
            "target_family": summary.metadata.get("target_family"),
        },
    }
