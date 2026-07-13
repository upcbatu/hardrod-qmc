from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import (
    PureWalkingConfig,
    PureWalkingResult,
    TransportedAuxiliaryForwardWalking,
)
from hrdmc.io.progress import ProgressBar
from hrdmc.monte_carlo.dmc.local import DMCStreamingSummary
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    run_streaming_seed,
)


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


def run_pure_walking_seed(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Run one DMC seed while consuming transport events with pure FW."""

    return run_pure_walking_seed_run(
        case,
        controls,
        seed,
        pure_config=pure_config,
        density_grid=density_grid,
        progress=progress,
        initialization=initialization,
        collective_rn=collective_rn,
        guide_family=guide_family,
    ).to_payload()


def run_pure_walking_seed_run(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
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
        collective_rn=collective_rn,
        guide_family=guide_family,
        transport_observer=observer,
    )
    mixed_r2_reference = summary.r2_radius if controls.store_every == 1 else None
    mixed_rms_reference = summary.rms_radius if controls.store_every == 1 else None
    pure_result = observer.result(
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_reference,
    )
    return PureWalkingSeedRun(
        seed=seed,
        dmc_summary=summary,
        pure_result=pure_result,
        schema_reference=(
            "dmc_summary_store_every_1"
            if controls.store_every == 1
            else "internal_fw_event_stream; dmc_summary cadence differs"
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
            "scheduled_move_count": summary.metadata.get("scheduled_move_count"),
            "local_step_count": summary.metadata.get("local_step_count"),
            "killed_count": summary.metadata.get("killed_count"),
            "resample_count": summary.metadata.get("resample_count"),
            "ess_fraction_min": summary.metadata.get("ess_fraction_min"),
            "log_weight_span_max": summary.metadata.get("log_weight_span_max"),
            "local_step_method": summary.metadata.get("local_step_method"),
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
