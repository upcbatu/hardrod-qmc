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
from hrdmc.monte_carlo.dmc.rn_block import RNBlockStreamingSummary
from hrdmc.systems.harmonic_com_transition import harmonic_com_ground_variance
from hrdmc.workflows.dmc.rn_block import (
    DEFAULT_RN_GUIDE_FAMILY,
    DEFAULT_RN_PROPOSAL_FAMILY,
    DEFAULT_RN_TARGET_FAMILY,
    RNCase,
    RNCollectiveProposalControls,
    RNRunControls,
    run_streaming_seed,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls


@dataclass(frozen=True)
class PureWalkingSeedRun:
    seed: int
    rn_summary: RNBlockStreamingSummary
    pure_result: PureWalkingResult
    schema_reference: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.pure_result.status,
            "rn_summary": compact_rn_seed_summary(self.rn_summary),
            "pure_walking": self.pure_result.to_summary_dict(),
            "schema_reference": self.schema_reference,
        }


def run_pure_walking_seed(
    case: RNCase,
    controls: RNRunControls,
    seed: int,
    *,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    initialization: RNInitializationControls | None = None,
    proposal: RNCollectiveProposalControls | None = None,
    proposal_family: str = DEFAULT_RN_PROPOSAL_FAMILY,
    guide_family: str = DEFAULT_RN_GUIDE_FAMILY,
    target_family: str = DEFAULT_RN_TARGET_FAMILY,
) -> dict[str, Any]:
    """Run one RN-DMC seed while consuming transport events with pure FW."""

    return run_pure_walking_seed_run(
        case,
        controls,
        seed,
        pure_config=pure_config,
        density_grid=density_grid,
        progress=progress,
        initialization=initialization,
        proposal=proposal,
        proposal_family=proposal_family,
        guide_family=guide_family,
        target_family=target_family,
    ).to_payload()


def run_pure_walking_seed_run(
    case: RNCase,
    controls: RNRunControls,
    seed: int,
    *,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    initialization: RNInitializationControls | None = None,
    proposal: RNCollectiveProposalControls | None = None,
    proposal_family: str = DEFAULT_RN_PROPOSAL_FAMILY,
    guide_family: str = DEFAULT_RN_GUIDE_FAMILY,
    target_family: str = DEFAULT_RN_TARGET_FAMILY,
) -> PureWalkingSeedRun:
    """Run one RN-DMC seed and keep both RN summary and FW result."""

    observer = TransportedAuxiliaryForwardWalking(pure_config)
    summary = run_streaming_seed(
        case,
        controls,
        seed,
        density_grid=density_grid,
        progress=progress,
        initialization=initialization,
        proposal=proposal,
        proposal_family=proposal_family,
        guide_family=guide_family,
        target_family=target_family,
        transport_observer=observer,
        transport_com_variance=_transport_com_variance(case, pure_config),
    )
    mixed_r2_reference = summary.r2_radius if controls.store_every == 1 else None
    mixed_rms_reference = summary.rms_radius if controls.store_every == 1 else None
    pure_result = observer.result(
        mixed_r2_reference=mixed_r2_reference,
        mixed_rms_radius_reference=mixed_rms_reference,
    )
    return PureWalkingSeedRun(
        seed=seed,
        rn_summary=summary,
        pure_result=pure_result,
        schema_reference=(
            "rn_summary_store_every_1"
            if controls.store_every == 1
            else "internal_fw_event_stream; rn_summary cadence differs"
        ),
    )


def compact_rn_seed_summary(summary: RNBlockStreamingSummary) -> dict[str, Any]:
    return {
        "mixed_energy": summary.mixed_energy,
        "r2_radius": summary.r2_radius,
        "rms_radius": summary.rms_radius,
        "density_integral": summary.density_integral,
        "lost_out_of_grid_sample_count": summary.lost_out_of_grid_sample_count,
        "metadata": {
            "stored_batch_count": summary.stored_batch_count,
            "sample_count": summary.sample_count,
            "rn_event_count": summary.metadata.get("rn_event_count"),
            "local_step_count": summary.metadata.get("local_step_count"),
            "killed_count": summary.metadata.get("killed_count"),
            "resample_count": summary.metadata.get("resample_count"),
            "ess_fraction_min": summary.metadata.get("ess_fraction_min"),
            "log_weight_span_max": summary.metadata.get("log_weight_span_max"),
            "guide_batch_backend": summary.metadata.get("guide_batch_backend"),
            "target_backend": summary.metadata.get("target_backend"),
            "proposal_backend": summary.metadata.get("proposal_backend"),
            "resolved_guide_family": summary.metadata.get("resolved_guide_family"),
            "target_family": summary.metadata.get("target_family"),
        },
    }


def _transport_com_variance(case: RNCase, config: PureWalkingConfig) -> float | None:
    if config.observable_source != "r2_rb":
        return None
    return harmonic_com_ground_variance(case.n_particles, case.omega)
