from __future__ import annotations

from typing import Any

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io.progress import ProgressBar
from hrdmc.theory import trapped_n2_finite_a_reference
from hrdmc.workflows.anchors.finite_a_n2.comparison import (
    FiniteAN2ReferenceTolerances,
    finite_a_n2_reference_comparison,
)
from hrdmc.workflows.dmc.benchmark_packet import summarize_benchmark_packet_case
from hrdmc.workflows.dmc.rn_block import (
    DEFAULT_RN_GUIDE_FAMILY,
    DEFAULT_RN_PROPOSAL_FAMILY,
    DEFAULT_RN_TARGET_FAMILY,
    RNCase,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls


def summarize_finite_a_n2_reference_case(
    case: RNCase,
    controls: RNRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    tolerances: FiniteAN2ReferenceTolerances,
    reference_grid_points: int = 1400,
    reference_y_max: float | None = None,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    trace_output_dir: Any | None = None,
    ess_warning_fraction: float = 0.20,
    ess_no_go_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: RNInitializationControls | None = None,
    proposal: RNCollectiveProposalControls | None = None,
    proposal_family: str = DEFAULT_RN_PROPOSAL_FAMILY,
    guide_family: str = DEFAULT_RN_GUIDE_FAMILY,
    target_family: str = DEFAULT_RN_TARGET_FAMILY,
) -> dict[str, Any]:
    """Run the production finite-a packet and compare it to the N=2 reference."""

    if case.n_particles != 2:
        raise ValueError("finite-a N=2 reference case requires N=2")
    if case.rod_length <= 0.0:
        raise ValueError("finite-a N=2 reference case requires rod_length > 0")
    reference = trapped_n2_finite_a_reference(
        rod_length=case.rod_length,
        omega=case.omega,
        grid_points=reference_grid_points,
        y_max=reference_y_max,
    )
    benchmark = summarize_benchmark_packet_case(
        case,
        controls,
        seeds,
        pure_config=pure_config,
        parallel_workers=parallel_workers,
        progress=progress,
        trace_output_dir=trace_output_dir,
        ess_warning_fraction=ess_warning_fraction,
        ess_no_go_fraction=ess_no_go_fraction,
        log_weight_span_warning=log_weight_span_warning,
        initialization=initialization,
        proposal=proposal,
        proposal_family=proposal_family,
        guide_family=guide_family,
        target_family=target_family,
    )
    comparison = finite_a_n2_reference_comparison(
        benchmark,
        reference,
        tolerances=tolerances,
    )
    return {
        "schema_version": "finite_a_n2_reference_case_v1",
        "status": comparison["status"],
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "omega": case.omega,
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": benchmark.get("parallel_workers"),
        "parallel_workers_requested": benchmark.get("parallel_workers_requested"),
        "reference_grid_points": reference_grid_points,
        "reference_y_max": reference_y_max,
        "initialization_mode": benchmark.get("initialization_mode"),
        "proposal_family": proposal_family,
        "guide_family": guide_family,
        "target_family": target_family,
        "component_log_scales": benchmark.get("component_log_scales"),
        "component_probabilities": benchmark.get("component_probabilities"),
        "reference": reference.to_metadata(),
        "comparison": comparison,
        "benchmark_packet": benchmark,
        "claim_boundary": comparison["claim_boundary"],
    }
