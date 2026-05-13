from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.runners import run_seed_batch
from hrdmc.workflows.dmc.pure_walking.seed import run_pure_walking_seed
from hrdmc.workflows.dmc.rn_block import (
    RNCase,
    RNCollectiveProposalControls,
    RNRunControls,
    controls_to_dict,
    make_grid,
    resolve_parallel_workers,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions import RNInitializationControls


def summarize_pure_walking_case(
    case: RNCase,
    controls: RNRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    initialization: RNInitializationControls | None = None,
    proposal: RNCollectiveProposalControls | None = None,
    proposal_family: str = "harmonic-mehler",
    guide_family: str = "auto",
) -> dict[str, Any]:
    """Run a reproducible transported-FW case packet."""

    initialization = RNInitializationControls() if initialization is None else initialization
    proposal = RNCollectiveProposalControls() if proposal is None else proposal
    proposal.validate()
    grid = make_grid(controls, case)
    config = pure_config_with_density_edges_if_needed(pure_config, grid)
    worker_count = resolve_parallel_workers(len(seeds), parallel_workers)
    seed_payloads, actual_workers = run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            _run_pure_walking_seed_worker,
            case,
            controls,
            seed,
            config,
            grid,
            progress_queue,
            initialization,
            proposal,
            proposal_family,
            guide_family,
        ),
        run_serial_seed=lambda seed: run_pure_walking_seed(
            case,
            controls,
            seed,
            pure_config=config,
            density_grid=grid,
            progress=progress,
            initialization=initialization,
            proposal=proposal,
            proposal_family=proposal_family,
            guide_family=guide_family,
        ),
    )
    return {
        "schema_version": "transported_pure_walking_case_v1",
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        "omega": case.omega,
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "parallel_workers_requested": worker_count,
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        **proposal.to_metadata(),
        "proposal_family": proposal_family,
        "guide_family": guide_family,
        "pure_config": pure_config_metadata(config),
        "status": case_status(seed_payloads),
        "seed_results": seed_payloads,
    }


def _run_pure_walking_seed_worker(
    case: RNCase,
    controls: RNRunControls,
    seed: int,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray,
    progress_queue: Any | None,
    initialization: RNInitializationControls,
    proposal: RNCollectiveProposalControls,
    proposal_family: str,
    guide_family: str,
) -> tuple[int, dict[str, Any]]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, run_pure_walking_seed(
            case,
            controls,
            seed,
            pure_config=pure_config,
            density_grid=density_grid,
            progress=worker_progress,
            initialization=initialization,
            proposal=proposal,
            proposal_family=proposal_family,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def case_status(seed_payloads: list[dict[str, Any]]) -> str:
    statuses = {str(payload["status"]) for payload in seed_payloads}
    for status in (
        "PURE_WALKING_SCHEMA_NO_GO",
        "PURE_WALKING_NO_BLOCKS_NO_GO",
        "PURE_WALKING_INSUFFICIENT_SAMPLES_NO_GO",
        "PURE_WALKING_PLATEAU_NO_GO",
    ):
        if status in statuses:
            return status
    return "PURE_WALKING_GO" if statuses == {"PURE_WALKING_GO"} else "PURE_WALKING_NO_GO"


def pure_config_metadata(config: PureWalkingConfig) -> dict[str, Any]:
    return {
        "lag_steps": list(config.lag_steps),
        "lag_unit": config.lag_unit,
        "observables": list(config.observables),
        "observable_source": config.observable_source,
        "min_block_count": config.min_block_count,
        "min_walker_weight_ess": config.min_walker_weight_ess,
        "block_size_steps": config.block_size_steps,
        "transport_mode": config.transport_mode,
        "collection_mode": config.collection_mode,
        "center": config.center,
        "plateau_sigma_threshold": config.plateau_sigma_threshold,
        "plateau_abs_tolerance": config.plateau_abs_tolerance,
        "density_plateau_relative_l2_tolerance": (
            config.density_plateau_relative_l2_tolerance
        ),
        "transport_invariant_tests_passed": list(config.transport_invariant_tests_passed),
    }


def pure_config_with_density_edges_if_needed(
    config: PureWalkingConfig,
    grid: np.ndarray,
) -> PureWalkingConfig:
    if "density" not in config.observables or config.density_bin_edges is not None:
        return config
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("density grid must contain at least two centers")
    dx = float(grid[1] - grid[0])
    edges = np.concatenate(
        ([grid[0] - 0.5 * dx], 0.5 * (grid[:-1] + grid[1:]), [grid[-1] + 0.5 * dx])
    )
    return PureWalkingConfig(
        lag_steps=config.lag_steps,
        lag_unit=config.lag_unit,
        observables=config.observables,
        observable_source=config.observable_source,
        density_bin_edges=edges,
        pair_bin_edges=config.pair_bin_edges,
        structure_k_values=config.structure_k_values,
        min_block_count=config.min_block_count,
        min_walker_weight_ess=config.min_walker_weight_ess,
        block_size_steps=config.block_size_steps,
        transport_mode=config.transport_mode,
        collection_mode=config.collection_mode,
        center=config.center,
        plateau_sigma_threshold=config.plateau_sigma_threshold,
        plateau_abs_tolerance=config.plateau_abs_tolerance,
        density_plateau_relative_l2_tolerance=config.density_plateau_relative_l2_tolerance,
        schema_atol=config.schema_atol,
        schema_rtol=config.schema_rtol,
        transport_invariant_tests_passed=config.transport_invariant_tests_passed,
    )
