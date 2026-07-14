from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.estimators.pure.forward_walking.results import (
    PURE_STATUS_ACCEPTED,
    PURE_STATUS_INSUFFICIENT_SAMPLES,
    PURE_STATUS_NO_BLOCKS,
    PURE_STATUS_PLATEAU_UNRESOLVED,
    PURE_STATUS_SCHEMA_INVALID,
)
from hrdmc.io.progress import ProgressBar, QueuedProgress
from hrdmc.runners import run_seed_batch
from hrdmc.systems.harmonic_com_transition import harmonic_com_ground_variance
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.pure_walking.seed import run_pure_walking_seed
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    controls_to_dict,
    make_grid,
    resolve_parallel_workers,
)


def summarize_pure_walking_case(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    """Run a reproducible transported-FW case packet."""

    initialization = InitializationControls() if initialization is None else initialization
    grid = make_grid(controls, case)
    config = pure_config_for_case(pure_config, grid=grid, case=case)
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
            collective_rn,
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
            collective_rn=collective_rn,
            guide_family=guide_family,
        ),
    )
    return {
        "schema_version": "transported_pure_walking_case_v3",
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        **case.unit_metadata(),
        "controls": controls_to_dict(controls),
        "seeds": seeds,
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "parallel_workers_requested": worker_count,
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "collective_rn_controls": (None if collective_rn is None else collective_rn.to_metadata()),
        "guide_family": guide_family,
        "pure_config": pure_config_metadata(config),
        "status": case_status(seed_payloads),
        "seed_results": seed_payloads,
    }


def _run_pure_walking_seed_worker(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    pure_config: PureWalkingConfig,
    density_grid: np.ndarray,
    progress_queue: Any | None,
    initialization: InitializationControls,
    collective_rn: CollectiveRNControls | None,
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
            collective_rn=collective_rn,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def case_status(seed_payloads: list[dict[str, Any]]) -> str:
    statuses = {str(payload["status"]) for payload in seed_payloads}
    for status in (
        PURE_STATUS_SCHEMA_INVALID,
        PURE_STATUS_NO_BLOCKS,
        PURE_STATUS_INSUFFICIENT_SAMPLES,
        PURE_STATUS_PLATEAU_UNRESOLVED,
    ):
        if status in statuses:
            return status
    return PURE_STATUS_ACCEPTED if statuses == {PURE_STATUS_ACCEPTED} else "seed_disagreement"


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
        "plateau_abs_tolerance": config.plateau_abs_tolerance,
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
