from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.workflows.anchors.exact_validation.models import TrappedTGSeedRun


def trapped_tg_pure_config(
    *,
    density_grid: np.ndarray,
    lag_steps: tuple[int, ...],
    observables: tuple[str, ...],
    min_block_count: int,
    min_walker_weight_ess: float,
    density_plateau_relative_l2_tolerance: float,
) -> PureWalkingConfig:
    clean_lags = tuple(sorted(set(int(lag) for lag in lag_steps)))
    clean_observables = tuple(dict.fromkeys(observables))
    if "r2" not in clean_observables:
        raise ValueError("exact validation packet requires transported FW r2")
    density_edges = (
        _edges_from_centers(density_grid)
        if "density" in clean_observables
        else None
    )
    return PureWalkingConfig(
        lag_steps=clean_lags,
        observables=clean_observables,
        density_bin_edges=density_edges,
        min_block_count=min_block_count,
        min_walker_weight_ess=min_walker_weight_ess,
        density_plateau_relative_l2_tolerance=density_plateau_relative_l2_tolerance,
        block_size_steps=1,
        transport_invariant_tests_passed=("lag0_identity",),
    )


def pure_config_payload(config: PureWalkingConfig) -> dict[str, Any]:
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
        "plateau_sigma_threshold": config.plateau_sigma_threshold,
        "plateau_abs_tolerance": config.plateau_abs_tolerance,
        "density_plateau_relative_l2_tolerance": (
            config.density_plateau_relative_l2_tolerance
        ),
        "schema_atol": config.schema_atol,
        "schema_rtol": config.schema_rtol,
        "transport_invariant_tests_passed": list(config.transport_invariant_tests_passed),
    }


def trapped_tg_seed_payload(run: TrappedTGSeedRun) -> dict[str, Any]:
    summary = run.rn_summary
    return {
        "seed": run.seed,
        "status": run.pure_result.status,
        "rn_summary": {
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
                "ess_min": summary.metadata.get("ess_min"),
                "ess_mean": summary.metadata.get("ess_mean"),
                "ess_fraction_min": summary.metadata.get("ess_fraction_min"),
                "log_weight_span_max": summary.metadata.get("log_weight_span_max"),
            },
        },
        "pure_walking": run.pure_result.to_summary_dict(),
    }


def _edges_from_centers(grid: np.ndarray) -> np.ndarray:
    centers = np.asarray(grid, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("density grid must contain at least two centers")
    dx = float(centers[1] - centers[0])
    return np.concatenate(
        (
            [centers[0] - 0.5 * dx],
            0.5 * (centers[:-1] + centers[1:]),
            [centers[-1] + 0.5 * dx],
        )
    )
