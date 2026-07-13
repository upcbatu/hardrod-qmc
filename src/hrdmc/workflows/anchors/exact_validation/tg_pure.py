from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.workflows.anchors.exact_validation.models import TrappedTGSeedRun
from hrdmc.workflows.dmc.pure_walking.seed import compact_dmc_seed_summary


def trapped_tg_pure_config(
    *,
    density_grid: np.ndarray,
    lag_steps: tuple[int, ...],
    density_lag_steps: tuple[int, ...] | None,
    observables: tuple[str, ...],
    min_block_count: int,
    min_walker_weight_ess: float,
    min_source_ancestor_ess: float,
    max_source_family_fraction: float,
    plateau_window_lag_count: int,
    collection_stride_steps: int,
    density_collection_stride_steps: int | None,
    density_plateau_relative_l2_tolerance: float,
) -> PureWalkingConfig:
    clean_lags = tuple(sorted(set(int(lag) for lag in lag_steps)))
    clean_observables = tuple(dict.fromkeys(observables))
    if "r2" not in clean_observables:
        raise ValueError("exact validation packet requires transported FW r2")
    density_edges = _edges_from_centers(density_grid) if "density" in clean_observables else None
    return PureWalkingConfig(
        lag_steps=clean_lags,
        observables=clean_observables,
        density_bin_edges=density_edges,
        min_block_count=min_block_count,
        min_walker_weight_ess=min_walker_weight_ess,
        min_source_ancestor_ess=min_source_ancestor_ess,
        max_source_family_fraction=max_source_family_fraction,
        plateau_window_lag_count=plateau_window_lag_count,
        collection_stride_steps=collection_stride_steps,
        density_lag_steps=density_lag_steps,
        density_collection_stride_steps=density_collection_stride_steps,
        density_plateau_relative_l2_tolerance=density_plateau_relative_l2_tolerance,
        block_size_steps=1,
        transport_invariant_tests_passed=("lag0_identity",),
    )


def pure_config_payload(config: PureWalkingConfig) -> dict[str, Any]:
    return {
        "lag_steps": list(config.lag_steps),
        "density_lag_steps": (
            None if config.density_lag_steps is None else list(config.density_lag_steps)
        ),
        "lag_unit": config.lag_unit,
        "observables": list(config.observables),
        "observable_source": config.observable_source,
        "min_block_count": config.min_block_count,
        "min_walker_weight_ess": config.min_walker_weight_ess,
        "min_source_ancestor_ess": config.min_source_ancestor_ess,
        "max_source_family_fraction": config.max_source_family_fraction,
        "block_size_steps": config.block_size_steps,
        "collection_stride_steps": config.collection_stride_steps,
        "density_collection_stride_steps": config.density_collection_stride_steps,
        "transport_mode": config.transport_mode,
        "collection_mode": config.collection_mode,
        "plateau_sigma_threshold": config.plateau_sigma_threshold,
        "plateau_abs_tolerance": config.plateau_abs_tolerance,
        "plateau_window_lag_count": config.plateau_window_lag_count,
        "density_plateau_relative_l2_tolerance": (config.density_plateau_relative_l2_tolerance),
        "schema_atol": config.schema_atol,
        "schema_rtol": config.schema_rtol,
        "transport_invariant_tests_passed": list(config.transport_invariant_tests_passed),
    }


def trapped_tg_seed_payload(run: TrappedTGSeedRun) -> dict[str, Any]:
    summary = run.dmc_summary
    compact_summary = compact_dmc_seed_summary(summary)
    compact_summary["metadata"].update(
        {
            "ess_min": summary.metadata.get("ess_min"),
            "ess_mean": summary.metadata.get("ess_mean"),
        }
    )
    return {
        "seed": run.seed,
        "status": run.pure_result.status,
        "dmc_summary": compact_summary,
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
