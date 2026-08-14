from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from hrdmc.artifacts.progress import ProgressBar, progress_bar
from hrdmc.sampling.dmc.engine import run_dmc_streaming
from hrdmc.sampling.dmc.results import DMCStreamingSummary
from hrdmc.sampling.dmc.telemetry import DMCTransportObserver
from hrdmc.sampling.dmc.transitions import DMCConfig
from hrdmc.sampling.initial_conditions import InitializationControls, prepare_initial_walkers
from hrdmc.system.settings import (
    DMCRunControls,
    TrappedCase,
    build_case_geometry,
    lda_target_rms,
    make_grid,
)
from hrdmc.trial.guide import DEFAULT_GUIDE_FAMILY, ReducedTGHardRodGuide, build_guide

MAX_AUTO_WORKERS = 6

def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds

def default_parallel_workers(seed_count: int, *, max_workers: int = MAX_AUTO_WORKERS) -> int:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    capped = min(seed_count, max_workers)
    if seed_count <= max_workers:
        return seed_count
    for preferred in (6, 5):
        if preferred <= capped and seed_count % preferred == 0:
            return preferred
    return capped

def resolve_parallel_workers(
    seed_count: int,
    requested_workers: int | None,
    *,
    max_workers: int = MAX_AUTO_WORKERS,
) -> int:
    if requested_workers is None or requested_workers == 0:
        return default_parallel_workers(seed_count, max_workers=max_workers)
    if requested_workers < 0:
        raise ValueError("parallel_workers must be non-negative")
    return min(seed_count, requested_workers)

def run_streaming_seed(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume: bool = False,
    initialization: InitializationControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    transport_observer: DMCTransportObserver | None = None,
) -> DMCStreamingSummary:
    rng = np.random.default_rng(seed)
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
    )
    grid = make_grid(controls, case) if density_grid is None else density_grid
    initialization = InitializationControls() if initialization is None else initialization
    target_initial_rms = (
        lda_target_rms(case, controls, grid)
        if initialization.mode in {"lda-rms-lattice", "lda-rms-logspread"}
        else None
    )
    initial = prepare_initial_walkers(
        system,
        guide,
        controls.walkers,
        rng,
        controls=initialization,
        target_initial_rms=target_initial_rms,
    )
    checkpoint_path = (
        checkpoint_dir / f"{case.case_id}_seed{seed}.npz" if checkpoint_dir is not None else None
    )
    summary = run_dmc_streaming(
        initial_walkers=initial.positions,
        guide=guide,
        system=system,
        density_grid=grid,
        config=DMCConfig(
            ess_resample_fraction=controls.ess_resample_fraction,
            drift_limiter=controls.drift_limiter,
        ),
        rng=rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
        checkpoint_path=checkpoint_path,
        checkpoint_every_steps=checkpoint_every_steps,
        resume=resume,
        checkpoint_identity={
            "case_id": case.case_id,
            "seed": int(seed),
            "guide_family": guide_family,
            "guide_parameters": {
                "relative_alpha": controls.relative_alpha,
            },
            "initialization": asdict(initialization),
        },
        transport_observer=transport_observer,
    )
    summary.metadata.update(initial.metadata)
    summary.metadata.update(case.unit_metadata())
    summary.metadata["guide_family"] = guide_family
    summary.metadata["resolved_guide_family"] = _guide_family_name(guide)
    summary.metadata["drift_limiter"] = controls.drift_limiter
    if controls.relative_alpha is not None:
        summary.metadata["relative_alpha"] = controls.relative_alpha
    initial_rms_value = initial.metadata["initial_rms_mean"]
    if not isinstance(initial_rms_value, int | float):
        raise RuntimeError("initial_rms_mean metadata must be numeric")
    initial_rms = float(initial_rms_value)
    summary.metadata["initial_to_production_rms_ratio"] = (
        float(summary.rms_radius / initial_rms) if initial_rms > 0.0 else float("nan")
    )
    return summary

def _guide_family_name(guide: object) -> str:
    if isinstance(guide, ReducedTGHardRodGuide):
        return "reduced-tg"
    return type(guide).__name__

def _dmc_total_steps(
    controls: DMCRunControls,
    *,
    seed_count: int,
    raw_validation: bool = False,
) -> int:
    multiplier = 2 if raw_validation else 1
    return multiplier * seed_count * (controls.burn_in_steps + controls.production_steps)

def dmc_progress_bar(
    *,
    controls: DMCRunControls,
    seed_count: int,
    label: str,
    enabled: bool,
    raw_validation: bool = False,
):
    return progress_bar(
        total=_dmc_total_steps(controls, seed_count=seed_count, raw_validation=raw_validation),
        label=label,
        enabled=enabled,
    )
