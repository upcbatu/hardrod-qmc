from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import (
    PureWalkingConfig,
    TransportedAuxiliaryForwardWalking,
)
from hrdmc.io.progress import QueuedProgress
from hrdmc.monte_carlo.dmc.collective_rn import (
    CollectiveRNConfig,
    CollectiveRNMove,
)
from hrdmc.monte_carlo.dmc.local import (
    DMCConfig,
    run_dmc_streaming,
)
from hrdmc.runners import run_seed_batch
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
    OpenHardRodTrapGapHTransformProposalKernel,
    OpenLineHardRodSystem,
    OrderedHarmonicOscillatorHeatKernel,
)
from hrdmc.theory import (
    trapped_tg_energy_total,
    trapped_tg_r2_radius,
    trapped_tg_rms_radius,
)
from hrdmc.wavefunctions.guides import ReducedTGHardRodGuide
from hrdmc.workflows.anchors.exact_validation.models import (
    TrappedTGAnchor,
    TrappedTGSeedRun,
)
from hrdmc.workflows.anchors.exact_validation.tg_comparison import (
    density_profile_payload,
    trapped_tg_exact_comparison,
)
from hrdmc.workflows.anchors.exact_validation.tg_pure import (
    pure_config_payload,
    trapped_tg_pure_config,
    trapped_tg_seed_payload,
)
from hrdmc.workflows.dmc.benchmark_packet.case import summarize_pure_seed_payloads
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.initial_conditions import initial_walkers
from hrdmc.workflows.dmc.trapped import DMCRunControls


def run_trapped_tg_anchor(
    anchor: TrappedTGAnchor,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    worker_count: int,
    energy_tolerance: float,
    pure_lag_steps: tuple[int, ...],
    pure_density_lag_steps: tuple[int, ...] | None,
    pure_observables: tuple[str, ...],
    pure_min_block_count: int,
    pure_min_walker_weight_ess: float,
    pure_min_source_ancestor_ess: float,
    pure_max_source_family_fraction: float,
    pure_plateau_window_lag_count: int,
    pure_collection_stride_steps: int,
    pure_density_collection_stride_steps: int | None,
    density_plateau_relative_l2_tolerance: float,
    pure_rms_plateau_relative_tolerance: float,
    pure_plateau_equivalence_confidence_level: float,
    pure_r2_relative_tolerance: float,
    pure_rms_relative_tolerance: float,
    pure_density_l2_tolerance: float,
    density_accounting_tolerance: float,
    density_shape_min_bins: int,
    progress: Any,
    collective_rn: CollectiveRNControls | None = None,
) -> dict[str, Any]:
    if anchor.n_particles < 2:
        raise ValueError("trapped TG DMC anchors require n_particles >= 2")
    density_grid = np.linspace(-controls.grid_extent, controls.grid_extent, controls.n_bins)
    pure_config = trapped_tg_pure_config(
        density_grid=density_grid,
        lag_steps=pure_lag_steps,
        density_lag_steps=pure_density_lag_steps,
        observables=pure_observables,
        min_block_count=pure_min_block_count,
        min_walker_weight_ess=pure_min_walker_weight_ess,
        min_source_ancestor_ess=pure_min_source_ancestor_ess,
        max_source_family_fraction=pure_max_source_family_fraction,
        plateau_window_lag_count=pure_plateau_window_lag_count,
        collection_stride_steps=pure_collection_stride_steps,
        density_collection_stride_steps=pure_density_collection_stride_steps,
        density_plateau_relative_l2_tolerance=density_plateau_relative_l2_tolerance,
        rms_plateau_relative_tolerance=pure_rms_plateau_relative_tolerance,
        plateau_equivalence_confidence_level=(pure_plateau_equivalence_confidence_level),
    )
    seed_runs, actual_workers = _run_trapped_tg_seed_runs(
        anchor,
        controls,
        seeds,
        density_grid,
        pure_config,
        worker_count=worker_count,
        progress=progress,
        collective_rn=collective_rn,
    )
    seed_summaries = [run.dmc_summary for run in seed_runs]
    seed_payloads = [trapped_tg_seed_payload(run) for run in seed_runs]
    pure_summary = summarize_pure_seed_payloads(seed_payloads, config=pure_config)
    energy_values = np.asarray([summary.mixed_energy for summary in seed_summaries])
    exact_energy = trapped_tg_energy_total(anchor.n_particles, anchor.omega)
    abs_error = abs(float(np.mean(energy_values)) - exact_energy)
    density_profile = density_profile_payload(anchor, seed_summaries, pure_summary)
    exact_comparison = trapped_tg_exact_comparison(
        anchor,
        seed_summaries=seed_summaries,
        pure_summary=pure_summary,
        density_profile=density_profile,
        energy_abs_error=abs_error,
        energy_tolerance=energy_tolerance,
        pure_r2_relative_tolerance=pure_r2_relative_tolerance,
        pure_rms_relative_tolerance=pure_rms_relative_tolerance,
        pure_density_l2_tolerance=pure_density_l2_tolerance,
        density_accounting_tolerance=density_accounting_tolerance,
        density_shape_min_bins=density_shape_min_bins,
    )
    return {
        "anchor_id": anchor.anchor_id,
        "status": exact_comparison["status"],
        "anchor_type": (
            "trapped_tg_dmc_with_collective_rn_plus_transported_fw"
            if collective_rn is not None
            else "trapped_tg_local_dmc_plus_transported_fw"
        ),
        "exact_solution": {
            "model": "zero-length hard rods in a harmonic trap",
            "units": "harmonic oscillator units",
            "formula": "TG harmonic mapping: E0 = N^2 / 2",
            "n_particles": anchor.n_particles,
            "energy_total": exact_energy,
            "r2_radius": trapped_tg_r2_radius(anchor.n_particles, anchor.omega),
            "rms_radius": trapped_tg_rms_radius(anchor.n_particles, anchor.omega),
        },
        "mixed_energy": float(np.mean(energy_values)),
        "mixed_energy_seed_stderr": _stderr(energy_values),
        "absolute_energy_error": abs_error,
        "relative_energy_error": abs_error / exact_energy,
        "energy_tolerance": energy_tolerance,
        "pure_config": pure_config_payload(pure_config),
        "pure_walking": pure_summary,
        "exact_comparison": exact_comparison,
        "density_profile": density_profile,
        "seed_count": len(seeds),
        "parallel_workers": actual_workers,
        "seed_summaries": seed_payloads,
    }


def _run_trapped_tg_seed_runs(
    anchor: TrappedTGAnchor,
    controls: DMCRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    *,
    worker_count: int,
    progress: Any,
    collective_rn: CollectiveRNControls | None,
) -> tuple[list[TrappedTGSeedRun], int]:
    return run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            _trapped_tg_seed_worker,
            anchor,
            controls,
            seed,
            density_grid,
            pure_config,
            progress_queue,
            collective_rn,
        ),
        run_serial_seed=lambda seed: _run_trapped_tg_seed(
            anchor,
            controls,
            seed,
            density_grid,
            pure_config,
            progress=progress,
            collective_rn=collective_rn,
        ),
    )


def _trapped_tg_seed_worker(
    anchor: TrappedTGAnchor,
    controls: DMCRunControls,
    seed: int,
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    progress_queue: Any | None = None,
    collective_rn: CollectiveRNControls | None = None,
) -> tuple[int, TrappedTGSeedRun]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, _run_trapped_tg_seed(
            anchor,
            controls,
            seed,
            density_grid,
            pure_config,
            progress=worker_progress,
            collective_rn=collective_rn,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def _run_trapped_tg_seed(
    anchor: TrappedTGAnchor,
    controls: DMCRunControls,
    seed: int,
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    *,
    progress: Any | None = None,
    collective_rn: CollectiveRNControls | None = None,
) -> TrappedTGSeedRun:
    system = OpenLineHardRodSystem(n_particles=anchor.n_particles, rod_length=0.0)
    trap = HarmonicTrap(omega=anchor.omega)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=anchor.omega,
        relative_alpha=controls.relative_alpha,
    )
    rng = np.random.default_rng(seed)
    observer = TransportedAuxiliaryForwardWalking(pure_config)
    scheduled_move = _exact_tg_collective_move(
        system,
        trap,
        collective_rn,
        dt=controls.dt,
    )
    summary = run_dmc_streaming(
        initial_walkers=initial_walkers(system, controls.walkers, rng),
        guide=guide,
        system=system,
        density_grid=density_grid,
        config=DMCConfig(
            ess_resample_fraction=controls.ess_resample_fraction,
            local_step_method=controls.local_step_method,
        ),
        scheduled_move=scheduled_move,
        rng=rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
        transport_observer=observer,
    )
    if collective_rn is not None:
        summary.metadata["collective_rn"] = {
            **collective_rn.to_metadata(),
            "target_family": "ordered-harmonic-exact",
        }
    pure_result = observer.result()
    return TrappedTGSeedRun(seed=seed, dmc_summary=summary, pure_result=pure_result)


def _exact_tg_collective_move(
    system: OpenLineHardRodSystem,
    trap: HarmonicTrap,
    controls: CollectiveRNControls | None,
    *,
    dt: float,
) -> CollectiveRNMove | None:
    if controls is None:
        return None
    controls.validate()
    proposal_kernel = (
        HarmonicMehlerKernel(trap=trap)
        if controls.proposal_family == "harmonic-mehler"
        else OpenHardRodTrapGapHTransformProposalKernel(system=system, trap=trap)
    )
    return CollectiveRNMove(
        config=CollectiveRNConfig(
            step_tau=dt,
            cadence_tau=controls.cadence_tau,
            component_log_scales=controls.component_log_scales,
            component_probabilities=controls.component_probabilities,
            include_guide_ratio=controls.include_guide_ratio,
        ),
        system=system,
        target_kernel=OrderedHarmonicOscillatorHeatKernel(system=system, trap=trap),
        proposal_kernel=proposal_kernel,
    )


def _stderr(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))
