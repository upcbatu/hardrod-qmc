from __future__ import annotations

from typing import Any

import numpy as np

from hrdmc.estimators.pure.forward_walking import (
    PureWalkingConfig,
    TransportedAuxiliaryForwardWalking,
)
from hrdmc.io.progress import QueuedProgress
from hrdmc.monte_carlo.dmc.rn_block import (
    RNBlockDMCConfig,
    run_rn_block_dmc_streaming,
)
from hrdmc.runners import run_seed_batch
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
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
from hrdmc.workflows.dmc.rn_block import RNRunControls, initial_walkers


def run_trapped_tg_anchor(
    anchor: TrappedTGAnchor,
    controls: RNRunControls,
    seeds: list[int],
    *,
    worker_count: int,
    energy_tolerance: float,
    pure_lag_steps: tuple[int, ...],
    pure_observables: tuple[str, ...],
    pure_min_block_count: int,
    pure_min_walker_weight_ess: float,
    density_plateau_relative_l2_tolerance: float,
    pure_r2_relative_tolerance: float,
    pure_rms_relative_tolerance: float,
    pure_density_l2_tolerance: float,
    density_accounting_tolerance: float,
    density_shape_min_bins: int,
    progress: Any,
) -> dict[str, Any]:
    if anchor.n_particles < 2:
        raise ValueError("trapped TG RN-DMC anchors require n_particles >= 2")
    density_grid = np.linspace(-controls.grid_extent, controls.grid_extent, controls.n_bins)
    pure_config = trapped_tg_pure_config(
        density_grid=density_grid,
        lag_steps=pure_lag_steps,
        observables=pure_observables,
        min_block_count=pure_min_block_count,
        min_walker_weight_ess=pure_min_walker_weight_ess,
        density_plateau_relative_l2_tolerance=density_plateau_relative_l2_tolerance,
    )
    seed_runs, actual_workers = _run_trapped_tg_seed_runs(
        anchor,
        controls,
        seeds,
        density_grid,
        pure_config,
        worker_count=worker_count,
        progress=progress,
    )
    seed_summaries = [run.rn_summary for run in seed_runs]
    seed_payloads = [trapped_tg_seed_payload(run) for run in seed_runs]
    pure_summary = summarize_pure_seed_payloads(seed_payloads)
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
        "status": exact_comparison["full_engine_gate"],
        "anchor_type": "trapped_tg_rn_dmc_plus_transported_fw",
        "exact_solution": {
            "model": "zero-length hard rods in a harmonic trap",
            "formula": "TG harmonic mapping: E0 = N^2 * omega / sqrt(2)",
            "n_particles": anchor.n_particles,
            "omega": anchor.omega,
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
    controls: RNRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    *,
    worker_count: int,
    progress: Any,
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
        ),
        run_serial_seed=lambda seed: _run_trapped_tg_seed(
            anchor,
            controls,
            seed,
            density_grid,
            pure_config,
            progress=progress,
        ),
    )


def _trapped_tg_seed_worker(
    anchor: TrappedTGAnchor,
    controls: RNRunControls,
    seed: int,
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    progress_queue: Any | None = None,
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
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def _run_trapped_tg_seed(
    anchor: TrappedTGAnchor,
    controls: RNRunControls,
    seed: int,
    density_grid: np.ndarray,
    pure_config: PureWalkingConfig,
    *,
    progress: Any | None = None,
) -> TrappedTGSeedRun:
    system = OpenLineHardRodSystem(n_particles=anchor.n_particles, rod_length=0.0)
    trap = HarmonicTrap(omega=anchor.omega)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=anchor.omega / np.sqrt(2.0),
    )
    rng = np.random.default_rng(seed)
    observer = TransportedAuxiliaryForwardWalking(pure_config)
    summary = run_rn_block_dmc_streaming(
        initial_walkers=initial_walkers(system, controls.walkers, rng),
        guide=guide,
        system=system,
        target_kernel=OrderedHarmonicOscillatorHeatKernel(system=system, trap=trap),
        proposal_kernel=HarmonicMehlerKernel(trap=trap),
        density_grid=density_grid,
        config=RNBlockDMCConfig(
            tau_block=controls.tau_block,
            rn_cadence_tau=controls.rn_cadence_tau,
            component_log_scales=(-0.02, 0.0, 0.02),
            component_probabilities=(0.25, 0.5, 0.25),
        ),
        rng=rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
        transport_observer=observer,
    )
    pure_result = observer.result()
    return TrappedTGSeedRun(seed=seed, rn_summary=summary, pure_result=pure_result)


def _stderr(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))
