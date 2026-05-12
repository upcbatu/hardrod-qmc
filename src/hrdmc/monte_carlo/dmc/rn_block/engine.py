from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hrdmc.io.progress import ProgressBar
from hrdmc.monte_carlo.dmc.common.guide_api import (
    evaluate_guide,
    guide_batch_backend,
)
from hrdmc.monte_carlo.dmc.common.population import (
    effective_sample_size,
    maybe_resample_population,
    maybe_resample_population_with_indices,
    normalize_log_weights,
    recenter_log_weights,
)
from hrdmc.monte_carlo.dmc.rn_block.config import RNBlockDMCConfig
from hrdmc.monte_carlo.dmc.rn_block.results import RNBlockDMCResult, RNBlockStreamingSummary
from hrdmc.monte_carlo.dmc.rn_block.streaming_state import RNBlockStreamingState
from hrdmc.monte_carlo.dmc.rn_block.transitions import (
    RNBlockLocalStep,
    advance_local_step,
    advance_rn_block,
    euler_drift_diffusion_step,
)
from hrdmc.monte_carlo.dmc.rn_block.transport import (
    RNTransportEvent,
    RNTransportObserver,
    com_rao_blackwell_r2_per_walker,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.systems.propagators import ProposalTransitionKernel, TargetTransitionKernel
from hrdmc.wavefunctions import DMCGuide

FloatArray = NDArray[np.float64]


def run_rn_block_dmc(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    target_kernel: TargetTransitionKernel,
    proposal_kernel: ProposalTransitionKernel,
    config: RNBlockDMCConfig | None = None,
    rng: np.random.Generator | None = None,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int = 1,
    local_step: RNBlockLocalStep | None = None,
    include_guide_ratio: bool = True,
    progress: ProgressBar | None = None,
) -> RNBlockDMCResult:
    """Run an RN-corrected DMC loop with injected physics owners.

    The RN block owns only the collective proposal and RN log-weight correction.
    The system, target kernel, proposal kernel, and guide are supplied by callers.
    """

    cfg = config or RNBlockDMCConfig()
    cfg.validate()
    _validate_run_inputs(dt, burn_in_steps, production_steps, store_every)
    rng = np.random.default_rng() if rng is None else rng
    stepper = euler_drift_diffusion_step if local_step is None else local_step
    positions = np.asarray(initial_walkers, dtype=float).copy()
    if positions.ndim != 2:
        raise ValueError("initial_walkers must have shape (n_walkers, n_particles)")
    if positions.shape[1] != system.n_particles:
        raise ValueError("initial walker particle count must match system")
    local_energies, valid = evaluate_guide(guide, positions)
    if not np.all(valid):
        raise ValueError("initial_walkers must all be valid finite guide configurations")

    log_weights = np.zeros(positions.shape[0], dtype=float)
    rn_interval_steps = max(1, int(round(cfg.rn_cadence_tau / dt)))
    snapshots: list[FloatArray] = []
    stored_energies: list[FloatArray] = []
    stored_weights: list[FloatArray] = []
    rn_event_count = 0
    local_step_count = 0
    killed_count = 0
    resample_count = 0
    ess_values: list[float] = []
    total_steps = burn_in_steps + production_steps

    for step_index in range(1, total_steps + 1):
        if step_index % rn_interval_steps == 0:
            advance = advance_rn_block(
                cfg,
                system,
                guide,
                target_kernel,
                proposal_kernel,
                rng,
                positions,
                local_energies,
                log_weights,
                include_guide_ratio=include_guide_ratio,
            )
            rn_event_count += 1
        else:
            advance = advance_local_step(
                stepper,
                guide,
                rng,
                positions,
                local_energies,
                log_weights,
                dt,
            )
            local_step_count += 1
        positions = advance.positions
        local_energies = advance.local_energies
        log_weights = advance.log_weights
        killed = advance.killed
        killed_count += int(np.count_nonzero(killed))
        log_weights = recenter_log_weights(log_weights)
        ess = effective_sample_size(log_weights)
        ess_values.append(ess)
        positions, local_energies, log_weights, resampled = maybe_resample_population(
            positions,
            local_energies,
            log_weights,
            rng,
            threshold_fraction=cfg.ess_resample_fraction,
        )
        if resampled:
            resample_count += 1
        if progress is not None:
            progress.update(1)

        if step_index > burn_in_steps:
            production_index = step_index - burn_in_steps
            if production_index % store_every == 0 or production_index == production_steps:
                snapshots.append(positions.copy())
                stored_energies.append(local_energies.copy())
                stored_weights.append(normalize_log_weights(log_weights))

    return RNBlockDMCResult(
        snapshots=np.vstack(snapshots),
        local_energies=np.concatenate(stored_energies),
        weights=np.concatenate(stored_weights),
        metadata={
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "rn_interval_steps": rn_interval_steps,
            "stored_batch_count": len(snapshots),
            "rn_event_count": rn_event_count,
            "local_step_count": local_step_count,
            "killed_count": killed_count,
            "resample_count": resample_count,
            "ess_min": float(np.min(ess_values)) if ess_values else float("nan"),
            "ess_mean": float(np.mean(ess_values)) if ess_values else float("nan"),
            "ess_resample_fraction": cfg.ess_resample_fraction,
            "include_guide_ratio": include_guide_ratio,
            "guide_batch_backend": guide_batch_backend(guide),
        },
    )


def run_rn_block_dmc_streaming(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    target_kernel: TargetTransitionKernel,
    proposal_kernel: ProposalTransitionKernel,
    density_grid: FloatArray,
    config: RNBlockDMCConfig | None = None,
    rng: np.random.Generator | None = None,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int = 1,
    local_step: RNBlockLocalStep | None = None,
    include_guide_ratio: bool = True,
    progress: ProgressBar | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume: bool = False,
    transport_observer: RNTransportObserver | None = None,
    transport_com_variance: float | None = None,
) -> RNBlockStreamingSummary:
    """Run RN-block DMC and accumulate compact observables during production."""

    cfg = config or RNBlockDMCConfig()
    cfg.validate()
    _validate_run_inputs(dt, burn_in_steps, production_steps, store_every)
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    rng = np.random.default_rng() if rng is None else rng
    stepper = euler_drift_diffusion_step if local_step is None else local_step
    grid = np.asarray(density_grid, dtype=float)
    rn_interval_steps = max(1, int(round(cfg.rn_cadence_tau / dt)))
    total_steps = burn_in_steps + production_steps
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None

    if resume and checkpoint_file is not None and checkpoint_file.exists():
        state = RNBlockStreamingState.from_checkpoint(
            checkpoint_file,
            rng=rng,
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            rn_interval_steps=rn_interval_steps,
            system=system,
            density_grid=grid,
        )
    else:
        state = RNBlockStreamingState.from_initial(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            density_grid=grid,
        )

    for step_index in range(state.step_start, total_steps + 1):
        if step_index % rn_interval_steps == 0:
            advance = advance_rn_block(
                cfg,
                system,
                guide,
                target_kernel,
                proposal_kernel,
                rng,
                state.positions,
                state.local_energies,
                state.log_weights,
                include_guide_ratio=include_guide_ratio,
            )
            state.rn_event_count += 1
        else:
            advance = advance_local_step(
                stepper,
                guide,
                rng,
                state.positions,
                state.local_energies,
                state.log_weights,
                dt,
            )
            state.local_step_count += 1
        state.positions = advance.positions
        state.local_energies = advance.local_energies
        finite_log_weights = advance.log_weights[np.isfinite(advance.log_weights)]
        weight_gauge_shift = (
            float(np.max(finite_log_weights)) if finite_log_weights.size else 0.0
        )
        state.log_weights = recenter_log_weights(advance.log_weights)
        ess = effective_sample_size(state.log_weights)
        state.record_step(killed=advance.killed, ess=ess, telemetry=advance.telemetry)
        log_weights_pre_resample = state.log_weights
        (
            state.positions,
            state.local_energies,
            state.log_weights,
            resampled,
            parent_indices,
        ) = maybe_resample_population_with_indices(
            state.positions,
            state.local_energies,
            state.log_weights,
            rng,
            threshold_fraction=cfg.ess_resample_fraction,
        )
        state.record_resample(resampled)
        if transport_observer is not None:
            production_step_id = (
                step_index - burn_in_steps if step_index > burn_in_steps else None
            )
            r2_rb = (
                None
                if transport_com_variance is None
                else com_rao_blackwell_r2_per_walker(
                    state.positions,
                    center=system.center,
                    com_variance=transport_com_variance,
                )
            )
            transport_observer.record_transport_event(
                RNTransportEvent(
                    step_id=step_index,
                    production_step_id=production_step_id,
                    block_id=state.rn_event_count,
                    positions=state.positions.copy(),
                    local_energy_per_walker=state.local_energies.copy(),
                    r2_rb_per_walker=None if r2_rb is None else r2_rb.copy(),
                    log_weights_pre_resample=log_weights_pre_resample.copy(),
                    log_weights_post_resample=state.log_weights.copy(),
                    parent_indices=parent_indices.copy(),
                    resampled=resampled,
                    weight_gauge_shift=weight_gauge_shift,
                )
            )
        if progress is not None:
            progress.update(1)

        state.record_production_if_due(
            step_index=step_index,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            dt=dt,
            system=system,
            guide=guide,
        )
        if checkpoint_file is not None and checkpoint_every_steps is not None and (
            step_index % checkpoint_every_steps == 0 or step_index == total_steps
        ):
            state.save_checkpoint(
                checkpoint_file,
                step_index=step_index,
                rng=rng,
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                rn_interval_steps=rn_interval_steps,
                system=system,
            )

    return state.to_summary(
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        rn_interval_steps=rn_interval_steps,
        ess_resample_fraction=cfg.ess_resample_fraction,
        include_guide_ratio=include_guide_ratio,
        guide=guide,
    )


def _validate_run_inputs(
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
) -> None:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be non-negative")
    if production_steps <= 0:
        raise ValueError("production_steps must be positive")
    if store_every <= 0:
        raise ValueError("store_every must be positive")
