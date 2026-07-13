from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.artifacts import implementation_identity
from hrdmc.artifacts.schema import to_jsonable
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
from hrdmc.monte_carlo.dmc.local.config import DMCConfig
from hrdmc.monte_carlo.dmc.local.results import DMCResult, DMCStreamingSummary
from hrdmc.monte_carlo.dmc.local.scheduled import ScheduledMove
from hrdmc.monte_carlo.dmc.local.streaming_state import DMCStreamingState
from hrdmc.monte_carlo.dmc.local.transitions import (
    DMCStep,
    advance_local_step,
    euler_drift_diffusion_step,
    metropolis_drift_diffusion_step,
)
from hrdmc.monte_carlo.dmc.local.transport import (
    DMCTransportEvent,
    DMCTransportObserver,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions.api import DMCGuide

FloatArray = NDArray[np.float64]


def run_dmc(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    config: DMCConfig | None = None,
    rng: np.random.Generator | None = None,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int = 1,
    local_step: DMCStep | None = None,
    scheduled_move: ScheduledMove | None = None,
    progress: ProgressBar | None = None,
) -> DMCResult:
    """Run local importance-sampled DMC with an optional scheduled move."""

    cfg = config or DMCConfig()
    cfg.validate()
    _validate_run_inputs(dt, burn_in_steps, production_steps, store_every)
    rng = np.random.default_rng() if rng is None else rng
    stepper = _resolve_local_step(cfg, local_step)
    positions = np.asarray(initial_walkers, dtype=float).copy()
    if positions.ndim != 2:
        raise ValueError("initial_walkers must have shape (n_walkers, n_particles)")
    if positions.shape[1] != system.n_particles:
        raise ValueError("initial walker particle count must match system")
    local_energies, valid = evaluate_guide(guide, positions)
    if not np.all(valid):
        raise ValueError("initial_walkers must all be valid finite guide configurations")

    log_weights = np.zeros(positions.shape[0], dtype=float)
    scheduled_move_interval_steps = _scheduled_interval_steps(scheduled_move, dt)
    snapshots: list[FloatArray] = []
    stored_energies: list[FloatArray] = []
    stored_weights: list[FloatArray] = []
    scheduled_move_count = 0
    local_step_count = 0
    killed_count = 0
    resample_count = 0
    ess_values: list[float] = []
    total_steps = burn_in_steps + production_steps

    for step_index in range(1, total_steps + 1):
        if scheduled_move is not None and step_index % scheduled_move_interval_steps == 0:
            advance = scheduled_move.advance(
                guide=guide,
                rng=rng,
                positions=positions,
                local_energies=local_energies,
                log_weights=log_weights,
            )
            scheduled_move_count += 1
        else:
            advance = advance_local_step(
                stepper,
                guide,
                rng,
                positions,
                local_energies,
                log_weights,
                dt,
                center=system.center,
                rod_length=system.rod_length,
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

    return DMCResult(
        snapshots=np.vstack(snapshots),
        local_energies=np.concatenate(stored_energies),
        weights=np.concatenate(stored_weights),
        metadata={
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "stored_batch_count": len(snapshots),
            "local_step_count": local_step_count,
            "killed_count": killed_count,
            "resample_count": resample_count,
            "ess_min": float(np.min(ess_values)) if ess_values else float("nan"),
            "ess_mean": float(np.mean(ess_values)) if ess_values else float("nan"),
            "ess_resample_fraction": cfg.ess_resample_fraction,
            "local_step_method": cfg.local_step_method,
            "drift_limiter": cfg.drift_limiter,
            "guide_batch_backend": guide_batch_backend(guide),
            **_scheduled_metadata(
                scheduled_move,
                event_count=scheduled_move_count,
                interval_steps=scheduled_move_interval_steps,
            ),
        },
    )


def run_dmc_streaming(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    density_grid: FloatArray,
    config: DMCConfig | None = None,
    rng: np.random.Generator | None = None,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int = 1,
    local_step: DMCStep | None = None,
    scheduled_move: ScheduledMove | None = None,
    progress: ProgressBar | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume: bool = False,
    checkpoint_identity: dict[str, Any] | None = None,
    transport_observer: DMCTransportObserver | None = None,
) -> DMCStreamingSummary:
    """Run DMC and accumulate compact observables during production."""

    cfg = config or DMCConfig()
    cfg.validate()
    _validate_run_inputs(dt, burn_in_steps, production_steps, store_every)
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    rng = np.random.default_rng() if rng is None else rng
    stepper = _resolve_local_step(cfg, local_step)
    grid = np.asarray(density_grid, dtype=float)
    scheduled_move_interval_steps = _scheduled_interval_steps(scheduled_move, dt)
    total_steps = burn_in_steps + production_steps
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpointing_active = resume or (
        checkpoint_file is not None and checkpoint_every_steps is not None
    )
    if checkpoint_every_steps is not None and checkpoint_file is None:
        raise ValueError("checkpoint_every_steps requires checkpoint_path")
    if resume and checkpoint_file is None:
        raise ValueError("resume requires checkpoint_path")
    if resume and checkpoint_file is not None and not checkpoint_file.exists():
        raise FileNotFoundError(f"DMC checkpoint does not exist: {checkpoint_file}")
    if transport_observer is not None and checkpointing_active:
        raise ValueError(
            "transport observers cannot be combined with checkpoint/resume until "
            "observer state is checkpointed"
        )
    if checkpointing_active and checkpoint_identity is None:
        raise ValueError(
            "checkpoint_identity is required to bind checkpoints to the guide configuration"
        )
    resume_identity = (
        None
        if not checkpointing_active
        else _build_resume_identity(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            config=cfg,
            local_step=local_step,
            scheduled_move=scheduled_move,
            scheduled_move_interval_steps=scheduled_move_interval_steps,
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            checkpoint_identity=checkpoint_identity,
        )
    )

    if resume and checkpoint_file is not None:
        if resume_identity is None:
            raise RuntimeError("resume identity was not constructed")
        state = DMCStreamingState.from_checkpoint(
            checkpoint_file,
            rng=rng,
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            scheduled_move_interval_steps=scheduled_move_interval_steps,
            scheduled_move_enabled=scheduled_move is not None,
            scheduled_move_name=None if scheduled_move is None else scheduled_move.name,
            system=system,
            density_grid=grid,
            resume_identity=resume_identity,
        )
    else:
        state = DMCStreamingState.from_initial(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            density_grid=grid,
        )

    for step_index in range(state.step_start, total_steps + 1):
        if scheduled_move is not None and step_index % scheduled_move_interval_steps == 0:
            advance = scheduled_move.advance(
                guide=guide,
                rng=rng,
                positions=state.positions,
                local_energies=state.local_energies,
                log_weights=state.log_weights,
            )
            state.scheduled_move_count += 1
        else:
            advance = advance_local_step(
                stepper,
                guide,
                rng,
                state.positions,
                state.local_energies,
                state.log_weights,
                dt,
                center=system.center,
                rod_length=system.rod_length,
            )
            state.local_step_count += 1
        state.positions = advance.positions
        state.local_energies = advance.local_energies
        finite_log_weights = advance.log_weights[np.isfinite(advance.log_weights)]
        weight_gauge_shift = float(np.max(finite_log_weights)) if finite_log_weights.size else 0.0
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
            production_step_id = step_index - burn_in_steps if step_index > burn_in_steps else None
            transport_observer.record_transport_event(
                DMCTransportEvent(
                    step_id=step_index,
                    production_step_id=production_step_id,
                    scheduled_move_count=state.scheduled_move_count,
                    positions=state.positions.copy(),
                    local_energy_per_walker=state.local_energies.copy(),
                    log_weights_pre_resample=log_weights_pre_resample.copy(),
                    log_weights_post_resample=state.log_weights.copy(),
                    parent_indices=parent_indices.copy(),
                    resampled=resampled,
                    weight_gauge_shift=weight_gauge_shift,
                )
            )
        if step_index == burn_in_steps:
            state.reset_interval_trace()
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
        if (
            checkpoint_file is not None
            and checkpoint_every_steps is not None
            and (step_index % checkpoint_every_steps == 0 or step_index == total_steps)
        ):
            if resume_identity is None:
                raise RuntimeError("checkpoint identity was not constructed")
            state.save_checkpoint(
                checkpoint_file,
                step_index=step_index,
                rng=rng,
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                scheduled_move_interval_steps=scheduled_move_interval_steps,
                scheduled_move_enabled=scheduled_move is not None,
                scheduled_move_name=None if scheduled_move is None else scheduled_move.name,
                system=system,
                resume_identity=resume_identity,
            )

    summary = state.to_summary(
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        scheduled_move_interval_steps=scheduled_move_interval_steps,
        scheduled_move_enabled=scheduled_move is not None,
        ess_resample_fraction=cfg.ess_resample_fraction,
        guide=guide,
        scheduled_move_metadata=_scheduled_metadata(
            scheduled_move,
            event_count=state.scheduled_move_count,
            interval_steps=scheduled_move_interval_steps,
        ),
    )
    summary.metadata["local_step_method"] = cfg.local_step_method
    summary.metadata["drift_limiter"] = cfg.drift_limiter
    return summary


def _resolve_local_step(
    config: DMCConfig,
    local_step: DMCStep | None,
) -> DMCStep:
    if local_step is not None:
        if config.drift_limiter != "none":
            raise ValueError("configured drift_limiter cannot be applied to a custom local_step")
        return local_step
    if config.local_step_method == "metropolis":
        return partial(
            metropolis_drift_diffusion_step,
            drift_limiter=config.drift_limiter,
        )
    return euler_drift_diffusion_step


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


def _scheduled_interval_steps(scheduled_move: ScheduledMove | None, dt: float) -> int:
    if scheduled_move is None:
        return 0
    scheduled_move.validate_timestep(dt)
    interval_steps = int(scheduled_move.interval_steps(dt))
    if interval_steps <= 0:
        raise ValueError("scheduled move interval must be positive")
    return interval_steps


def _scheduled_metadata(
    scheduled_move: ScheduledMove | None,
    *,
    event_count: int,
    interval_steps: int,
) -> dict:
    metadata = {
        "scheduled_move_enabled": scheduled_move is not None,
        "scheduled_move_name": None if scheduled_move is None else scheduled_move.name,
        "scheduled_move_interval_steps": interval_steps,
        "scheduled_move_count": event_count,
    }
    if scheduled_move is not None:
        metadata.update(scheduled_move.metadata())
    return metadata


def _build_resume_identity(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    config: DMCConfig,
    local_step: DMCStep | None,
    scheduled_move: ScheduledMove | None,
    scheduled_move_interval_steps: int,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
    checkpoint_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    if checkpoint_identity is None:
        raise ValueError("checkpoint_identity is required")
    walkers = np.asarray(initial_walkers, dtype=float)
    if walkers.ndim != 2 or walkers.shape[1] != system.n_particles:
        raise ValueError("initial_walkers must have shape (n_walkers, n_particles)")
    step_identity = (
        {
            "kind": "configured",
            "method": config.local_step_method,
            "drift_limiter": config.drift_limiter,
        }
        if local_step is None
        else {
            "kind": "callable",
            "module": getattr(local_step, "__module__", type(local_step).__module__),
            "qualname": getattr(local_step, "__qualname__", type(local_step).__qualname__),
        }
    )
    scheduled_identity = (
        None
        if scheduled_move is None
        else {
            "name": scheduled_move.name,
            "interval_steps": scheduled_move_interval_steps,
            "metadata": scheduled_move.metadata(),
        }
    )
    implementation = implementation_identity()
    if implementation.get("status") != "identified":
        raise RuntimeError("checkpointing requires an identifiable scientific source tree")
    identity = {
        "engine": "local_importance_sampled_dmc",
        "implementation": implementation,
        "run": {
            "dt": dt,
            "burn_in_steps": burn_in_steps,
            "production_steps": production_steps,
            "store_every": store_every,
            "walker_count": int(walkers.shape[0]),
        },
        "algorithm": {
            "ess_resample_fraction": config.ess_resample_fraction,
            "local_step": step_identity,
            "guide_batch_backend": guide_batch_backend(guide),
        },
        "system": {
            "n_particles": system.n_particles,
            "rod_length": system.rod_length,
            "center": system.center,
        },
        "scheduled_move": scheduled_identity,
        "caller": checkpoint_identity,
    }
    normalized = to_jsonable(identity)
    if not isinstance(normalized, dict):
        raise TypeError("checkpoint identity must normalize to a mapping")
    return normalized
