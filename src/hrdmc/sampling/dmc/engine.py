from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.artifacts.progress import ProgressBar
from hrdmc.artifacts.schema import to_jsonable
from hrdmc.sampling.dmc.accumulator import DMCStreamingState
from hrdmc.sampling.dmc.guide_api import (
    guide_batch_backend,
)
from hrdmc.sampling.dmc.population import (
    effective_sample_size,
    maybe_resample_population_with_indices,
    recenter_log_weights,
)
from hrdmc.sampling.dmc.results import DMCStreamingSummary
from hrdmc.sampling.dmc.telemetry import (
    DMCTransportEvent,
    DMCTransportObserver,
)
from hrdmc.sampling.dmc.transitions import (
    DMCConfig,
    DMCStep,
    advance_local_step,
    metropolis_drift_diffusion_step,
)
from hrdmc.system.geometry import OpenLineHardRodSystem
from hrdmc.trial.guide import DMCGuide

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _RunSetup:
    config: DMCConfig
    rng: np.random.Generator
    stepper: DMCStep
    total_steps: int
    checkpoint_file: Path | None
    checkpoint_every_steps: int | None
    resume_identity: dict[str, Any] | None
    state: DMCStreamingState


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
    progress: ProgressBar | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume: bool = False,
    checkpoint_identity: dict[str, Any] | None = None,
    transport_observer: DMCTransportObserver | None = None,
) -> DMCStreamingSummary:
    """Run DMC and accumulate compact observables during production."""
    setup = _prepare_run(
        initial_walkers=initial_walkers,
        guide=guide,
        system=system,
        density_grid=density_grid,
        config=config,
        rng=rng,
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        local_step=local_step,
        checkpoint_path=checkpoint_path,
        checkpoint_every_steps=checkpoint_every_steps,
        resume=resume,
        checkpoint_identity=checkpoint_identity,
        transport_observer=transport_observer,
    )
    state = setup.state
    for step_index in range(state.step_start, setup.total_steps + 1):
        advance = advance_local_step(
            setup.stepper,
            guide,
            setup.rng,
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
            setup.rng,
            threshold_fraction=setup.config.ess_resample_fraction,
        )
        state.record_resample(resampled)
        if transport_observer is not None:
            production_step_id = step_index - burn_in_steps if step_index > burn_in_steps else None
            transport_observer.record_transport_event(
                DMCTransportEvent(
                    step_id=step_index,
                    production_step_id=production_step_id,
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
            setup.checkpoint_file is not None
            and setup.checkpoint_every_steps is not None
            and (step_index % setup.checkpoint_every_steps == 0 or step_index == setup.total_steps)
        ):
            if setup.resume_identity is None:
                raise RuntimeError("checkpoint identity was not constructed")
            state.save_checkpoint(
                setup.checkpoint_file,
                step_index=step_index,
                rng=setup.rng,
                dt=dt,
                burn_in_steps=burn_in_steps,
                production_steps=production_steps,
                store_every=store_every,
                system=system,
                resume_identity=setup.resume_identity,
            )
    summary = state.to_summary(
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        ess_resample_fraction=setup.config.ess_resample_fraction,
        guide=guide,
    )
    summary.metadata["drift_limiter"] = setup.config.drift_limiter
    return summary


def _prepare_run(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    density_grid: FloatArray,
    config: DMCConfig | None,
    rng: np.random.Generator | None,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
    local_step: DMCStep | None,
    checkpoint_path: str | Path | None,
    checkpoint_every_steps: int | None,
    resume: bool,
    checkpoint_identity: dict[str, Any] | None,
    transport_observer: DMCTransportObserver | None,
) -> _RunSetup:
    cfg = config or DMCConfig()
    cfg.validate()
    _validate_run_inputs(dt, burn_in_steps, production_steps, store_every)
    if checkpoint_every_steps is not None and checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive")
    generator = np.random.default_rng() if rng is None else rng
    stepper = _resolve_local_step(cfg, local_step)
    grid = np.asarray(density_grid, dtype=float)
    total_steps = burn_in_steps + production_steps
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    checkpointing_active = resume or (
        checkpoint_file is not None and checkpoint_every_steps is not None
    )
    _validate_checkpoint_options(
        checkpoint_file=checkpoint_file,
        checkpoint_every_steps=checkpoint_every_steps,
        resume=resume,
        checkpointing_active=checkpointing_active,
        checkpoint_identity=checkpoint_identity,
        transport_observer=transport_observer,
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
            dt=dt,
            burn_in_steps=burn_in_steps,
            production_steps=production_steps,
            store_every=store_every,
            checkpoint_identity=checkpoint_identity,
        )
    )
    state = _initial_state(
        resume=resume,
        checkpoint_file=checkpoint_file,
        resume_identity=resume_identity,
        rng=generator,
        initial_walkers=initial_walkers,
        guide=guide,
        system=system,
        grid=grid,
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
    )
    return _RunSetup(
        config=cfg,
        rng=generator,
        stepper=stepper,
        total_steps=total_steps,
        checkpoint_file=checkpoint_file,
        checkpoint_every_steps=checkpoint_every_steps,
        resume_identity=resume_identity,
        state=state,
    )


def _validate_checkpoint_options(
    *,
    checkpoint_file: Path | None,
    checkpoint_every_steps: int | None,
    resume: bool,
    checkpointing_active: bool,
    checkpoint_identity: dict[str, Any] | None,
    transport_observer: DMCTransportObserver | None,
) -> None:
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


def _initial_state(
    *,
    resume: bool,
    checkpoint_file: Path | None,
    resume_identity: dict[str, Any] | None,
    rng: np.random.Generator,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    grid: FloatArray,
    dt: float,
    burn_in_steps: int,
    production_steps: int,
    store_every: int,
) -> DMCStreamingState:
    if not resume:
        return DMCStreamingState.from_initial(
            initial_walkers=initial_walkers,
            guide=guide,
            system=system,
            density_grid=grid,
        )
    if checkpoint_file is None or resume_identity is None:
        raise RuntimeError("resume configuration was not constructed")
    return DMCStreamingState.from_checkpoint(
        checkpoint_file,
        rng=rng,
        dt=dt,
        burn_in_steps=burn_in_steps,
        production_steps=production_steps,
        store_every=store_every,
        system=system,
        density_grid=grid,
        resume_identity=resume_identity,
    )


def _resolve_local_step(
    config: DMCConfig,
    local_step: DMCStep | None,
) -> DMCStep:
    if local_step is not None:
        if config.drift_limiter != "none":
            raise ValueError("configured drift_limiter cannot be applied to a custom local_step")
        return local_step
    return partial(
        metropolis_drift_diffusion_step,
        drift_limiter=config.drift_limiter,
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


def _build_resume_identity(
    *,
    initial_walkers: FloatArray,
    guide: DMCGuide,
    system: OpenLineHardRodSystem,
    config: DMCConfig,
    local_step: DMCStep | None,
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
            "method": "metropolis",
            "drift_limiter": config.drift_limiter,
        }
        if local_step is None
        else {
            "kind": "callable",
            "module": getattr(local_step, "__module__", type(local_step).__module__),
            "qualname": getattr(local_step, "__qualname__", type(local_step).__qualname__),
        }
    )
    identity = {
        "engine": "local_importance_sampled_dmc",
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
        "caller": checkpoint_identity,
    }
    normalized = to_jsonable(identity)
    if not isinstance(normalized, dict):
        raise TypeError("checkpoint identity must normalize to a mapping")
    return normalized
