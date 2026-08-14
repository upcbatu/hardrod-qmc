from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.sampling.mala import mala_step
from hrdmc.sampling.vmc.results import (
    VMCAttemptCounts,
    VMCObserver,
    VMCStreamingResult,
    VMCTransitionEvent,
)
from hrdmc.sampling.vmc.transitions import (
    VMCConfig,
    random_scan_rwm_step,
    require_batched_guide,
)
from hrdmc.trial.guide import BatchedDMCGuide

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
@dataclass(frozen=True)
class _GuideState:
    log_values: FloatArray
    gradients: FloatArray
    local_energies: FloatArray
    valid: BoolArray
def run_vmc_streaming(
    *,
    initial_positions: FloatArray,
    guide: BatchedDMCGuide,
    config: VMCConfig,
    seed: int,
    observer: VMCObserver | None = None,
) -> VMCStreamingResult:
    """Run bounded-memory batched VMC against the guide-squared target."""
    config.validate()
    batched_guide = require_batched_guide(guide)
    positions = _require_initial_positions(initial_positions, config.walkers)
    initial_state = _evaluate_state(batched_guide, positions)
    _require_finite_state("initial", positions, initial_state)
    log_values = initial_state.log_values
    local_energies = initial_state.local_energies
    rng = np.random.default_rng(seed)
    burn_attempts = VMCAttemptCounts()
    production_attempts = VMCAttemptCounts()
    last_state = initial_state
    start = time.perf_counter()
    for phase, phase_steps in (
        ("burn_in", config.burn_in_steps),
        ("production", config.production_steps),
    ):
        for phase_step in range(1, phase_steps + 1):
            previous_positions = positions
            if config.method == "rwm":
                transition = random_scan_rwm_step(
                    rng,
                    positions,
                    batched_guide,
                    config.step_size,
                    log_values,
                )
                positions = transition.positions
                log_values = transition.log_values
                step_attempts = _attempts_from_masks(
                    accepted=transition.accepted,
                    invalid=transition.invalid_proposal,
                    nonfinite=transition.nonfinite_proposal,
                    metropolis_rejected=transition.metropolis_rejected,
                )
            else:
                transition = mala_step(
                    rng,
                    positions,
                    batched_guide,
                    config.dt,
                    local_energies,
                    drift_limiter=config.drift_limiter,
                )
                positions = transition.positions
                local_energies = transition.local_energies
                step_attempts = _attempts_from_masks(
                    accepted=transition.accepted,
                    invalid=transition.domain_invalid_proposal,
                    nonfinite=transition.nonfinite_proposal,
                    metropolis_rejected=(
                        transition.metropolis_rejected & (~transition.nonfinite_proposal)
                    ),
                )
            if phase == "burn_in":
                burn_attempts = burn_attempts.plus(step_attempts)
            else:
                production_attempts = production_attempts.plus(step_attempts)
                if observer is not None:
                    last_state = _evaluate_state(batched_guide, positions)
                    _require_finite_state("production", positions, last_state)
                    log_values = last_state.log_values
                    local_energies = last_state.local_energies
                    observer.record_vmc_transition(
                        _transition_event(
                            state=last_state,
                            previous_positions=previous_positions,
                            positions=positions,
                            production_step=phase_step,
                        )
                    )
    _require_attempt_total(
        "burn-in",
        burn_attempts,
        config.walkers * config.burn_in_steps,
    )
    _require_attempt_total(
        "production",
        production_attempts,
        config.walkers * config.production_steps,
    )
    if observer is None:
        last_state = _evaluate_state(batched_guide, positions)
        _require_finite_state("final", positions, last_state)
        log_values = last_state.log_values
        local_energies = last_state.local_energies
    wall_seconds = time.perf_counter() - start
    return VMCStreamingResult(
        burn_in_attempts=burn_attempts,
        production_attempts=production_attempts,
        production_sample_count=config.walkers * config.production_steps,
        seed=int(seed),
        wall_seconds=wall_seconds,
        metadata=_metadata(
            config=config,
            particles=positions.shape[1],
        ),
    )
def _evaluate_state(guide: BatchedDMCGuide, positions: FloatArray) -> _GuideState:
    log_values, log_finite = guide.batch_log_value(positions)
    gradients, _, local_energies, derivative_finite = guide.batch_grad_lap_local(positions)
    domain_valid = guide.valid_batch(positions)
    state = _GuideState(
        log_values=np.asarray(log_values, dtype=float),
        gradients=np.asarray(gradients, dtype=float),
        local_energies=np.asarray(local_energies, dtype=float),
        valid=(
            np.asarray(domain_valid, dtype=bool)
            & np.asarray(log_finite, dtype=bool)
            & np.asarray(derivative_finite, dtype=bool)
        ),
    )
    _require_state_shapes(positions, state)
    return state
def _require_initial_positions(positions: FloatArray, walkers: int) -> FloatArray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[0] != walkers or values.shape[1] <= 0:
        raise ValueError("initial_positions must have shape (config.walkers, particles)")
    if not np.all(np.isfinite(values)):
        raise ValueError("initial_positions must be finite")
    return values.copy()
def _require_state_shapes(positions: FloatArray, state: _GuideState) -> None:
    walkers = positions.shape[0]
    if state.gradients.shape != positions.shape:
        raise ValueError("batch_grad_lap_local returned a gradient with the wrong shape")
    for label, values in (
        ("batch_log_value", state.log_values),
        ("local energy", state.local_energies),
        ("guide validity", state.valid),
    ):
        if values.shape != (walkers,):
            raise ValueError(f"{label} must return one value per walker")
def _require_finite_state(
    label: str,
    positions: FloatArray,
    state: _GuideState,
) -> None:
    finite = (
        state.valid
        & np.all(np.isfinite(positions), axis=1)
        & np.isfinite(state.log_values)
        & np.all(np.isfinite(state.gradients), axis=1)
        & np.isfinite(state.local_energies)
    )
    if not np.all(finite):
        raise ValueError(f"{label} VMC ensemble contains invalid or non-finite walkers")
def _attempts_from_masks(
    *,
    accepted: BoolArray,
    invalid: BoolArray,
    nonfinite: BoolArray,
    metropolis_rejected: BoolArray,
) -> VMCAttemptCounts:
    masks = tuple(
        np.asarray(values, dtype=bool)
        for values in (accepted, invalid, nonfinite, metropolis_rejected)
    )
    shape = masks[0].shape
    if len(shape) != 1 or any(values.shape != shape for values in masks[1:]):
        raise ValueError("transition outcome masks must share shape (walkers,)")
    partition = sum(values.astype(np.int8) for values in masks)
    if not np.all(partition == 1):
        raise RuntimeError("transition outcomes must form an exact attempt partition")
    return VMCAttemptCounts(
        attempted=shape[0],
        accepted=int(np.sum(masks[0])),
        invalid_proposals=int(np.sum(masks[1])),
        nonfinite_proposals=int(np.sum(masks[2])),
        metropolis_rejections=int(np.sum(masks[3])),
    )
def _transition_event(
    *,
    state: _GuideState,
    previous_positions: FloatArray,
    positions: FloatArray,
    production_step: int,
) -> VMCTransitionEvent:
    return VMCTransitionEvent(
        production_step=production_step,
        previous_positions=_readonly_copy(previous_positions),
        positions=_readonly_copy(positions),
        gradients=_readonly_copy(state.gradients),
        local_energies=_readonly_copy(state.local_energies),
        valid=_readonly_copy(state.valid),
    )
def _readonly_copy(values: NDArray[Any]) -> NDArray[Any]:
    copied = np.asarray(values).copy()
    copied.flags.writeable = False
    return copied
def _require_attempt_total(
    label: str,
    attempts: VMCAttemptCounts,
    expected: int,
) -> None:
    if attempts.attempted != expected:
        raise RuntimeError(f"{label} VMC attempt count does not match the run plan")
def _metadata(
    *,
    config: VMCConfig,
    particles: int,
) -> dict[str, object]:
    return {
        "method": config.method,
        "proposal": (
            "random_scan_single_particle_symmetric_rwm"
            if config.method == "rwm"
            else "whole_configuration_metropolis_drift_diffusion"
        ),
        "walkers": config.walkers,
        "particles": particles,
        "burn_in_steps": config.burn_in_steps,
        "production_steps": config.production_steps,
        "dt": config.dt if config.method == "mala" else None,
        "step_size": config.step_size if config.method == "rwm" else None,
        "drift_limiter": config.drift_limiter,
        "sorting_enabled": False,
        "branching_enabled": False,
        "walker_weights_enabled": False,
        "population_resampling_enabled": False,
    }
