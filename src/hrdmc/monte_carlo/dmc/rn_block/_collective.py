from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hrdmc.monte_carlo.dmc.rn_block._collective_coordinates import (
    breathing_log_jacobian,
    inverse_scale_reduced_cloud,
    scale_reduced_cloud,
)
from hrdmc.monte_carlo.dmc.rn_block.config import RNBlockDMCConfig
from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.systems.propagators import ProposalTransitionKernel, TargetTransitionKernel

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CollectiveProposal:
    """Sampled RN collective-block proposal plus its proposal densities."""

    x_new: FloatArray
    component_ids: IntArray
    log_q_forward: FloatArray
    log_q_reverse: FloatArray
    selected_log_component_probability: FloatArray
    selected_log_jacobian: FloatArray


def sample_collective_mixture(
    config: RNBlockDMCConfig,
    system: OpenLineHardRodSystem,
    proposal_kernel: ProposalTransitionKernel,
    rng: np.random.Generator,
    x_old: FloatArray,
) -> CollectiveProposal:
    """Sample from Q_theta without owning the Hamiltonian or target kernel."""

    config.validate()
    x_old = _as_batch(x_old)
    probs = np.asarray(config.component_probabilities, dtype=float)
    log_scales = np.asarray(config.component_log_scales, dtype=float)
    component_ids = rng.choice(log_scales.size, size=x_old.shape[0], p=probs)
    base = proposal_kernel.sample(rng, x_old, config.tau_block)
    x_new = np.empty_like(base)
    selected_log_jac = np.empty(x_old.shape[0], dtype=float)
    for component_id, log_scale in enumerate(log_scales):
        mask = component_ids == component_id
        if not np.any(mask):
            continue
        x_new[mask] = scale_reduced_cloud(system, base[mask], log_scale=float(log_scale))
        selected_log_jac[mask] = breathing_log_jacobian(system, float(log_scale))
    log_q_forward = log_collective_mixture_density(config, system, proposal_kernel, x_old, x_new)
    log_q_reverse = log_collective_mixture_density(config, system, proposal_kernel, x_new, x_old)
    return CollectiveProposal(
        x_new=x_new,
        component_ids=component_ids.astype(np.int64),
        log_q_forward=log_q_forward,
        log_q_reverse=log_q_reverse,
        selected_log_component_probability=np.log(probs[component_ids]),
        selected_log_jacobian=selected_log_jac,
    )


def log_collective_mixture_density(
    config: RNBlockDMCConfig,
    system: OpenLineHardRodSystem,
    proposal_base_kernel: TargetTransitionKernel,
    x_old: FloatArray,
    x_new: FloatArray,
) -> FloatArray:
    """Return log Q_theta(x_new | x_old) for the RN collective proposal."""

    config.validate()
    x_old = _as_batch(x_old)
    x_new = _as_batch(x_new)
    if x_old.shape != x_new.shape:
        raise ValueError("x_old and x_new must have matching shapes")
    probs = np.asarray(config.component_probabilities, dtype=float)
    log_scales = np.asarray(config.component_log_scales, dtype=float)
    terms = np.empty((log_scales.size, x_old.shape[0]), dtype=float)
    for idx, (prob, log_scale) in enumerate(zip(probs, log_scales, strict=True)):
        preimage = inverse_scale_reduced_cloud(system, x_new, log_scale=float(log_scale))
        log_base = proposal_base_kernel.log_density(x_old, preimage, config.tau_block)
        inverse_log_jac = -breathing_log_jacobian(system, float(log_scale))
        terms[idx] = np.log(float(prob)) + log_base + inverse_log_jac
    return _logsumexp(terms, axis=0)


def _logsumexp(values: FloatArray, axis: int) -> FloatArray:
    vmax = np.max(values, axis=axis, keepdims=True)
    finite_max = np.isfinite(vmax)
    shifted = np.zeros_like(values, dtype=float)
    mask = np.broadcast_to(finite_max, values.shape)
    vmax_broadcast = np.broadcast_to(vmax, values.shape)
    shifted[mask] = np.exp(values[mask] - vmax_broadcast[mask])
    summed = np.sum(shifted, axis=axis)
    base = np.squeeze(vmax, axis=axis)
    valid = np.squeeze(finite_max, axis=axis) & (summed > 0.0)
    out = np.full_like(base, -np.inf, dtype=float)
    out[valid] = base[valid] + np.log(summed[valid])
    return out


def _as_batch(x: FloatArray) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("configuration array must be one- or two-dimensional")
    return arr
