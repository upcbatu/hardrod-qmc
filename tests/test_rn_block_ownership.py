from __future__ import annotations

import numpy as np
import pytest

from hrdmc.monte_carlo.dmc.rn_block import (
    RNBlockDMCConfig,
    RNBlockDMCResult,
    log_collective_mixture_density,
    rn_log_increment,
    sample_collective_mixture,
)
from hrdmc.systems import (
    HarmonicMehlerKernel,
    HarmonicTrap,
    OpenHardRodTrapPrimitiveKernel,
    OrderedHarmonicMehlerKernel,
)
from hrdmc.systems.open_line import OpenLineHardRodSystem


def test_rn_config_does_not_own_physical_system_parameters() -> None:
    cfg = RNBlockDMCConfig()
    payload = cfg.__dict__

    for forbidden in ("n_particles", "rod_length", "omega", "center", "target_kernel"):
        assert forbidden not in payload


def test_rn_log_increment_uses_external_target_and_proposal_densities() -> None:
    log_k = np.array([-1.5, -2.0])
    log_q = np.array([-1.0, -3.0])

    np.testing.assert_allclose(rn_log_increment(log_k, log_q), np.array([-0.5, 1.0]))

    with pytest.raises(ValueError, match="matching shapes"):
        rn_log_increment(log_k, np.array([-1.0]))


def test_collective_proposal_density_is_separate_from_system_target_kernel() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    proposal_kernel = HarmonicMehlerKernel(HarmonicTrap(omega=0.1))
    target_weak = OpenHardRodTrapPrimitiveKernel(system, HarmonicTrap(omega=0.1))
    target_strong = OpenHardRodTrapPrimitiveKernel(system, HarmonicTrap(omega=0.2))
    cfg = RNBlockDMCConfig(
        tau_block=0.01,
        component_log_scales=(-0.02, 0.0, 0.02),
        component_probabilities=(0.25, 0.5, 0.25),
    )
    x_old = np.array([[-1.2, 0.0, 1.3], [-1.4, -0.2, 1.1]])
    x_new = np.array([[-1.1, 0.1, 1.4], [-1.3, -0.1, 1.2]])

    log_q = log_collective_mixture_density(cfg, system, proposal_kernel, x_old, x_new)
    log_k_weak = target_weak.log_density(x_old, x_new, cfg.tau_block)
    log_k_strong = target_strong.log_density(x_old, x_new, cfg.tau_block)

    assert log_q.shape == (2,)
    assert log_k_weak.shape == (2,)
    assert log_k_strong.shape == (2,)
    assert not np.allclose(log_k_weak, log_k_strong)
    np.testing.assert_allclose(
        log_collective_mixture_density(cfg, system, proposal_kernel, x_old, x_new),
        log_q,
    )


def test_collective_proposal_sampling_uses_external_proposal_kernel() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    proposal_kernel = HarmonicMehlerKernel(HarmonicTrap(omega=0.1))
    cfg = RNBlockDMCConfig(
        tau_block=0.01,
        component_log_scales=(0.0,),
        component_probabilities=(1.0,),
    )
    x_old = np.array([[-1.2, 0.0, 1.3], [-1.4, -0.2, 1.1]])

    proposal = sample_collective_mixture(
        cfg,
        system,
        proposal_kernel,
        np.random.default_rng(123),
        x_old,
    )

    assert proposal.x_new.shape == x_old.shape
    assert proposal.component_ids.shape == (x_old.shape[0],)
    assert proposal.log_q_forward.shape == (x_old.shape[0],)
    assert proposal.log_q_reverse.shape == (x_old.shape[0],)
    np.testing.assert_array_equal(proposal.component_ids, np.zeros(x_old.shape[0], dtype=np.int64))
    np.testing.assert_allclose(proposal.selected_log_jacobian, np.zeros(x_old.shape[0]))
    np.testing.assert_allclose(
        proposal.log_q_forward,
        proposal_kernel.log_density(x_old, proposal.x_new, cfg.tau_block),
    )


def test_primitive_target_kernel_rejects_wrong_particle_count() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    kernel = OpenHardRodTrapPrimitiveKernel(system, HarmonicTrap(omega=0.1))

    with pytest.raises(ValueError, match="particle count"):
        kernel.log_density(np.array([[0.0, 1.0]]), np.array([[0.1, 1.1]]), tau=0.01)


def test_ordered_harmonic_mehler_kernel_is_exact_tg_limit_only() -> None:
    with pytest.raises(ValueError, match="zero rod length"):
        OrderedHarmonicMehlerKernel(
            OpenLineHardRodSystem(n_particles=3, rod_length=0.1),
            HarmonicTrap(omega=0.1),
        )


def test_ordered_harmonic_mehler_kernel_returns_batch_log_density() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.0)
    kernel = OrderedHarmonicMehlerKernel(system, HarmonicTrap(omega=0.1))
    x_old = np.array([[-1.0, 0.0, 1.0], [-1.2, 0.1, 1.3]])
    x_new = np.array([[-0.9, 0.0, 1.1], [-1.1, 0.2, 1.4]])

    log_density = kernel.log_density(x_old, x_new, tau=0.01)

    assert log_density.shape == (2,)
    assert np.all(np.isfinite(log_density))
    with pytest.raises(ValueError, match="particle count"):
        kernel.log_density(np.array([[0.0, 1.0]]), np.array([[0.1, 1.1]]), tau=0.01)


def test_rn_result_observables_delegate_to_weighted_estimators() -> None:
    result = RNBlockDMCResult(
        snapshots=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
        local_energies=np.array([1.0, 3.0]),
        weights=np.array([0.25, 0.75]),
        metadata={},
    )

    observables = result.estimate_weighted_observables(
        valid_mask=np.array([True, True]),
        grid=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        center=0.0,
        n_particles=2,
    )

    np.testing.assert_allclose(observables.mixed_energy, 2.5)
    np.testing.assert_allclose(observables.r2_radius, 3.25)
    np.testing.assert_allclose(observables.density_integral, 2.0)


def test_rn_result_trace_stationarity_delegates_to_analysis() -> None:
    result = RNBlockDMCResult(
        snapshots=np.array([[-1.0, 1.0]]),
        local_energies=np.array([1.0]),
        weights=np.array([1.0]),
        metadata={},
    )
    times = np.arange(64, dtype=float)
    values = np.ones_like(times)

    diagnostics = result.trace_stationarity_diagnostics(times, values)

    assert diagnostics.stationarity_clean
