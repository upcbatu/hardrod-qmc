from __future__ import annotations

import numpy as np
import pytest

from hrdmc.systems import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.wavefunctions import ReducedTGHardRodGuide, trapped_guide_kernels


def test_reduced_tg_guide_rejects_unordered_and_contact_configurations() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    guide = ReducedTGHardRodGuide(system=system, trap=HarmonicTrap(omega=0.1), alpha=0.07)

    assert guide.log_value(np.array([0.0, -1.0, 1.0])) == float("-inf")
    assert guide.log_value(np.array([0.0, 1.0])) == float("-inf")
    assert guide.log_value(np.array([-0.4, 0.0, 1.0])) == float("-inf")
    assert np.isfinite(guide.log_value(np.array([-1.4, 0.1, 1.5])))


def test_reduced_tg_guide_derivatives_match_finite_difference() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=HarmonicTrap(omega=0.1),
        alpha=0.07,
        pair_power=1.0,
    )
    positions = np.array([-1.4, 0.1, 1.5])
    step = 1e-5
    log0 = guide.log_value(positions)

    fd_grad = np.zeros(system.n_particles)
    fd_lap = np.zeros(system.n_particles)
    for particle in range(system.n_particles):
        plus = positions.copy()
        minus = positions.copy()
        plus[particle] += step
        minus[particle] -= step
        fd_grad[particle] = (guide.log_value(plus) - guide.log_value(minus)) / (2.0 * step)
        fd_lap[particle] = (guide.log_value(plus) - 2.0 * log0 + guide.log_value(minus)) / step**2

    np.testing.assert_allclose(guide.grad_log_value(positions), fd_grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(guide.lap_log_value(positions), fd_lap, rtol=5e-5, atol=5e-5)


def test_reduced_tg_guide_local_energy_uses_guide_derivatives_and_trap() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    trap = HarmonicTrap(omega=0.2)
    guide = ReducedTGHardRodGuide(system=system, trap=trap, alpha=0.08, pair_power=1.0)
    positions = np.array([-2.0, -0.8, 0.7, 2.1])

    grad = guide.grad_log_value(positions)
    lap = guide.lap_log_value(positions)
    expected = -np.sum(lap + grad * grad) + trap.total(positions)

    np.testing.assert_allclose(guide.local_energy(positions), expected)


def test_reduced_tg_batch_methods_match_scalar_methods() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    trap = HarmonicTrap(omega=0.2)
    guide = ReducedTGHardRodGuide(system=system, trap=trap, alpha=0.08, pair_power=1.0)
    positions = np.array(
        [
            [-2.0, -0.8, 0.7, 2.1],
            [-2.4, -0.5, 0.8, 2.7],
            [-1.0, -0.7, 0.6, 1.8],
        ]
    )

    log_values, log_finite = guide.batch_log_value(positions)
    grad, lap, local, finite = guide.batch_grad_lap_local(positions)
    valid = guide.valid_batch(positions)

    assert guide.batch_backend in {"numba", "python"}
    expected_valid = np.array([True, True, False])
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(log_finite, expected_valid)
    np.testing.assert_array_equal(finite, expected_valid)
    for idx in np.flatnonzero(expected_valid):
        np.testing.assert_allclose(log_values[idx], guide.log_value(positions[idx]))
        np.testing.assert_allclose(grad[idx], guide.grad_log_value(positions[idx]))
        np.testing.assert_allclose(lap[idx], guide.lap_log_value(positions[idx]))
        np.testing.assert_allclose(local[idx], guide.local_energy(positions[idx]))


def test_reduced_tg_batch_methods_have_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trapped_guide_kernels, "NUMBA_AVAILABLE", False)
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4)
    guide = ReducedTGHardRodGuide(system=system, trap=HarmonicTrap(omega=0.1), alpha=0.07)
    positions = np.array([[-1.4, 0.1, 1.5], [0.0, -1.0, 1.0]])

    log_values, finite = guide.batch_log_value(positions)
    grad, lap, local, local_finite = guide.batch_grad_lap_local(positions)

    assert guide.batch_backend == "python"
    assert np.isfinite(log_values[0])
    assert finite.tolist() == [True, False]
    assert grad.shape == positions.shape
    assert lap.shape == positions.shape
    assert local.shape == (positions.shape[0],)
    assert local_finite.tolist() == [True, False]


def test_reduced_tg_guide_requires_matching_trap_center() -> None:
    system = OpenLineHardRodSystem(n_particles=3, rod_length=0.4, center=0.0)

    with pytest.raises(ValueError, match="centers"):
        ReducedTGHardRodGuide(system=system, trap=HarmonicTrap(omega=0.1, center=0.2), alpha=0.07)
