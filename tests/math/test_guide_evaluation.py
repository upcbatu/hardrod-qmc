from __future__ import annotations

import numpy as np

from hrdmc.system.geometry import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.trial.guide import ReducedTGHardRodGuide


def _finite_difference_log_derivatives(
    guide: ReducedTGHardRodGuide,
    positions: np.ndarray,
    *,
    step: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    center = guide.log_value(positions)
    gradient = np.zeros_like(positions)
    laplacian = np.zeros_like(positions)
    for particle in range(positions.size):
        plus = positions.copy()
        minus = positions.copy()
        plus[particle] += step
        minus[particle] -= step
        gradient[particle] = (guide.log_value(plus) - guide.log_value(minus)) / (2.0 * step)
        laplacian[particle] = (
            guide.log_value(plus) - 2.0 * center + guide.log_value(minus)
        ) / step**2
    return gradient, laplacian


def test_guide_derivatives_match_finite_differences() -> None:
    guide = ReducedTGHardRodGuide(
        system=OpenLineHardRodSystem(n_particles=3, rod_length=0.4),
        trap=HarmonicTrap(omega=0.1),
        alpha=0.07,
    )
    positions = np.asarray([-1.4, 0.1, 1.5])
    gradient, laplacian = _finite_difference_log_derivatives(guide, positions)

    np.testing.assert_allclose(guide.grad_log_value(positions), gradient, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(guide.lap_log_value(positions), laplacian, rtol=5.0e-5, atol=5.0e-5)


def test_local_energy_uses_log_derivatives_and_the_trap() -> None:
    trap = HarmonicTrap(omega=0.2)
    guide = ReducedTGHardRodGuide(
        system=OpenLineHardRodSystem(n_particles=4, rod_length=0.5),
        trap=trap,
        alpha=0.08,
    )
    positions = np.asarray([-2.0, -0.8, 0.7, 2.1])
    gradient = guide.grad_log_value(positions)
    laplacian = guide.lap_log_value(positions)

    expected = -0.5 * np.sum(laplacian + gradient * gradient) + trap.total(positions)
    np.testing.assert_allclose(guide.local_energy(positions), expected)


def test_harmonic_unit_local_energy_matches_the_closed_form() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=HarmonicTrap(omega=1.0),
        alpha=1.0,
    )
    positions = np.asarray([-2.0, -0.8, 0.7, 2.1])
    free_gaps = np.diff(positions) - system.rod_length
    indices = np.arange(1, system.n_particles, dtype=float)
    expected = (
        system.n_particles**2 / 2.0
        + system.rod_length**2 * system.n_particles * (system.n_particles**2 - 1) / 24.0
        + system.rod_length / 2.0 * np.sum(indices * (system.n_particles - indices) * free_gaps)
    )

    np.testing.assert_allclose(guide.local_energy(positions), expected)


def test_relative_width_derivatives_remain_self_consistent() -> None:
    trap = HarmonicTrap(omega=1.0)
    guide = ReducedTGHardRodGuide(
        system=OpenLineHardRodSystem(n_particles=4, rod_length=0.5),
        trap=trap,
        alpha=1.0,
        relative_alpha=2.0,
    )
    positions = np.asarray([-2.0, -0.8, 0.7, 2.1])
    gradient, laplacian = _finite_difference_log_derivatives(guide, positions)

    np.testing.assert_allclose(guide.grad_log_value(positions), gradient, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(guide.lap_log_value(positions), laplacian, rtol=5.0e-5, atol=5.0e-5)
    expected = -0.5 * np.sum(laplacian + gradient * gradient) + trap.total(positions)
    np.testing.assert_allclose(guide.local_energy(positions), expected, rtol=5.0e-5, atol=5.0e-5)


def test_batch_evaluation_matches_scalar_evaluation() -> None:
    guide = ReducedTGHardRodGuide(
        system=OpenLineHardRodSystem(n_particles=4, rod_length=0.5),
        trap=HarmonicTrap(omega=0.2),
        alpha=0.08,
    )
    positions = np.asarray([[-2.0, -0.8, 0.7, 2.1], [-2.4, -0.5, 0.8, 2.7], [-1.0, -0.7, 0.6, 1.8]])
    log_values, log_finite = guide.batch_log_value(positions)
    gradient, laplacian, local, finite = guide.batch_grad_lap_local(positions)

    expected_valid = np.asarray([True, True, False])
    np.testing.assert_array_equal(log_finite, expected_valid)
    np.testing.assert_array_equal(finite, expected_valid)
    for index in np.flatnonzero(expected_valid):
        np.testing.assert_allclose(log_values[index], guide.log_value(positions[index]))
        np.testing.assert_allclose(gradient[index], guide.grad_log_value(positions[index]))
        np.testing.assert_allclose(laplacian[index], guide.lap_log_value(positions[index]))
        np.testing.assert_allclose(local[index], guide.local_energy(positions[index]))


def test_default_relative_width_matches_an_explicit_unit_width() -> None:
    system = OpenLineHardRodSystem(n_particles=4, rod_length=0.5)
    trap = HarmonicTrap(omega=1.0)
    positions = np.asarray([[-2.0, -0.8, 0.7, 2.1]])
    implicit = ReducedTGHardRodGuide(system=system, trap=trap, alpha=1.0)
    explicit = ReducedTGHardRodGuide(system=system, trap=trap, alpha=1.0, relative_alpha=1.0)

    np.testing.assert_allclose(
        implicit.batch_log_value(positions)[0], explicit.batch_log_value(positions)[0]
    )
    np.testing.assert_allclose(
        implicit.batch_grad_lap_local(positions)[2],
        explicit.batch_grad_lap_local(positions)[2],
    )
