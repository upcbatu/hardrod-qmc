from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from hrdmc.sampling.mala import mala_step


class _QuadraticGuide:
    def valid_batch(self, positions: np.ndarray) -> np.ndarray:
        return np.all(np.diff(positions, axis=1) > 0.0, axis=1)

    def batch_log_value(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return -0.5 * np.sum(positions * positions, axis=1), self.valid_batch(positions)

    def batch_grad_lap_local(self, positions: np.ndarray) -> tuple[np.ndarray, ...]:
        valid = self.valid_batch(positions)
        return (
            -positions,
            -np.ones_like(positions),
            np.sum(positions * positions, axis=1),
            valid,
        )


@pytest.mark.parametrize(
    ("dt", "limiter", "expected"),
    [
        (0.01, "none", [-2.05530930014476, -0.48757599702760]),
        (0.05, "none", [-2.06839671446161, -0.45839942468856]),
        (0.01, "umrigar", [-2.05569405787409, -0.48758223145126]),
        (0.05, "umrigar", [-2.07678873615165, -0.45855375155690]),
    ],
)
def test_mala_transition_and_next_rng_draw_are_frozen(
    dt: float, limiter: str, expected: list[float]
) -> None:
    positions = np.asarray([[-2.0, -0.5], [-1.0, 0.0], [0.5, 2.0]])
    rng = np.random.default_rng(2468)
    result = mala_step(
        rng,
        positions,
        cast(Any, _QuadraticGuide()),
        dt=dt,
        drift_limiter=limiter,
        local_energies=np.sum(positions * positions, axis=1),
    )

    np.testing.assert_array_equal(result.accepted, [True, True, True])
    np.testing.assert_allclose(result.positions[0], expected, rtol=0.0, atol=1.0e-14)
    assert rng.random() == pytest.approx(0.9520529482609905)
