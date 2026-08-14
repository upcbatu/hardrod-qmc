from __future__ import annotations

import numpy as np

from hrdmc.trial.kernel import (
    _reduced_tg_grad_lap_local_batch_python,
    reduced_tg_grad_lap_local_batch,
)


def test_numba_dispatch_matches_the_independent_python_kernel() -> None:
    positions = np.asarray([[-2.5, -0.8, 0.9, 2.7], [-3.1, -1.4, 0.2, 1.9]])
    offsets = np.asarray([-1.5, -0.5, 0.5, 1.5])
    kwargs = {
        "rod_length": 1.0,
        "alpha": 1.0,
        "relative_alpha": 1.6224444406063525,
        "center": 0.0,
        "omega2": 1.0,
        "pair_power": 1.0,
    }

    dispatched = reduced_tg_grad_lap_local_batch(positions, offsets, **kwargs)
    reference = _reduced_tg_grad_lap_local_batch_python(positions, offsets, **kwargs)

    for observed, expected in zip(dispatched[:-1], reference[:-1], strict=True):
        np.testing.assert_allclose(observed, expected, rtol=1.0e-9, atol=0.0)
    np.testing.assert_array_equal(dispatched[-1], reference[-1])
