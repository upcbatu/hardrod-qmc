from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.estimators.forward_walking.contributions import weighted_density_profile
from hrdmc.estimators.forward_walking.transported import (
    estimate_transported_auxiliary_forward_walking,
)
from hrdmc.sampling.dmc.telemetry import DMCTransportEvent

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "results/dmc/final_matrix/thesis_5seed_all_optimized_final_v1"
EXPECTED_LAGS = {
    "N10_A0": ([8000, 12000, 16000, 20000], [800, 1600, 2800]),
    "N10_A0.1": ([8000, 12000, 16000, 20000], [800, 1600, 2800]),
    "N10_A1": ([3200, 4800, 6400, 8000], [1600, 3200, 5600]),
    "N10_A10": ([40000, 80000, 160000], [16000, 32000, 56000]),
    "N20_A0": ([8000, 12000, 16000, 20000], [800, 1600, 2800]),
    "N20_A0.1": ([8000, 12000, 16000, 20000], [800, 1600, 2800]),
    "N20_A1": ([32000, 48000, 64000, 80000], [3200, 6400, 11200]),
    "N20_A10": ([20000, 40000, 60000], [8000, 16000, 28000]),
}


def test_lag_zero_equals_the_mixed_r2_reference() -> None:
    events = [_event(step, np.asarray([[1.0], [3.0]])) for step in range(1, 5)]
    result = estimate_transported_auxiliary_forward_walking(
        events,
        PureWalkingConfig(
            lag_steps=(0,),
            block_size_steps=2,
            min_block_count=2,
            min_walker_weight_ess=1.0,
        ),
        mixed_r2_reference=5.0,
        mixed_rms_radius_reference=np.sqrt(5.0),
    )

    np.testing.assert_allclose(result.observable_results["r2"].values_by_lag[0], 5.0)


def test_parent_map_transport_matches_manual_clone_and_kill() -> None:
    events = [
        _event(1, np.asarray([[1.0], [5.0]])),
        _event(2, np.asarray([[9.0], [9.0]]), parents=np.asarray([1, 1])),
    ]
    result = estimate_transported_auxiliary_forward_walking(
        events,
        PureWalkingConfig(
            lag_steps=(0, 1), block_size_steps=1, min_block_count=1, min_walker_weight_ess=1.0
        ),
    )

    np.testing.assert_allclose(result.observable_results["r2"].values_by_lag[1], 25.0)


def test_composed_parent_maps_are_associative() -> None:
    first = np.asarray([2, 0, 2])
    second = np.asarray([1, 1, 0])
    events = [
        _event(1, np.asarray([[1.0], [2.0], [3.0]])),
        _event(2, np.zeros((3, 1)), parents=first),
        _event(3, np.zeros((3, 1)), parents=second),
    ]
    result = estimate_transported_auxiliary_forward_walking(
        events,
        PureWalkingConfig(
            lag_steps=(0, 2), block_size_steps=1, min_block_count=1, min_walker_weight_ess=1.0
        ),
    )

    expected = np.mean(np.asarray([1.0, 4.0, 9.0])[first[second]])
    np.testing.assert_allclose(result.observable_results["r2"].values_by_lag[2], expected)


def test_weight_gauge_shift_cancels() -> None:
    config = PureWalkingConfig(
        lag_steps=(0,), block_size_steps=2, min_block_count=2, min_walker_weight_ess=1.0
    )
    plain = estimate_transported_auxiliary_forward_walking(
        [_event(step, np.asarray([[1.0], [3.0]])) for step in range(1, 5)], config
    )
    shifted = estimate_transported_auxiliary_forward_walking(
        [_event(step, np.asarray([[1.0], [3.0]]), gauge=17.0) for step in range(1, 5)],
        config,
    )

    np.testing.assert_allclose(
        plain.observable_results["r2"].values_by_lag[0],
        shifted.observable_results["r2"].values_by_lag[0],
    )


def test_rao_blackwell_r2_is_computed_by_the_estimator() -> None:
    result = estimate_transported_auxiliary_forward_walking(
        [_event(1, np.asarray([[9.0, 11.0], [-2.0, 2.0]]))],
        PureWalkingConfig(
            lag_steps=(0,),
            block_size_steps=1,
            min_block_count=1,
            min_walker_weight_ess=1.0,
            observable_source="r2_rb",
            r2_rb_com_variance=0.25,
        ),
    )

    np.testing.assert_allclose(result.observable_results["r2"].values_by_lag[0], 2.75)


def test_rao_blackwell_density_is_normalized_even_and_shift_invariant() -> None:
    rng = np.random.default_rng(17)
    positions = rng.normal(size=(64, 20))
    weights = rng.random(64)
    weights /= np.sum(weights)
    edges = np.linspace(-10.0, 10.0, 401)
    kwargs = {
        "bin_edges": edges,
        "walker_weights": weights,
        "source": "com_rao_blackwell",
        "com_variance": 1.0 / 40.0,
        "parity_average": True,
    }

    density = weighted_density_profile(positions, **kwargs)
    shifted = weighted_density_profile(positions + 31.0, **kwargs)

    np.testing.assert_allclose(np.sum(density * np.diff(edges)), 20.0, atol=1.0e-12)
    np.testing.assert_allclose(density, density[::-1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(density, shifted, rtol=0.0, atol=1.0e-13)


def test_density_uses_its_own_lag_ladder_and_collection_stride() -> None:
    events = [_event(step, np.asarray([[-0.5, 0.5], [-0.5, 1.5]])) for step in range(1, 9)]
    result = estimate_transported_auxiliary_forward_walking(
        events,
        PureWalkingConfig(
            lag_steps=(0, 1),
            density_lag_steps=(0, 2),
            density_collection_stride_steps=2,
            density_plateau_window_lag_count=2,
            block_size_steps=1,
            min_block_count=1,
            min_walker_weight_ess=1.0,
            observables=("r2", "density"),
            density_bin_edges=np.asarray([-1.0, 0.0, 1.0, 2.0]),
        ),
    )

    assert tuple(result.observable_results["r2"].lag_steps) == (0, 1)
    assert tuple(result.observable_results["density"].lag_steps) == (0, 2)
    assert result.observable_results["density"].block_count_by_lag[0] == 4
    assert result.observable_results["density"].block_count_by_lag[2] == 3


def test_tracked_matrix_uses_the_predeclared_observable_lags() -> None:
    rows = json.loads((MATRIX / "final_matrix_summary.json").read_text(encoding="utf-8"))["rows"]

    assert {
        row["case"]: (row["r2_selected_window_lags"], row["density_selected_window_lags"])
        for row in rows
    } == EXPECTED_LAGS


def _event(
    step: int,
    positions: np.ndarray,
    *,
    parents: np.ndarray | None = None,
    gauge: float = 0.0,
) -> DMCTransportEvent:
    resolved_parents = np.arange(positions.shape[0]) if parents is None else parents
    return DMCTransportEvent(
        step_id=step,
        production_step_id=step,
        positions=positions,
        local_energy_per_walker=np.sum(positions * positions, axis=1),
        log_weights_pre_resample=np.zeros(positions.shape[0]),
        log_weights_post_resample=np.zeros(positions.shape[0]),
        parent_indices=resolved_parents.astype(np.int64),
        resampled=not np.array_equal(resolved_parents, np.arange(positions.shape[0])),
        weight_gauge_shift=gauge,
    )
