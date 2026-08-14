"""Assert the frozen thesis numbers. Tolerance oracles: the kernels are numba-compiled."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from hrdmc.estimators.energy_response import (
    PairedEnergyResponsePoint,
    paired_trap_r2_from_energy_response,
)
from hrdmc.sampling.dmc.run import run_streaming_seed
from hrdmc.system.settings import DMCRunControls, parse_case
from hrdmc.trial.kernel import (
    _reduced_tg_grad_lap_local_batch_python,
    reduced_tg_grad_lap_local_batch,
)
from hrdmc.trial.numba_backend import numba_backend_name
from hrdmc.uncertainty.forward_walking.run import _validate_controls as _validate_fw_controls
from hrdmc.uncertainty.population import _require_common_identity

RELATIVE_TOLERANCE = 1.0e-9

# Measured on this tree and on pre-compaction HEAD f85e19d; identical.
PARITY_CASE = "N10_A0"
PARITY_SEED = 7001
PARITY_CONTROLS = {
    "dt": 0.0025,
    "walkers": 64,
    "burn_tau": 0.5,
    "production_tau": 1.0,
    "store_every": 10,
    "grid_extent": 12.0,
    "n_bins": 240,
    "drift_limiter": "none",
}
PARITY_GUIDE_FAMILY = "reduced-tg"
PARITY_EXPECTED = {
    "mixed_energy": 50.0,
    "r2_radius": 6.21019167694895,
    "density_sum": 59.75,
    "density_max": 1.35837890625,
    "density_first_moment": -0.001623953974895407,
}

# Measured on this tree and on pre-compaction HEAD f85e19d.  The historical
# contact correction changes these values by less than the frozen tolerance.
UMRIGAR_PARITY_CASE = "N10_A0.1"
UMRIGAR_PARITY_SEED = 7001
UMRIGAR_PARITY_CONTROLS = {
    "dt": 0.0025,
    "walkers": 64,
    "burn_tau": 0.5,
    "production_tau": 1.0,
    "store_every": 10,
    "grid_extent": 20.0,
    "n_bins": 240,
    "drift_limiter": "umrigar",
    "relative_alpha": 1.0637325870622627,
}
UMRIGAR_PARITY_EXPECTED = {
    "mixed_energy": 56.61405251917114,
    "r2_radius": 7.045090933536422,
    "density_sum": 59.75,
    "density_max": 1.3141958361459367,
}

# main.tex:959-960 prints R_rms/a_ho = 58.85127(74).
HF_SOURCE = Path("data/energy_response/N20_A10_h0025_5seed.csv")
HF_N_PARTICLES = 20
HF_EXPECTED_RADIUS = 58.85127
HF_EXPECTED_STDERR = 0.00074


def _fail(label: str, observed: object, expected: object) -> str:
    return f"{label}: observed {observed!r}, expected {expected!r}"


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0)


def check_fixed_seed_parity() -> list[str]:
    return _check_fixed_seed_case(
        case_id=PARITY_CASE,
        seed=PARITY_SEED,
        controls_payload=PARITY_CONTROLS,
        expected=PARITY_EXPECTED,
    )


def check_umrigar_fixed_seed_parity() -> list[str]:
    return _check_fixed_seed_case(
        case_id=UMRIGAR_PARITY_CASE,
        seed=UMRIGAR_PARITY_SEED,
        controls_payload=UMRIGAR_PARITY_CONTROLS,
        expected=UMRIGAR_PARITY_EXPECTED,
    )


def _check_fixed_seed_case(
    *,
    case_id: str,
    seed: int,
    controls_payload: dict[str, float | int | str | None],
    expected: dict[str, float],
) -> list[str]:
    case = parse_case(case_id)
    controls = DMCRunControls(**controls_payload)  # type: ignore[arg-type]
    summary = run_streaming_seed(
        case,
        controls,
        seed=seed,
        guide_family=PARITY_GUIDE_FAMILY,
    )
    density = np.asarray(summary.density, dtype=float)
    grid = np.linspace(
        -controls.grid_extent,
        controls.grid_extent,
        density.size,
    )
    observed = {
        "mixed_energy": float(summary.mixed_energy),
        "r2_radius": float(summary.r2_radius),
        "density_sum": float(np.sum(density)),
        "density_max": float(np.max(density)),
        "density_first_moment": float(np.sum(density * grid) / np.sum(density)),
    }
    return [
        _fail(f"{case_id} {name}", observed[name], expected_value)
        for name, expected_value in expected.items()
        if not _close(observed[name], expected_value)
    ]


def check_hellmann_feynman(root: Path) -> list[str]:
    source = root / HF_SOURCE
    if not source.is_file():
        return [f"energy-response input is missing: {source}"]
    with source.open(newline="", encoding="utf-8") as handle:
        points = tuple(
            PairedEnergyResponsePoint(
                seed=int(row["seed"]),
                relative_lambda_offset=float(row["relative_lambda_offset"]),
                lambda_value=float(row["lambda_value"]),
                energy=float(row["energy"]),
            )
            for row in csv.DictReader(handle)
        )
    result = paired_trap_r2_from_energy_response(points, n_particles=HF_N_PARTICLES)
    failures: list[str] = []
    radius = round(float(result.rms_radius), 5)
    stderr = round(float(result.rms_radius_seed_stderr), 5)
    if radius != HF_EXPECTED_RADIUS:
        failures.append(_fail("hellmann_feynman rms_radius", radius, HF_EXPECTED_RADIUS))
    if stderr != HF_EXPECTED_STDERR:
        failures.append(_fail("hellmann_feynman stderr", stderr, HF_EXPECTED_STDERR))
    return failures


def check_backend_parity() -> list[str]:
    positions = np.array(
        [
            [-2.5, -0.8, 0.9, 2.7],
            [-3.1, -1.4, 0.2, 1.9],
        ],
        dtype=float,
    )
    rod_length = 1.0
    n_particles = positions.shape[1]
    offsets = rod_length * (np.arange(n_particles, dtype=float) - 0.5 * (n_particles - 1))
    kwargs = {
        "rod_length": rod_length,
        "alpha": 1.0,
        "relative_alpha": 1.6224444406063525,
        "center": 0.0,
        "omega2": 1.0,
        "pair_power": 1.0,
    }
    dispatched = reduced_tg_grad_lap_local_batch(positions, offsets, **kwargs)
    reference = _reduced_tg_grad_lap_local_batch_python(positions, offsets, **kwargs)
    labels = ("gradient", "laplacian", "local_energy", "finite_mask")
    failures: list[str] = []
    for label, got, want in zip(labels, dispatched, reference, strict=True):
        if label == "finite_mask":
            if not np.array_equal(got, want):
                failures.append(_fail(f"backend parity {label}", got.tolist(), want.tolist()))
            continue
        if not np.allclose(got, want, rtol=RELATIVE_TOLERANCE, atol=0.0):
            worst = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
            failures.append(f"backend parity {label}: max absolute difference {worst:.3e}")
    return failures


def check_population_guard_rejection() -> list[str]:
    common = {
        "dt": 0.01,
        "drift_limiter": "umrigar",
        "relative_alpha": 1.5,
        "production_tau": 480.0,
    }
    first = SimpleNamespace(case_id="N10_A1", dt=0.01, controls={**common, "walkers": 256})
    second = SimpleNamespace(
        case_id="N10_A1",
        dt=0.01,
        controls={**common, "walkers": 512, "production_tau": 481.0},
    )
    try:
        _require_common_identity(cast(Any, (first, second)))
    except ValueError as exc:
        expected = "may vary only walker count"
        return (
            [] if expected in str(exc) else [_fail("population guard message", str(exc), expected)]
        )
    return ["population ladder accepted a non-walker control mismatch"]


def check_forward_walking_guard_rejection() -> list[str]:
    try:
        _validate_fw_controls(
            0.001,
            0.03,
            0.95,
            anchor_treatment=(0.01, 512),
            candidate_treatment=(0.01, 512),
        )
    except ValueError as exc:
        expected = "candidate treatment equals the anchor"
        return (
            []
            if str(exc) == expected
            else [_fail("forward-walking guard message", str(exc), expected)]
        )
    return ["forward-walking sensitivity accepted the anchor treatment as its candidate"]


def report(title: str, failures: list[str]) -> None:
    if not failures:
        print(f"  ok    {title}")
        return
    print(f"  FAIL  {title}")
    for line in failures:
        print(f"          {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()

    backend = numba_backend_name()
    print(f"science: kernel backend is {backend}")
    if backend != "numba":
        print("science: the thesis numbers were produced under numba; install the dmc extra")
        return 1

    parity = check_fixed_seed_parity()
    umrigar_parity = check_umrigar_fixed_seed_parity()
    hf = check_hellmann_feynman(root)
    backends = check_backend_parity()
    population_guard = check_population_guard_rejection()
    forward_walking_guard = check_forward_walking_guard_rejection()

    report("fixed-seed DMC observables", parity)
    report("finite-A Umrigar fixed-seed DMC observables", umrigar_parity)
    report("published Hellmann-Feynman radius", hf)
    report("numba and python kernel agreement", backends)
    report("population ladder rejects non-walker control changes", population_guard)
    report("forward-walking candidate differs from its anchor", forward_walking_guard)

    total = (
        len(parity)
        + len(umrigar_parity)
        + len(hf)
        + len(backends)
        + len(population_guard)
        + len(forward_walking_guard)
    )
    if total:
        print(f"science: {total} failures")
        return 1
    print("science: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
