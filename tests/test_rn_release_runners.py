from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}

sys.path.insert(0, str(ROOT / "src"))
_RN_WORKFLOW = importlib.import_module("hrdmc.workflows.dmc.rn_block")
default_parallel_workers = _RN_WORKFLOW.default_parallel_workers
resolve_parallel_workers = _RN_WORKFLOW.resolve_parallel_workers


def _run_script(script: str, *args: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), *args, "--no-write"],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=ENV,
        text=True,
    )
    return json.loads(completed.stdout)


def test_rn_seed_worker_policy_caps_and_prefers_balanced_batches() -> None:
    assert default_parallel_workers(3) == 3
    assert default_parallel_workers(6) == 6
    assert default_parallel_workers(10) == 5
    assert default_parallel_workers(12) == 6
    assert default_parallel_workers(40) == 5
    assert default_parallel_workers(48) == 6
    assert default_parallel_workers(7) == 6
    assert resolve_parallel_workers(3, 8) == 3
    assert resolve_parallel_workers(10, 8) == 8
    assert resolve_parallel_workers(10, 12) == 10
    assert resolve_parallel_workers(50, 12) == 12


def test_rn_exact_tg_runner_checks_known_harmonic_energy() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/exact_tg_trap.py",
        "--n-particles",
        "3",
        "--omega",
        "0.1",
        "--seeds",
        "301",
        "--walkers",
        "8",
        "--burn-tau",
        "0.004",
        "--production-tau",
        "0.006",
        "--store-every",
        "1",
        "--energy-tolerance",
        "1e-10",
    )

    assert payload["status"] == "passed"
    assert payload["exact_solution"]["formula"] == "E0 = N^2 * omega / sqrt(2)"
    assert payload["absolute_energy_error"] <= 1e-10


def test_homogeneous_exact_grid_runner_includes_finite_a_ring_anchor() -> None:
    payload = _run_script(
        "experiments/validation/homogeneous_ring_exact_grid.py",
        "--n-values",
        "4,8",
        "--eta-values",
        "0.1,0.5",
        "--samples-per-case",
        "2",
        "--tolerance",
        "1e-7",
    )

    assert payload["status"] == "passed"
    assert payload["formula"] == "E_N/N = pi^2 * (N^2 - 1) / (3 * (L - N*a)^2)"
    assert payload["case_count"] == 4
    assert payload["max_energy_per_particle_abs_error"] <= 1e-7


def test_rn_trapped_stationarity_runner_emits_release_gates() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/trapped_stationarity_grid.py",
        "--cases",
        "N4_a0.5_omega0.1",
        "--seeds",
        "301",
        "--walkers",
        "8",
        "--burn-tau",
        "0.004",
        "--production-tau",
        "0.006",
        "--store-every",
        "1",
        "--grid-extent",
        "30",
        "--n-bins",
        "80",
    )

    assert payload["status"] == "completed"
    assert payload["case_count"] == 1
    row = payload["cases"][0]
    assert row["case_id"] == "N4_a0.5_omega0.1"
    assert "rhat_energy" in row
    assert "neff_energy" in row
    assert "density_accounting_clean" in row
    assert "mixed_energy_conservative_stderr" in row
    assert "uncertainty_status" in row
    assert "max_spread_blocking_z" in row
    assert "old_case_gate" in row
    assert "hygiene_gate" in row
    assert "final_classification" in row
    assert "blocking_plateau_energy" in row
    assert "blocked_zscore_max_energy" in row
    assert "robust_zscore_max_energy" in row
    assert "ess_fraction_min" in row
    assert "log_weight_span_max" in row
    energy_chain = row["diagnostics"]["energy"]["chain_diagnostics"][0]
    assert "first_last_blocking_z" in energy_chain
    assert "spread_veto" in energy_chain
    assert "trend_clean" in energy_chain
    assert "block_means" in energy_chain
