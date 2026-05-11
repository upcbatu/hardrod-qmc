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
_IO_ARTIFACTS = importlib.import_module("hrdmc.io.artifacts")
verify_run_manifest = _IO_ARTIFACTS.verify_run_manifest


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


def _run_script_with_output(script: str, output_dir: Path, *args: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            *args,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env=ENV,
        text=True,
    )
    return json.loads(completed.stdout)


def test_rn_validate_runner_checks_streaming_raw_equivalence() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/validate_streaming.py",
        "--walkers",
        "8",
        "--burn-tau",
        "0.004",
        "--production-tau",
        "0.006",
        "--store-every",
        "1",
    )

    assert payload["status"] == "completed"
    assert payload["validation"]["streaming_matches_raw"]


def test_rn_validate_runner_accepts_progress_flag() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/validate_streaming.py",
        "--walkers",
        "4",
        "--burn-tau",
        "0.004",
        "--production-tau",
        "0.004",
        "--store-every",
        "1",
        "--progress",
    )

    assert payload["status"] == "completed"


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


def test_rn_single_case_runner_emits_lda_comparison_summary() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/single_case.py",
        "--case",
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
    assert payload["case"]["case_id"] == "N4_a0.5_omega0.1"
    assert payload["case"]["density_integral"] > 0.0
    assert "energy_dmc_minus_lda" in payload["case"]


def test_rn_single_case_runner_expands_grid_for_weak_trap_support() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/single_case.py",
        "--case",
        "N8_a0.5_omega0.05",
        "--seeds",
        "301",
        "--walkers",
        "32",
        "--burn-tau",
        "0.02",
        "--production-tau",
        "0.01",
        "--store-every",
        "2",
        "--grid-extent",
        "20",
        "--n-bins",
        "80",
    )

    assert payload["status"] == "completed"
    assert payload["case"]["effective_grid_extent"] > payload["case"]["controls"]["grid_extent"]
    assert payload["case"]["density_integral"] > 0.0


def test_rn_single_case_runner_records_auto_parallel_plan() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/single_case.py",
        "--case",
        "N4_a0.5_omega0.1",
        "--seeds",
        "301,302,303",
        "--walkers",
        "8",
        "--burn-tau",
        "0.004",
        "--production-tau",
        "0.004",
        "--store-every",
        "1",
        "--parallel-workers",
        "0",
    )

    assert payload["status"] == "completed"
    assert payload["case"]["parallel_workers_requested"] == 3
    assert 1 <= payload["case"]["parallel_workers"] <= 3


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


def test_rn_grid_runner_handles_multiple_cases() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/grid.py",
        "--cases",
        "N4_a0.5_omega0.1,N4_a0.5_omega0.2",
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
    assert payload["case_count"] == 2
    assert [row["case_id"] for row in payload["cases"]] == [
        "N4_a0.5_omega0.1",
        "N4_a0.5_omega0.2",
    ]


def test_rn_grid_runner_writes_manifest_and_resume_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "rn_grid"
    payload = _run_script_with_output(
        "experiments/dmc/rn_block/grid.py",
        output_dir,
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

    assert payload["schema_version"] == "rn_block_grid_v1"
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "case_table.csv").is_file()
    assert (output_dir / "checkpoint.json").is_file()
    ok, errors = verify_run_manifest(output_dir / "run_manifest.json")
    assert ok, errors

    resumed = _run_script_with_output(
        "experiments/dmc/rn_block/grid.py",
        output_dir,
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
        "--resume",
    )

    assert resumed["case_count"] == 1


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


def test_rn_gate_audit_sweep_runner_emits_control_rows() -> None:
    payload = _run_script(
        "experiments/dmc/rn_block/gate_audit_sweep.py",
        "--cases",
        "N4_a0.5_omega0.1",
        "--seeds",
        "301",
        "--base-dt",
        "0.002",
        "--base-walkers",
        "4",
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
    assert payload["row_count"] == 6
    assert "fits" in payload
    assert "final_classification" in payload["rows"][0]
