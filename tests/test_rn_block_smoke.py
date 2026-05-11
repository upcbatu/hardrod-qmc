from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rn_block_smoke_script_runs_without_writing() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "dmc" / "rn_block" / "smoke.py"),
            "--walkers",
            "8",
            "--burn-in",
            "4",
            "--production",
            "6",
            "--store-every",
            "2",
            "--no-write",
        ],
        check=True,
        capture_output=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "completed"
    assert payload["benchmark_tier"] == "RN-block DMC smoke"
    assert payload["summary_mode"] == "streaming"
    assert payload["valid_snapshot_fraction"] == 1.0
    assert payload["n_samples"] == 24
    assert payload["density_integral"] == 4.0
    assert payload["rn_block"]["rn_event_count"] > 0
