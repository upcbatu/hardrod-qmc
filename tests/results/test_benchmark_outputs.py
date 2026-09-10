from __future__ import annotations

import json
from pathlib import Path

from hrdmc.artifacts.manifest import verify_run_manifest
from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.plotting.figures.benchmark_packet import write_benchmark_packet_plots
from hrdmc.production.benchmark.run import run_benchmark_packet_workflow
from hrdmc.system.settings import DMCRunControls, parse_case


def test_unavailable_density_has_a_renderable_diagnostic(tmp_path: Path) -> None:
    paths = write_benchmark_packet_plots(
        tmp_path,
        {
            "case_id": "N2_A0",
            "status": "insufficient_information",
            "estimates": {"density": {"x": [], "value": [float("nan")]}},
        },
        formats=("png",),
    )
    assert len(paths) == 6
    assert all((tmp_path / path).is_file() for path in paths)


def test_optional_plot_failure_preserves_numeric_packet(tmp_path: Path, monkeypatch) -> None:
    def calculated(*args, **kwargs):
        return {
            "case_id": "N2_A0",
            "status": "insufficient_information",
            "seed_results": [],
            "pure_config": {},
            "estimates": {"energy": {"value": 2.0, "stderr": None}},
        }

    def unavailable(*args, **kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(
        "hrdmc.production.benchmark.run.summarize_benchmark_packet_case", calculated
    )
    monkeypatch.setattr("hrdmc.production.benchmark.run.write_benchmark_packet_plots", unavailable)
    result = run_benchmark_packet_workflow(
        parse_case("N2_A0"),
        DMCRunControls(0.01, 8, 0.1, 1, 1, 10, 20),
        [17],
        pure_config=PureWalkingConfig((0, 1)),
        output_dir=tmp_path,
        plot_formats=("png",),
    )
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["estimates"]["energy"]["value"] == 2.0
    assert payload["plot_status"] == "failed"
    assert result.status == "insufficient_information"
    valid, errors = verify_run_manifest(tmp_path / "run_manifest.json")
    assert valid, errors
