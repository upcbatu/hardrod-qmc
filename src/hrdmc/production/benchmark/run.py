from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts.layout import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.estimators.forward_walking.config import PureWalkingConfig
from hrdmc.plotting.figures.benchmark_packet import write_benchmark_packet_plots
from hrdmc.production.benchmark.case import summarize_benchmark_packet_case
from hrdmc.production.benchmark.outputs import (
    write_benchmark_packet_artifacts,
)
from hrdmc.sampling.dmc.run import dmc_progress_bar
from hrdmc.sampling.initial_conditions import InitializationControls
from hrdmc.system.guide_registry import (
    validate_production_reduced_tg_binding,
)
from hrdmc.system.settings import DMCRunControls, TrappedCase
from hrdmc.trial.guide import DEFAULT_GUIDE_FAMILY


@dataclass(frozen=True)
class BenchmarkPacketWorkflowResult:
    payload: dict[str, Any]
    status: str
    summary: dict[str, Any]
    artifacts: dict[str, str | None]


def run_benchmark_packet_workflow(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    pure_config: PureWalkingConfig,
    parallel_workers: int | None = None,
    progress: bool = False,
    output_dir: Path | None = None,
    write_artifacts: bool = True,
    write_plots: bool = True,
    plot_formats: tuple[str, ...] = ("png", "pdf"),
    command: list[str] | None = None,
    ess_warning_fraction: float = 0.20,
    ess_invalid_fraction: float = 0.10,
    log_weight_span_warning: float = 50.0,
    initialization: InitializationControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    guide_parameter_source: str = "explicit",
    allow_unvalidated_guide: bool = False,
    guide_provenance: dict[str, Any] | None = None,
) -> BenchmarkPacketWorkflowResult:
    """Run, assemble, and optionally persist one trapped-DMC benchmark packet."""
    started = time.perf_counter()
    validate_production_reduced_tg_binding(
        case=case,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
        source=guide_parameter_source,
        allow_unvalidated=allow_unvalidated_guide,
    )
    initialization = InitializationControls() if initialization is None else initialization
    resolved_output_dir = output_dir or artifact_dir(
        repo_root_from(Path(__file__)),
        ArtifactRoute("dmc", "local", "benchmark_packet"),
    )
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label="DMC benchmark",
        enabled=progress,
    ) as bar:
        payload = summarize_benchmark_packet_case(
            case,
            controls,
            seeds,
            pure_config=pure_config,
            parallel_workers=parallel_workers,
            progress=bar,
            trace_output_dir=resolved_output_dir if write_artifacts else None,
            ess_warning_fraction=ess_warning_fraction,
            ess_invalid_fraction=ess_invalid_fraction,
            log_weight_span_warning=log_weight_span_warning,
            initialization=initialization,
            guide_family=guide_family,
        )
    payload["guide_parameters"] = {
        "relative_alpha": controls.relative_alpha,
        "source": guide_parameter_source,
    }
    payload["guide_parameters"]["validation"] = (
        "not_independently_validated"
        if guide_parameter_source == "explicit" and controls.relative_alpha is not None
        else "exact_hard_point"
        if case.rod_length == 0.0
        else "validated"
        if guide_parameter_source != "explicit"
        else "not_independently_validated"
    )
    payload["guide_parameters"]["provenance"] = guide_provenance
    payload["execution_status"] = "completed"
    payload["calculation_wall_seconds"] = time.perf_counter() - started
    paths: dict[str, Path] = {}
    if write_artifacts:
        plot_paths: list[str] = []
        paths = write_benchmark_packet_artifacts(
            resolved_output_dir,
            payload=payload,
            case_id=case.case_id,
            seeds=seeds,
            controls=controls,
            parallel_workers=parallel_workers,
            initialization=initialization,
            guide_family=guide_family,
            guide_parameter_source=guide_parameter_source,
            plot_paths=plot_paths,
            plot_formats=plot_formats,
            command=command,
        )
        if write_plots:
            try:
                plot_paths = write_benchmark_packet_plots(
                    resolved_output_dir,
                    payload,
                    formats=plot_formats,
                )
                payload["plots"] = plot_paths
                payload["plot_status"] = "completed"
            except Exception as exc:
                # Numerical artifacts already exist; an optional renderer cannot erase them.
                payload["plot_status"] = "failed"
                payload["plot_error"] = f"{type(exc).__name__}: {exc}"
            payload["wall_seconds_before_final_write"] = time.perf_counter() - started
            paths = write_benchmark_packet_artifacts(
                resolved_output_dir,
                payload=payload,
                case_id=case.case_id,
                seeds=seeds,
                controls=controls,
                parallel_workers=parallel_workers,
                initialization=initialization,
                guide_family=guide_family,
                guide_parameter_source=guide_parameter_source,
                plot_paths=plot_paths,
                plot_formats=plot_formats,
                command=command,
            )
    energy = payload.get("estimates", {}).get("energy", {})
    return BenchmarkPacketWorkflowResult(
        payload=payload,
        status=str(payload.get("status", payload.get("classification", "completed"))),
        summary={
            "case": case.case_id,
            "seed_count": len(seeds),
            "energy": energy.get("value") if isinstance(energy, dict) else None,
            "energy_stderr": energy.get("stderr") if isinstance(energy, dict) else None,
            "energy_status": payload.get("energy_validation_status"),
            "forward_walking_status": payload.get("pure_fw_validation_status"),
            "guide_validation": payload["guide_parameters"]["validation"],
            "plot_status": payload.get("plot_status", "not_requested"),
            "calculation_wall_seconds": payload["calculation_wall_seconds"],
        },
        artifacts={
            "summary": str(paths["summary"]) if "summary" in paths else None,
            "run_manifest": (str(paths["run_manifest"]) if "run_manifest" in paths else None),
            "output_dir": str(resolved_output_dir) if write_artifacts else None,
        },
    )
