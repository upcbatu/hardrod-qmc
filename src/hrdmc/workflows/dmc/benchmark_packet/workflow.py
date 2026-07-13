from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.estimators.pure.forward_walking import PureWalkingConfig
from hrdmc.plotting import write_benchmark_packet_plots
from hrdmc.workflows.dmc.benchmark_packet.case import summarize_benchmark_packet_case
from hrdmc.workflows.dmc.benchmark_packet.outputs import (
    write_benchmark_packet_artifacts,
)
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.guide_validation import (
    validate_production_contact_guide_binding,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    dmc_progress_bar,
)


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
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    guide_parameter_source: str = "explicit",
    guide_parameter_source_sha256: str | None = None,
    guide_parameter_source_manifest_sha256: str | None = None,
    guide_parameter_source_identity_fingerprint: str | None = None,
) -> BenchmarkPacketWorkflowResult:
    """Run, assemble, and optionally persist one trapped-DMC benchmark packet."""

    validate_production_contact_guide_binding(
        case=case,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
        source=guide_parameter_source,
        source_sha256=guide_parameter_source_sha256,
        source_manifest_sha256=guide_parameter_source_manifest_sha256,
        source_identity_fingerprint=guide_parameter_source_identity_fingerprint,
    )
    initialization = InitializationControls() if initialization is None else initialization
    resolved_output_dir = output_dir or artifact_dir(
        repo_root_from(Path(__file__)),
        ArtifactRoute("dmc", "local", "benchmark_packet"),
    )
    label = "DMC benchmark with collective RN" if collective_rn is not None else "DMC benchmark"
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds),
        label=label,
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
            collective_rn=collective_rn,
            guide_family=guide_family,
        )

    payload["guide_parameters"] = {
        "relative_alpha": controls.relative_alpha,
        "contact_beta": controls.contact_beta,
        "source": guide_parameter_source,
        "source_sha256": guide_parameter_source_sha256,
        "source_manifest_sha256": guide_parameter_source_manifest_sha256,
        "source_identity_fingerprint": guide_parameter_source_identity_fingerprint,
    }

    paths: dict[str, Path] = {}
    if write_artifacts:
        plot_paths: list[str] = []
        if write_plots:
            plot_paths = write_benchmark_packet_plots(
                resolved_output_dir,
                payload,
                formats=plot_formats,
            )
            payload["plots"] = plot_paths
        paths = write_benchmark_packet_artifacts(
            resolved_output_dir,
            payload=payload,
            case_id=case.case_id,
            seeds=seeds,
            controls=controls,
            parallel_workers=parallel_workers,
            initialization=initialization,
            collective_rn=collective_rn,
            guide_family=guide_family,
            guide_parameter_source=guide_parameter_source,
            guide_parameter_source_sha256=guide_parameter_source_sha256,
            guide_parameter_source_manifest_sha256=guide_parameter_source_manifest_sha256,
            guide_parameter_source_identity_fingerprint=(
                guide_parameter_source_identity_fingerprint
            ),
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
            "collective_rn": collective_rn is not None,
            "energy": energy.get("value") if isinstance(energy, dict) else None,
            "energy_stderr": energy.get("stderr") if isinstance(energy, dict) else None,
        },
        artifacts={
            "summary": str(paths["summary"]) if "summary" in paths else None,
            "run_manifest": (str(paths["run_manifest"]) if "run_manifest" in paths else None),
            "output_dir": str(resolved_output_dir) if write_artifacts else None,
        },
    )
