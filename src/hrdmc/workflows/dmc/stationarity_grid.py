from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hrdmc.artifacts import ArtifactRoute, artifact_dir, repo_root_from
from hrdmc.workflows.dmc.collective_rn import CollectiveRNControls
from hrdmc.workflows.dmc.guide_validation import (
    validate_production_contact_guide_binding,
)
from hrdmc.workflows.dmc.initial_conditions import InitializationControls
from hrdmc.workflows.dmc.stationarity import classify_grid, summarize_stationarity_case
from hrdmc.workflows.dmc.stationarity_outputs import write_stationarity_grid_artifacts
from hrdmc.workflows.dmc.trapped import (
    DEFAULT_GUIDE_FAMILY,
    DMCRunControls,
    TrappedCase,
    controls_to_dict,
    dmc_progress_bar,
)

STATIONARITY_GRID_SCHEMA_VERSION = "dmc_trapped_stationarity_grid_v2"


@dataclass(frozen=True)
class StationarityGridWorkflowResult:
    payload: dict[str, Any]
    status: str
    summary: dict[str, Any]
    artifacts: dict[str, str | None]


def run_stationarity_grid_workflow(
    cases: list[TrappedCase],
    controls: DMCRunControls,
    seeds: list[int],
    *,
    parallel_workers: int | None = None,
    progress: bool = False,
    output_dir: Path | None = None,
    write_artifacts: bool = True,
    write_plots: bool = True,
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
) -> StationarityGridWorkflowResult:
    """Run, assemble, and optionally persist a trapped-DMC stationarity grid."""

    if not cases:
        raise ValueError("stationarity grid requires at least one case")
    if guide_family == "contact-corrected-reduced-tg" and len(cases) != 1:
        raise ValueError("one validated contact-guide artifact binds exactly one case")
    validate_production_contact_guide_binding(
        case=cases[0],
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
        ArtifactRoute("dmc", "local", "trapped_stationarity_grid"),
    )
    with dmc_progress_bar(
        controls=controls,
        seed_count=len(seeds) * len(cases),
        label=(
            "DMC stationarity with collective RN"
            if collective_rn is not None
            else "DMC stationarity"
        ),
        enabled=progress,
    ) as bar:
        rows = [
            summarize_stationarity_case(
                case,
                controls,
                seeds,
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
            for case in cases
        ]

    guide_parameters = {
        "relative_alpha": controls.relative_alpha,
        "contact_beta": controls.contact_beta,
        "source": guide_parameter_source,
        "source_sha256": guide_parameter_source_sha256,
        "source_manifest_sha256": guide_parameter_source_manifest_sha256,
        "source_identity_fingerprint": guide_parameter_source_identity_fingerprint,
    }
    classification = classify_grid(rows)
    payload = {
        "schema_version": STATIONARITY_GRID_SCHEMA_VERSION,
        "status": classification,
        "classification": classification,
        "diagnostic": "finite-A trapped DMC stationarity",
        "controls": controls_to_dict(controls),
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "collective_rn": None if collective_rn is None else collective_rn.to_metadata(),
        "guide_family": guide_family,
        "guide_parameters": guide_parameters,
        "case_count": len(rows),
        "cases": rows,
    }
    config = {
        "cases": [case.case_id for case in cases],
        "seeds": seeds,
        "controls": controls_to_dict(controls),
        "parallel_workers": parallel_workers,
        "initialization_mode": initialization.mode,
        "init_width_log_sigma": initialization.init_width_log_sigma,
        "breathing_preburn_steps": initialization.breathing_preburn_steps,
        "breathing_preburn_log_step": initialization.breathing_preburn_log_step,
        "collective_rn": None if collective_rn is None else collective_rn.to_metadata(),
        "guide_family": guide_family,
        "guide_parameters": guide_parameters,
    }
    paths: dict[str, Path] = {}
    if write_artifacts:
        paths = write_stationarity_grid_artifacts(
            resolved_output_dir,
            payload=payload,
            rows=rows,
            config=config,
            plots=write_plots,
            command=command,
        )

    return StationarityGridWorkflowResult(
        payload=payload,
        status=str(payload["status"]),
        summary={
            "case_count": len(rows),
            "seed_count": len(seeds),
            "classification": payload["classification"],
        },
        artifacts={
            "summary": str(paths["summary"]) if "summary" in paths else None,
            "case_table": str(paths["case_table"]) if "case_table" in paths else None,
            "run_manifest": (str(paths["run_manifest"]) if "run_manifest" in paths else None),
        },
    )
