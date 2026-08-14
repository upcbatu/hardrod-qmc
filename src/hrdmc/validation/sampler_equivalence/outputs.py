from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.manifest import ensure_dir, write_csv, write_json
from hrdmc.system.settings import TrappedCase
from hrdmc.validation.sampler_equivalence.binding import GuideParameterBinding
from hrdmc.validation.sampler_equivalence.models import (
    VMCSamplerChoice,
    VMCSamplingControls,
    VMCValidationPolicy,
)
from hrdmc.validation.sampler_equivalence.seed import VMCSeedRun


def write_vmc_case_packet(
    output_dir: Path,
    *,
    case: TrappedCase,
    binding: GuideParameterBinding,
    controls: VMCSamplingControls,
    policy: VMCValidationPolicy,
    sampler_choices: dict[str, VMCSamplerChoice],
    runs: list[VMCSeedRun],
    assessment: dict[str, Any],
) -> dict[str, Path]:
    directory = _fresh_packet_directory(output_dir)
    density_edges = _common_profile_edges(runs, "density")
    gap_edges = _common_profile_edges(runs, "free_gap_distribution")
    summary = {
        "status": assessment["status"],
        "case_id": case.case_id,
        "units": case.unit_metadata(),
        "guide_binding": binding.to_dict(),
        "controls": controls.to_dict(),
        "policy": policy.to_dict(),
        "sampler_choices": {name: choice.to_dict() for name, choice in sampler_choices.items()},
        "seed_design": {
            "seed_count_per_sampler": {
                sampler: sum(run.sampler == sampler for run in runs) for sampler in sampler_choices
            },
            "sampler_seed_streams_disjoint": _sampler_seeds_disjoint(runs),
            "independent_initial_arrays": True,
            "walkers_not_counted_as_independent_chains": True,
        },
        "assessment": _compact_assessment(assessment),
        "seed_results": [run.to_dict() for run in runs],
        "grids": {
            "density": {"bin_edges": density_edges.tolist()},
            "free_gap_distribution": {"bin_edges": gap_edges.tolist()},
        },
        "canonical_tables": {
            "seed_summary": "seed_table.csv",
            "estimator_block_trace": "block_trace.csv",
            "density_profiles": "density_by_seed.csv",
            "free_gap_profiles": "free_gap_distribution_by_seed.csv",
        },
    }
    seed_table = _write_seed_table(directory / "seed_table.csv", runs)
    trace_table = _write_trace_table(directory / "block_trace.csv", runs)
    density_table = _write_profile_table(
        directory / "density_by_seed.csv",
        runs,
        "density",
    )
    gap_table = _write_profile_table(
        directory / "free_gap_distribution_by_seed.csv",
        runs,
        "free_gap_distribution",
    )
    summary_path = directory / "summary.json"
    write_json(summary_path, summary)
    return {
        "summary": summary_path,
        "seed_table": seed_table,
        "block_trace": trace_table,
        "density_by_seed": density_table,
        "free_gap_distribution_by_seed": gap_table,
    }
def _fresh_packet_directory(path: Path) -> Path:
    directory = Path(path)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "VMC validation packets are immutable; choose a new empty output directory"
        )
    return ensure_dir(directory)
def _compact_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(assessment)
    for sampler in payload.get("samplers", {}).values():
        if not isinstance(sampler, dict):
            continue
        sampler.pop("density_by_seed", None)
        sampler.pop("free_gap_distribution_by_seed", None)
    equivalence = payload.get("sampler_equivalence")
    if isinstance(equivalence, dict):
        for row in equivalence.get("profiles", {}).values():
            if isinstance(row, dict):
                row.pop("mean_difference", None)
    return payload
def _common_profile_edges(runs: list[VMCSeedRun], name: str) -> np.ndarray:
    if not runs:
        raise ValueError("VMC packet requires at least one seed run")
    first = np.asarray(getattr(runs[0].estimates, name).bin_edges, dtype=float)
    for run in runs[1:]:
        current = np.asarray(getattr(run.estimates, name).bin_edges, dtype=float)
        if not np.array_equal(current, first):
            raise ValueError(f"VMC {name} grids differ across seed runs")
    return first
def _write_seed_table(path: Path, runs: list[VMCSeedRun]) -> Path:
    fields = [
        "case_id",
        "sampler",
        "seed",
        "initializer_seed",
        "acceptance_fraction",
        "invalid_proposal_fraction",
        "nonfinite_proposal_fraction",
        "metropolis_rejection_fraction",
        "production_sample_count",
        "block_count",
        "wall_seconds",
        "density_out_of_grid_mass",
        "free_gap_out_of_grid_mass",
    ]
    rows = []
    for run in runs:
        attempts = run.engine.production_attempts
        denominator = attempts.attempted
        rows.append(
            {
                "case_id": run.case_id,
                "sampler": run.sampler,
                "seed": run.seed,
                "initializer_seed": run.initializer_seed,
                "acceptance_fraction": attempts.accepted / denominator,
                "invalid_proposal_fraction": attempts.invalid_proposals / denominator,
                "nonfinite_proposal_fraction": attempts.nonfinite_proposals / denominator,
                "metropolis_rejection_fraction": attempts.metropolis_rejections / denominator,
                "production_sample_count": run.engine.production_sample_count,
                "block_count": len(run.estimates.records),
                "wall_seconds": run.engine.wall_seconds,
                "density_out_of_grid_mass": run.estimates.density.out_of_grid_mass,
                "free_gap_out_of_grid_mass": run.estimates.free_gap_distribution.out_of_grid_mass,
            }
        )
    return write_csv(path, rows, fieldnames=fields)
def _write_trace_table(path: Path, runs: list[VMCSeedRun]) -> Path:
    cutoff_labels = [
        f"t_grad_cutoff_{value.epsilon:g}"
        for value in runs[0].estimates.records[0].means.truncated_gradient
    ]
    excluded_labels = [
        f"excluded_probability_{value.epsilon:g}"
        for value in runs[0].estimates.records[0].means.truncated_gradient
    ]
    fields = [
        "case_id",
        "sampler",
        "seed",
        "block_index",
        "first_step",
        "last_step",
        "batch_count",
        "t_local",
        "trap",
        "e_local",
        "r2",
        "weighted_free_gap",
        *cutoff_labels,
        *excluded_labels,
    ]
    rows = []
    for run in runs:
        for block_index, record in enumerate(run.estimates.records):
            row: dict[str, Any] = {
                "case_id": run.case_id,
                "sampler": run.sampler,
                "seed": run.seed,
                "block_index": block_index,
                "first_step": record.first_step,
                "last_step": record.last_step,
                "batch_count": record.batch_count,
                "t_local": record.means.t_local,
                "trap": record.means.trap,
                "e_local": record.means.e_local,
                "r2": record.means.r2,
                "weighted_free_gap": record.means.weighted_free_gap,
            }
            for value in record.means.truncated_gradient:
                row[f"t_grad_cutoff_{value.epsilon:g}"] = value.unconditional_t_grad
                row[f"excluded_probability_{value.epsilon:g}"] = value.excluded_probability
            rows.append(row)
    return write_csv(path, rows, fieldnames=fields)
def _write_profile_table(
    path: Path,
    runs: list[VMCSeedRun],
    name: str,
) -> Path:
    first = getattr(runs[0].estimates, name)
    edges = np.asarray(first.bin_edges, dtype=float)
    fields = ["bin_left", "bin_right", "bin_center", "bin_width"] + [
        f"{run.sampler}_seed{run.seed}" for run in runs
    ]
    densities = {
        (run.sampler, run.seed): np.asarray(getattr(run.estimates, name).density, dtype=float)
        for run in runs
    }
    rows = []
    for index in range(edges.size - 1):
        row: dict[str, Any] = {
            "bin_left": edges[index],
            "bin_right": edges[index + 1],
            "bin_center": 0.5 * (edges[index] + edges[index + 1]),
            "bin_width": edges[index + 1] - edges[index],
        }
        for run in runs:
            row[f"{run.sampler}_seed{run.seed}"] = densities[(run.sampler, run.seed)][index]
        rows.append(row)
    return write_csv(path, rows, fieldnames=fields)
def _sampler_seeds_disjoint(runs: list[VMCSeedRun]) -> bool:
    rwm = {run.seed for run in runs if run.sampler == "random_walk_metropolis"}
    mala = {run.seed for run in runs if run.sampler == "branching_free_mala"}
    return rwm.isdisjoint(mala)
