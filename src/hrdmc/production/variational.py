"""General VMC runs using the existing sampler and variational observables."""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.artifacts.manifest import write_csv, write_json, write_run_manifest
from hrdmc.estimators.variational import VariationalObserver, VariationalStreamingAccumulator
from hrdmc.sampling.initial_conditions import (
    hard_core_preserving_breathing_scale,
    initial_walkers,
)
from hrdmc.sampling.vmc.engine import run_vmc_streaming
from hrdmc.sampling.vmc.transitions import VMCConfig
from hrdmc.statistics.timeseries import diagnose_chains
from hrdmc.system.guide_selection import GuideSelection
from hrdmc.system.settings import TrappedCase, build_case_geometry
from hrdmc.trial.guide import ReducedTGHardRodGuide


@dataclass(frozen=True)
class VariationalRunControls:
    sampler: VMCConfig
    block_size: int = 100
    grid_extent: float = 35.0
    bins: int = 840
    initial_scale: float = 1.0
    cutoff_epsilons: tuple[float, ...] = (0.01, 0.02, 0.04)
    rhat_limit: float = 1.01
    min_effective_samples: float = 400.0

    def validate(self) -> None:
        self.sampler.validate()
        if self.block_size < 1 or self.sampler.production_steps % self.block_size:
            raise ValueError("production-steps must be a positive multiple of block-size")
        if not math.isfinite(self.grid_extent) or self.grid_extent <= 0 or self.bins < 2:
            raise ValueError("grid-extent must be positive and bins at least 2")
        if not math.isfinite(self.initial_scale) or self.initial_scale <= 0:
            raise ValueError("initial-scale must be finite and positive")
        if not self.cutoff_epsilons or any(
            not math.isfinite(x) or x <= 0 for x in self.cutoff_epsilons
        ):
            raise ValueError("cutoff-epsilons must be finite and positive")
        if self.rhat_limit <= 1 or self.min_effective_samples <= 0:
            raise ValueError("rhat-limit must exceed 1 and min-effective-samples must be positive")


def run_variational_seed(
    case: TrappedCase,
    guide_selection: GuideSelection,
    controls: VariationalRunControls,
    seed: int,
) -> dict[str, Any]:
    controls.validate()
    system, trap = build_case_geometry(case)
    guide = ReducedTGHardRodGuide(
        system=system,
        trap=trap,
        alpha=case.omega,
        relative_alpha=guide_selection.relative_alpha,
    )
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    positions = initial_walkers(system, controls.sampler.walkers, rng)
    if controls.initial_scale != 1:
        positions = np.vstack(
            [
                hard_core_preserving_breathing_scale(
                    row, case.rod_length, controls.initial_scale, 0
                )
                for row in positions
            ]
        )
    accumulator = VariationalStreamingAccumulator(
        seed=seed,
        block_size=controls.block_size,
        maximum_records=controls.sampler.production_steps // controls.block_size,
        density_bin_edges=np.linspace(
            -controls.grid_extent, controls.grid_extent, controls.bins + 1
        ),
        free_gap_bin_edges=np.linspace(0, 2 * controls.grid_extent, controls.bins + 1),
        cutoff_epsilons=np.asarray(controls.cutoff_epsilons),
    )
    observer = VariationalObserver(center=0, rod_length=case.rod_length, accumulator=accumulator)
    engine = run_vmc_streaming(
        initial_positions=positions,
        guide=guide,
        config=controls.sampler,
        seed=seed,
        observer=observer,
    )
    stream = observer.finish()
    return {
        "seed": seed,
        "engine": asdict(engine),
        "blocks": [asdict(record) for record in stream.records],
        "density": {
            "bin_edges": stream.density.bin_edges,
            "value": stream.density.density,
            "in_grid_mass": stream.density.in_grid_mass,
            "out_of_grid_mass": stream.density.out_of_grid_mass,
        },
        "free_gaps": {
            "bin_edges": stream.free_gap_distribution.bin_edges,
            "value": stream.free_gap_distribution.density,
            "out_of_grid_mass": stream.free_gap_distribution.out_of_grid_mass,
        },
    }


def summarize_variational_runs(
    rows: list[dict[str, Any]],
    controls: VariationalRunControls,
) -> dict[str, Any]:
    estimates, diagnostics = {}, {}
    for name in ("e_local", "t_local", "trap", "r2", "weighted_free_gap"):
        chains = [np.asarray([block["means"][name] for block in row["blocks"]]) for row in rows]
        means = np.asarray([float(np.mean(chain)) for chain in chains])
        estimates[name] = {
            "value": float(np.mean(means)),
            "stderr": float(np.std(means, ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else None,
            "seed_means": means.tolist(),
        }
        diagnostics[name] = (
            diagnose_chains(
                [np.arange(len(chain), dtype=float) for chain in chains],
                chains,
                rhat_threshold=controls.rhat_limit,
                min_effective_samples=controls.min_effective_samples,
            ).to_dict()
            if len(rows) >= 2 and min(map(len, chains)) >= 8
            else {"classification": "insufficient_information"}
        )
    return {
        "estimates": estimates,
        "diagnostics": diagnostics,
        "rms_radius": math.sqrt(estimates["r2"]["value"]),
        "uncertainty_method": "standard error of independent seed means",
        "guide_validation": "not_performed_by_this_run",
    }


def run_variational_workflow(
    case: TrappedCase,
    guide: GuideSelection,
    controls: VariationalRunControls,
    seeds: list[int],
    *,
    workers: int,
    output: Path,
    command: list[str],
) -> dict[str, Any]:
    controls.validate()
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    arguments = [(case, guide, controls, seed) for seed in seeds]
    rows = []
    if workers == 1:
        for arguments_for_seed in arguments:
            row = run_variational_seed(*arguments_for_seed)
            write_json(output / f"seed_{row['seed']}.json", row)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(seeds))) as pool:
            futures = [pool.submit(run_variational_seed, *args) for args in arguments]
            for future in futures:
                row = future.result()
                write_json(output / f"seed_{row['seed']}.json", row)
                rows.append(row)
    summary = summarize_variational_runs(rows, controls)
    summary.update(
        case_id=case.case_id,
        seeds=seeds,
        guide=asdict(guide),
        controls=asdict(controls),
        execution_status="completed",
        status="completed_with_diagnostics",
        wall_seconds=time.perf_counter() - started,
        out_of_grid_mass_by_seed=[row["density"]["out_of_grid_mass"] for row in rows],
    )
    write_json(output / "summary.json", summary)
    _write_variational_tables(output, rows)
    write_run_manifest(
        output,
        run_name="vmc_run",
        config={
            "case": case.case_id,
            "guide": asdict(guide),
            "controls": asdict(controls),
            "seeds": seeds,
            "workers": workers,
            "command": command,
        },
        artifacts=sorted(output.glob("*.json")) + sorted(output.glob("*.csv")),
        status=summary["status"],
    )
    return summary


def _write_variational_tables(output: Path, rows: list[dict[str, Any]]) -> None:
    edges = np.asarray(rows[0]["density"]["bin_edges"])
    profiles = np.asarray([row["density"]["value"] for row in rows])
    means = np.mean(profiles, axis=0)
    errors = np.std(profiles, axis=0, ddof=1) / np.sqrt(len(rows)) if len(rows) > 1 else None
    write_csv(
        output / "density.csv",
        [
            {
                "q_left": left,
                "q_right": right,
                "q": (left + right) / 2,
                "density": means[i],
                "stderr": None if errors is None else errors[i],
                **{f"seed_{row['seed']}": profiles[j, i] for j, row in enumerate(rows)},
            }
            for i, (left, right) in enumerate(pairwise(edges))
        ],
    )
    write_csv(
        output / "blocks.csv",
        [
            {
                "seed": row["seed"],
                "first_step": block["first_step"],
                "last_step": block["last_step"],
                **{
                    key: value
                    for key, value in block["means"].items()
                    if key != "truncated_gradient"
                },
            }
            for row in rows
            for block in row["blocks"]
        ],
    )
