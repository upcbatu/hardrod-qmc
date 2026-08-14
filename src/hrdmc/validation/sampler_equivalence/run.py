from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from hrdmc.artifacts.progress import ProgressBar
from hrdmc.system.settings import TrappedCase
from hrdmc.validation.sampler_equivalence.assessment import assess_vmc_case
from hrdmc.validation.sampler_equivalence.binding import GuideParameterBinding
from hrdmc.validation.sampler_equivalence.models import (
    VMCSamplerChoice,
    VMCSamplingControls,
    VMCValidationPolicy,
)
from hrdmc.validation.sampler_equivalence.outputs import write_vmc_case_packet
from hrdmc.validation.sampler_equivalence.seed import VMCSeedRun, run_vmc_seed


@dataclass(frozen=True)
class VMCValidationCaseResult:
    status: str
    assessment: dict[str, Any]
    runs: tuple[VMCSeedRun, ...]
    artifacts: dict[str, str]
    actual_workers: int
def run_vmc_validation_case(
    case: TrappedCase,
    binding: GuideParameterBinding,
    *,
    controls: VMCSamplingControls,
    policy: VMCValidationPolicy,
    sampler_choices: dict[str, VMCSamplerChoice],
    rwm_seeds: tuple[int, ...],
    mala_seeds: tuple[int, ...],
    parallel_workers: int,
    output_dir,
    progress: ProgressBar | None = None,
) -> VMCValidationCaseResult:
    controls.validate()
    policy.validate()
    _validate_sampler_choices(sampler_choices)
    _validate_production_seeds(sampler_choices, rwm_seeds, mala_seeds)
    if parallel_workers <= 0 or parallel_workers > 5:
        raise ValueError("parallel_workers must lie in [1, 5]")
    tasks = [("random_walk_metropolis", seed, seed + 3_000_000) for seed in rwm_seeds] + [
        ("branching_free_mala", seed, seed + 4_000_000) for seed in mala_seeds
    ]
    runs, actual_workers = _run_seed_tasks(
        case,
        binding,
        controls=controls,
        sampler_choices=sampler_choices,
        tasks=tasks,
        parallel_workers=min(parallel_workers, len(tasks)),
        progress=progress,
    )
    assessment = assess_vmc_case(case, runs, policy=policy)
    artifact_paths = {}
    if output_dir is not None:
        artifact_paths = write_vmc_case_packet(
            output_dir,
            case=case,
            binding=binding,
            controls=controls,
            policy=policy,
            sampler_choices=sampler_choices,
            runs=runs,
            assessment=assessment.payload,
        )
    return VMCValidationCaseResult(
        status=assessment.status,
        assessment=assessment.payload,
        runs=tuple(runs),
        artifacts={name: str(path) for name, path in artifact_paths.items()},
        actual_workers=actual_workers,
    )
def _run_seed_tasks(
    case: TrappedCase,
    binding: GuideParameterBinding,
    *,
    controls: VMCSamplingControls,
    sampler_choices: dict[str, VMCSamplerChoice],
    tasks: list[tuple[str, int, int]],
    parallel_workers: int,
    progress: ProgressBar | None,
) -> tuple[list[VMCSeedRun], int]:
    if parallel_workers <= 1:
        return (
            [
                run_vmc_seed(
                    case,
                    binding,
                    sampler_choices[sampler],
                    controls,
                    seed,
                    initializer_seed=initializer_seed,
                    progress=progress,
                )
                for sampler, seed, initializer_seed in tasks
            ],
            1,
        )
    # Unlike DMC, progress is not required for correctness.  Avoid coupling the
    # production worker payload to a process manager; each completed seed adds
    # its declared production transitions to the parent bar.
    runs: list[VMCSeedRun] = []
    try:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    run_vmc_seed,
                    case,
                    binding,
                    sampler_choices[sampler],
                    controls,
                    seed,
                    initializer_seed=initializer_seed,
                )
                for sampler, seed, initializer_seed in tasks
            ]
            for future in as_completed(futures):
                runs.append(future.result())
                if progress is not None:
                    progress.update(controls.production_steps)
    except (OSError, PermissionError):
        return _run_seed_tasks(
            case,
            binding,
            controls=controls,
            sampler_choices=sampler_choices,
            tasks=tasks,
            parallel_workers=1,
            progress=progress,
        )
    return sorted(runs, key=lambda run: (run.sampler, run.seed)), parallel_workers
def _validate_sampler_choices(
    choices: dict[str, VMCSamplerChoice],
) -> None:
    expected = {"random_walk_metropolis", "branching_free_mala"}
    if set(choices) != expected:
        raise ValueError("production requires exactly RWM and branching-free MALA choices")
    for name, choice in choices.items():
        choice.validate()
        if choice.method != name:
            raise ValueError("sampler choice key and method disagree")
def _validate_production_seeds(
    choices: dict[str, VMCSamplerChoice],
    rwm_seeds: tuple[int, ...],
    mala_seeds: tuple[int, ...],
) -> None:
    if len(rwm_seeds) != 5 or len(mala_seeds) != 5:
        raise ValueError("thesis validation requires exactly five seeds per sampler")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in (*rwm_seeds, *mala_seeds)
    ):
        raise ValueError("production seeds must be non-negative integers")
    if len(set(rwm_seeds)) != 5 or len(set(mala_seeds)) != 5:
        raise ValueError("production sampler seeds must be unique")
    if not set(rwm_seeds).isdisjoint(mala_seeds):
        raise ValueError("RWM and MALA production seeds must be disjoint")
    rwm_initializer_seeds = tuple(seed + 3_000_000 for seed in rwm_seeds)
    mala_initializer_seeds = tuple(seed + 4_000_000 for seed in mala_seeds)
    production_streams = (
        rwm_seeds,
        rwm_initializer_seeds,
        mala_seeds,
        mala_initializer_seeds,
    )
    if not _all_unique(*production_streams):
        raise ValueError("all production transition and initializer RNG streams must be unique")
def _all_unique(*streams: tuple[int, ...]) -> bool:
    seeds = [seed for stream in streams for seed in stream]
    return len(seeds) == len(set(seeds))
