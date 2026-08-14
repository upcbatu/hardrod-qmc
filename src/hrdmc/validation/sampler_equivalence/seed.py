from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.artifacts.progress import ProgressBar
from hrdmc.estimators.variational import (
    VariationalObserver,
    VariationalStreamingAccumulator,
    VariationalStreamResult,
)
from hrdmc.sampling.vmc.engine import run_vmc_streaming
from hrdmc.sampling.vmc.results import (
    MultiplexedVMCObserver,
    VMCStreamingResult,
    VMCTransitionEvent,
)
from hrdmc.sampling.vmc.transitions import VMCConfig
from hrdmc.system.settings import TrappedCase, build_case_geometry
from hrdmc.trial.guide import build_guide
from hrdmc.validation.sampler_equivalence.binding import GuideParameterBinding
from hrdmc.validation.sampler_equivalence.initialization import prepare_vmc_initial_batch
from hrdmc.validation.sampler_equivalence.models import (
    VMCSamplerChoice,
    VMCSamplingControls,
    density_bin_edges,
    free_gap_bin_edges,
)


@dataclass(frozen=True)
class VMCSeedRun:
    case_id: str
    sampler: str
    seed: int
    initializer_seed: int
    initialization: dict[str, Any]
    engine: VMCStreamingResult
    estimates: VariationalStreamResult
    def to_dict(self) -> dict[str, Any]:
        """Return compact seed metadata; canonical series live in packet tables."""
        return {
            "case_id": self.case_id,
            "sampler": self.sampler,
            "seed": self.seed,
            "initializer_seed": self.initializer_seed,
            "initialization": self.initialization,
            "engine": {
                "burn_in_attempts": self.engine.burn_in_attempts.to_dict(),
                "production_attempts": self.engine.production_attempts.to_dict(),
                "production_sample_count": self.engine.production_sample_count,
                "seed": self.engine.seed,
                "wall_seconds": self.engine.wall_seconds,
                "metadata": self.engine.metadata,
            },
            "estimates": _compact_stream_payload(self.estimates),
        }
def run_vmc_seed(
    case: TrappedCase,
    binding: GuideParameterBinding,
    sampler: VMCSamplerChoice,
    controls: VMCSamplingControls,
    seed: int,
    *,
    initializer_seed: int,
    progress: ProgressBar | None = None,
) -> VMCSeedRun:
    """Run one independently initialized, seed-preserving VMC ensemble."""
    sampler.validate()
    controls.validate()
    if binding.case_id != case.case_id:
        raise ValueError("guide binding case does not match VMC seed case")
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=binding.guide_family,
        relative_alpha=binding.relative_alpha,
    )
    initial = prepare_vmc_initial_batch(
        case,
        walkers=controls.walkers,
        seed=initializer_seed,
    )
    density_edges = density_bin_edges(case, controls.density_bins)
    gap_edges = free_gap_bin_edges(case, controls.free_gap_bins)
    cutoff_epsilons = np.asarray(controls.cutoff_epsilons, dtype=float)
    accumulator = VariationalStreamingAccumulator(
        seed=seed,
        block_size=controls.block_steps,
        maximum_records=controls.maximum_records,
        density_bin_edges=density_edges,
        free_gap_bin_edges=gap_edges,
        cutoff_epsilons=cutoff_epsilons,
    )
    observer = VariationalObserver(
        center=system.center,
        rod_length=system.rod_length,
        accumulator=accumulator,
    )
    observers: tuple[Any, ...] = (observer,)
    if progress is not None:
        observers = (*observers, _ProgressObserver(progress))
    engine = run_vmc_streaming(
        initial_positions=initial.positions,
        guide=guide,
        config=VMCConfig(
            walkers=controls.walkers,
            burn_in_steps=controls.burn_in_steps,
            production_steps=controls.production_steps,
            method=sampler.engine_method,
            dt=(sampler.proposal_scale if sampler.engine_method == "mala" else 0.01),
            step_size=(sampler.proposal_scale if sampler.engine_method == "rwm" else 0.5),
            drift_limiter=sampler.drift_limiter,
        ),
        seed=seed,
        observer=MultiplexedVMCObserver(observers=observers),
    )
    estimates = observer.finish()
    if len(estimates.records) != controls.maximum_records:
        raise RuntimeError("VMC estimator stream did not preserve the declared block plan")
    return VMCSeedRun(
        case_id=case.case_id,
        sampler=sampler.method,
        seed=int(seed),
        initializer_seed=int(initializer_seed),
        initialization=initial.metadata,
        engine=engine,
        estimates=estimates,
    )
def _compact_stream_payload(result: VariationalStreamResult) -> dict[str, Any]:
    return {
        "estimator_type": "variational",
        "seed": result.seed,
        "block_size": result.block_size,
        "maximum_records": result.maximum_records,
        "production_transition_count": result.production_transition_count,
        "configuration_count": result.configuration_count,
        "particle_count": result.particle_count,
        "block_count": len(result.records),
        "density_accounting": _compact_histogram_accounting(result.density),
        "free_gap_accounting": _compact_histogram_accounting(result.free_gap_distribution),
    }
def _compact_histogram_accounting(histogram: Any) -> dict[str, Any]:
    return {
        "normalization_denominator": histogram.normalization_denominator,
        "in_grid_count": histogram.in_grid_count,
        "out_of_grid_count": histogram.out_of_grid_count,
        "in_grid_mass": histogram.in_grid_mass,
        "out_of_grid_mass": histogram.out_of_grid_mass,
        "expected_total_mass": histogram.expected_total_mass,
    }
class _ProgressObserver:
    def __init__(self, progress: ProgressBar) -> None:
        self._progress = progress
    def record_vmc_transition(self, event: VMCTransitionEvent) -> None:
        del event
        self._progress.update(1)
