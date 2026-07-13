from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hrdmc.analysis import relative_density_l2_error
from hrdmc.artifacts import (
    build_run_provenance,
    config_fingerprint,
    ensure_dir,
    write_json,
    write_run_manifest,
)
from hrdmc.estimators.mixed import estimate_weighted_observables
from hrdmc.io.checkpoint import read_checkpoint, write_checkpoint
from hrdmc.io.progress import (
    ProgressBar,
    QueuedProgress,
    progress_bar,
)
from hrdmc.monte_carlo.dmc.local import (
    DMCConfig,
    DMCStreamingSummary,
    DMCTransportObserver,
    run_dmc,
    run_dmc_streaming,
)
from hrdmc.runners import run_seed_batch
from hrdmc.systems import (
    BASE_TRAP_QUADRATIC_COUPLING,
    HarmonicTrap,
    OpenLineHardRodSystem,
)
from hrdmc.theory import lda_density_profile, lda_rms_radius, lda_total_energy
from hrdmc.theory.units import HO_TRAP_OMEGA, harmonic_oscillator_unit_metadata
from hrdmc.wavefunctions.guides import (
    ContactCorrectedReducedTGHardRodGuide,
    FixedGuideQuadraticResponse,
    GapHCorrectedHardRodGuide,
    ReducedTGHardRodGuide,
)
from hrdmc.workflows.dmc.collective_rn import (
    CollectiveRNControls,
    build_collective_rn_extension,
)
from hrdmc.workflows.dmc.initial_conditions import (
    InitializationControls,
    initial_walkers,
    prepare_initial_walkers,
)

HO_CASE_RE = re.compile(r"^N(?P<n>\d+)_A(?P<A>[0-9.]+)$")
MAX_AUTO_WORKERS = 6
TRAPPED_GRID_SCHEMA_VERSION = "dmc_grid_v1"
TRAPPED_SINGLE_CASE_SCHEMA_VERSION = "dmc_single_case_v1"
GUIDE_FAMILIES = (
    "reduced-tg",
    "contact-corrected-reduced-tg",
    "gap-h-corrected",
)
DEFAULT_GUIDE_FAMILY = "reduced-tg"


@dataclass(frozen=True)
class TrappedCase:
    n_particles: int
    rod_length: float

    def __post_init__(self) -> None:
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.rod_length < 0.0:
            raise ValueError("rod_length must be non-negative")

    @property
    def case_id(self) -> str:
        return f"N{self.n_particles}_A{self.rod_length:g}"

    @property
    def omega(self) -> float:
        """Trap frequency after oscillator-unit nondimensionalization."""

        return HO_TRAP_OMEGA

    @property
    def rod_length_ho(self) -> float:
        return self.rod_length

    def unit_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            **harmonic_oscillator_unit_metadata(),
            "case_parameterization": "harmonic_oscillator_units",
            "rod_length_ho": self.rod_length,
        }
        return metadata


@dataclass(frozen=True)
class DMCRunControls:
    dt: float
    walkers: int
    burn_tau: float
    production_tau: float
    store_every: int
    grid_extent: float
    n_bins: int
    ess_resample_fraction: float = 0.35
    local_step_method: str = "metropolis"
    relative_alpha: float | None = None
    contact_beta: float | None = None
    response_lambda: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.walkers <= 0:
            raise ValueError("walkers must be positive")
        if self.burn_tau < 0.0 or self.production_tau <= 0.0:
            raise ValueError("burn_tau must be non-negative and production_tau positive")
        if self.store_every <= 0:
            raise ValueError("store_every must be positive")
        if not np.isfinite(self.grid_extent) or self.grid_extent <= 0.0:
            raise ValueError("grid_extent must be finite and positive")
        if self.n_bins < 2:
            raise ValueError("n_bins must be at least two")
        if not 0.0 <= self.ess_resample_fraction <= 1.0:
            raise ValueError("ess_resample_fraction must lie in [0, 1]")
        if self.local_step_method not in {"euler", "metropolis"}:
            raise ValueError("local_step_method must be 'euler' or 'metropolis'")

    @property
    def burn_in_steps(self) -> int:
        return max(1, int(round(self.burn_tau / self.dt)))

    @property
    def production_steps(self) -> int:
        return max(1, int(round(self.production_tau / self.dt)))


def parse_case(case_id: str) -> TrappedCase:
    ho_match = HO_CASE_RE.match(case_id)
    if ho_match is not None:
        return TrappedCase(
            n_particles=int(ho_match.group("n")),
            rod_length=float(ho_match.group("A")),
        )
    raise ValueError(
        f"invalid case id: {case_id}. Use N*_A* harmonic-oscillator units, e.g. N8_A0.2"
    )


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def default_parallel_workers(seed_count: int, *, max_workers: int = MAX_AUTO_WORKERS) -> int:
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    capped = min(seed_count, max_workers)
    if seed_count <= max_workers:
        return seed_count
    for preferred in (6, 5):
        if preferred <= capped and seed_count % preferred == 0:
            return preferred
    return capped


def resolve_parallel_workers(
    seed_count: int,
    requested_workers: int | None,
    *,
    max_workers: int = MAX_AUTO_WORKERS,
) -> int:
    if requested_workers is None or requested_workers == 0:
        return default_parallel_workers(seed_count, max_workers=max_workers)
    if requested_workers < 0:
        raise ValueError("parallel_workers must be non-negative")
    return min(seed_count, requested_workers)


def build_guide(
    case: TrappedCase,
    system: OpenLineHardRodSystem,
    trap: HarmonicTrap,
    *,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    relative_alpha: float | None = None,
    contact_beta: float | None = None,
    response_lambda: float | None = None,
) -> (
    ReducedTGHardRodGuide
    | ContactCorrectedReducedTGHardRodGuide
    | GapHCorrectedHardRodGuide
    | FixedGuideQuadraticResponse
):
    """Build only the importance-sampling guide for a trapped DMC run."""

    if guide_family not in GUIDE_FAMILIES:
        raise ValueError(f"unknown guide family: {guide_family}")
    if guide_family == "gap-h-corrected":
        if relative_alpha is not None or contact_beta is not None:
            raise ValueError("gap-h-corrected owns its shape; omit relative_alpha and contact_beta")
        guide = GapHCorrectedHardRodGuide(
            system=system,
            trap=trap,
            alpha=case.omega,
        )
    elif guide_family == "contact-corrected-reduced-tg":
        if contact_beta is None:
            raise ValueError("contact-corrected-reduced-tg requires an explicit contact_beta")
        guide = ContactCorrectedReducedTGHardRodGuide(
            system=system,
            trap=trap,
            alpha=case.omega,
            relative_alpha=relative_alpha,
            contact_beta=contact_beta,
        )
    else:
        if contact_beta is not None:
            raise ValueError("contact_beta requires guide_family='contact-corrected-reduced-tg'")
        guide = ReducedTGHardRodGuide(
            system=system,
            trap=trap,
            alpha=case.omega,
            relative_alpha=relative_alpha,
        )
    if response_lambda is not None:
        guide = FixedGuideQuadraticResponse(
            base_guide=guide,
            lambda_value=response_lambda,
            lambda0=BASE_TRAP_QUADRATIC_COUPLING,
            center=system.center,
        )
    return guide


def build_case_geometry(case: TrappedCase) -> tuple[OpenLineHardRodSystem, HarmonicTrap]:
    return (
        OpenLineHardRodSystem(n_particles=case.n_particles, rod_length=case.rod_length),
        HarmonicTrap(omega=case.omega),
    )


def make_grid(controls: DMCRunControls, case: TrappedCase | None = None) -> np.ndarray:
    extent = controls.grid_extent
    if case is None:
        return np.linspace(-extent, extent, controls.n_bins)
    system, trap = build_case_geometry(case)
    for _attempt in range(8):
        grid = np.linspace(-extent, extent, controls.n_bins)
        try:
            lda = lda_density_profile(
                grid,
                trap.values(grid),
                n_particles=float(system.n_particles),
                rod_length=system.rod_length,
            )
        except ValueError as exc:
            if not any(
                message in str(exc)
                for message in (
                    "density cloud",
                    "grid is too small for the requested hard-rod excluded volume",
                )
            ):
                raise
            extent *= 1.5
        else:
            dynamic_extent = max(extent, 20.0, 6.0 * lda_rms_radius(lda, center=trap.center))
            if dynamic_extent <= extent * (1.0 + 1e-5):
                return grid
            extent = dynamic_extent
    raise ValueError("failed to build a grid containing the LDA density cloud")


def lda_target_rms(case: TrappedCase, controls: DMCRunControls, grid: np.ndarray) -> float:
    system, trap = build_case_geometry(case)
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    return lda_rms_radius(lda, center=trap.center)


def run_streaming_seed(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    density_grid: np.ndarray | None = None,
    progress: ProgressBar | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume: bool = False,
    initialization: InitializationControls | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
    transport_observer: DMCTransportObserver | None = None,
) -> DMCStreamingSummary:
    if controls.response_lambda is not None and collective_rn is not None:
        raise ValueError("quadratic energy-response sampling requires collective RN to be disabled")
    rng = np.random.default_rng(seed)
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
        response_lambda=controls.response_lambda,
    )
    scheduled_move = (
        None
        if collective_rn is None
        else build_collective_rn_extension(
            system=system,
            trap=trap,
            controls=collective_rn,
            dt=controls.dt,
        )
    )
    grid = make_grid(controls, case) if density_grid is None else density_grid
    initialization = InitializationControls() if initialization is None else initialization
    target_initial_rms = (
        lda_target_rms(case, controls, grid)
        if initialization.mode in {"lda-rms-lattice", "lda-rms-logspread"}
        else None
    )
    initial = prepare_initial_walkers(
        system,
        guide,
        controls.walkers,
        rng,
        controls=initialization,
        target_initial_rms=target_initial_rms,
    )
    checkpoint_path = (
        checkpoint_dir / f"{case.case_id}_seed{seed}.npz" if checkpoint_dir is not None else None
    )
    summary = run_dmc_streaming(
        initial_walkers=initial.positions,
        guide=guide,
        system=system,
        density_grid=grid,
        config=DMCConfig(
            ess_resample_fraction=controls.ess_resample_fraction,
            local_step_method=controls.local_step_method,
        ),
        scheduled_move=scheduled_move,
        rng=rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
        checkpoint_path=checkpoint_path,
        checkpoint_every_steps=checkpoint_every_steps,
        resume=resume,
        checkpoint_identity={
            "case_id": case.case_id,
            "seed": int(seed),
            "guide_family": guide_family,
            "guide_parameters": {
                "relative_alpha": controls.relative_alpha,
                "contact_beta": controls.contact_beta,
                "response_lambda": controls.response_lambda,
            },
            "initialization": asdict(initialization),
        },
        transport_observer=transport_observer,
    )
    summary.metadata.update(initial.metadata)
    if collective_rn is not None:
        summary.metadata["collective_rn"] = collective_rn.to_metadata()
    summary.metadata.update(case.unit_metadata())
    summary.metadata["guide_family"] = guide_family
    summary.metadata["resolved_guide_family"] = _guide_family_name(guide)
    summary.metadata["local_step_method"] = controls.local_step_method
    if controls.relative_alpha is not None:
        summary.metadata["relative_alpha"] = controls.relative_alpha
    if controls.contact_beta is not None:
        summary.metadata["contact_beta"] = controls.contact_beta
    if controls.response_lambda is not None:
        summary.metadata["response_lambda"] = controls.response_lambda
        summary.metadata["response_lambda0"] = BASE_TRAP_QUADRATIC_COUPLING
    initial_rms_value = initial.metadata["initial_rms_mean"]
    if not isinstance(initial_rms_value, int | float):
        raise RuntimeError("initial_rms_mean metadata must be numeric")
    initial_rms = float(initial_rms_value)
    summary.metadata["initial_to_production_rms_ratio"] = (
        float(summary.rms_radius / initial_rms) if initial_rms > 0.0 else float("nan")
    )
    return summary


def validate_streaming_against_raw(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    *,
    progress: ProgressBar | None = None,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
        response_lambda=controls.response_lambda,
    )
    scheduled_move = (
        None
        if collective_rn is None
        else build_collective_rn_extension(
            system=system,
            trap=trap,
            controls=collective_rn,
            dt=controls.dt,
        )
    )
    grid = make_grid(controls, case)
    raw_rng = np.random.default_rng(seed)
    raw_initial = initial_walkers(system, controls.walkers, raw_rng)
    config = DMCConfig(
        ess_resample_fraction=controls.ess_resample_fraction,
        local_step_method=controls.local_step_method,
    )
    raw = run_dmc(
        initial_walkers=raw_initial,
        guide=guide,
        system=system,
        config=config,
        scheduled_move=scheduled_move,
        rng=raw_rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
    )
    streaming_rng = np.random.default_rng(seed)
    streaming_initial = initial_walkers(system, controls.walkers, streaming_rng)
    streaming = run_dmc_streaming(
        initial_walkers=streaming_initial,
        guide=guide,
        system=system,
        density_grid=grid,
        config=config,
        scheduled_move=scheduled_move,
        rng=streaming_rng,
        dt=controls.dt,
        burn_in_steps=controls.burn_in_steps,
        production_steps=controls.production_steps,
        store_every=controls.store_every,
        progress=progress,
    )
    valid_mask = np.asarray([system.is_valid(snapshot) for snapshot in raw.snapshots], dtype=bool)
    raw_observables = estimate_weighted_observables(
        raw.snapshots,
        raw.local_energies,
        raw.weights,
        valid_mask,
        grid,
        center=system.center,
        n_particles=system.n_particles,
    )
    return {
        "seed": seed,
        "case_id": case.case_id,
        "streaming_matches_raw": bool(
            np.allclose(streaming.mixed_energy, raw_observables.mixed_energy)
            and np.allclose(streaming.r2_radius, raw_observables.r2_radius)
            and np.allclose(streaming.density, raw_observables.density.n_x)
        ),
        "mixed_energy_diff": float(streaming.mixed_energy - raw_observables.mixed_energy),
        "r2_radius_diff": float(streaming.r2_radius - raw_observables.r2_radius),
        "density_max_abs_diff": float(
            np.max(np.abs(streaming.density - raw_observables.density.n_x))
        ),
        "streaming_sample_count": streaming.sample_count,
        "raw_sample_count": int(raw.snapshots.shape[0]),
        "raw_guide_batch_backend": raw.metadata["guide_batch_backend"],
        "streaming_guide_batch_backend": streaming.metadata["guide_batch_backend"],
        "guide_family": guide_family,
        "collective_rn_enabled": collective_rn is not None,
        "resolved_guide_family": _guide_family_name(guide),
    }


def summarize_case(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    *,
    parallel_workers: int | None = None,
    progress: ProgressBar | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume_seed_checkpoints: bool = False,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> dict[str, Any]:
    grid = make_grid(controls, case)
    worker_count = resolve_parallel_workers(len(seeds), parallel_workers)
    seed_summaries, actual_worker_count = _run_seed_summaries(
        case,
        controls,
        seeds,
        grid,
        worker_count=worker_count,
        progress=progress,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=checkpoint_every_steps,
        resume_seed_checkpoints=resume_seed_checkpoints,
        collective_rn=collective_rn,
        guide_family=guide_family,
    )
    density = np.mean([summary.density for summary in seed_summaries], axis=0)
    energy_values = np.asarray([summary.mixed_energy for summary in seed_summaries], dtype=float)
    rms_values = np.asarray([summary.rms_radius for summary in seed_summaries], dtype=float)
    system, trap = build_case_geometry(case)
    guide = build_guide(
        case,
        system,
        trap,
        guide_family=guide_family,
        relative_alpha=controls.relative_alpha,
        contact_beta=controls.contact_beta,
        response_lambda=controls.response_lambda,
    )
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    return {
        "case_id": case.case_id,
        "n_particles": case.n_particles,
        "rod_length": case.rod_length,
        **case.unit_metadata(),
        "seeds": seeds,
        "controls": controls_to_dict(controls),
        "effective_grid_extent": float(max(abs(grid[0]), abs(grid[-1]))),
        "mixed_energy": _mean(energy_values),
        "mixed_energy_seed_stderr": _stderr(energy_values),
        "rms_radius": _mean(rms_values),
        "rms_radius_seed_stderr": _stderr(rms_values),
        "density_integral": float(np.sum(density) * (grid[1] - grid[0])),
        "lost_out_of_grid_sample_count_total": int(
            sum(summary.lost_out_of_grid_sample_count for summary in seed_summaries)
        ),
        "lda_total_energy": lda_total_energy(lda, rod_length=system.rod_length),
        "energy_dmc_minus_lda": _mean(energy_values)
        - lda_total_energy(lda, rod_length=system.rod_length),
        "lda_rms_radius": lda_rms_radius(lda, center=trap.center),
        "rms_dmc_minus_lda": _mean(rms_values) - lda_rms_radius(lda, center=trap.center),
        "density_relative_l2": relative_density_l2_error(grid, density, lda.n_x),
        "seed_count": len(seeds),
        "parallel_workers": actual_worker_count,
        "parallel_workers_requested": worker_count,
        "collective_rn_enabled": collective_rn is not None,
        "collective_rn_controls": (None if collective_rn is None else collective_rn.to_metadata()),
        "guide_family": guide_family,
        "resolved_guide_family": _guide_family_name(guide),
        "guide_batch_backend": ",".join(
            sorted({str(summary.metadata["guide_batch_backend"]) for summary in seed_summaries})
        ),
        "target_backend": ",".join(
            sorted({str(summary.metadata.get("target_backend", "")) for summary in seed_summaries})
        ),
        "proposal_backend": ",".join(
            sorted(
                {str(summary.metadata.get("proposal_backend", "")) for summary in seed_summaries}
            )
        ),
        "seed_summaries": [
            {
                "seed": seed,
                "mixed_energy": summary.mixed_energy,
                "rms_radius": summary.rms_radius,
                "density_integral": summary.density_integral,
                "lost_out_of_grid_sample_count": summary.lost_out_of_grid_sample_count,
                "killed_count": summary.metadata["killed_count"],
                "resample_count": summary.metadata["resample_count"],
                "ess_min": summary.metadata["ess_min"],
                "ess_mean": summary.metadata["ess_mean"],
                "ess_fraction_min": summary.metadata["ess_fraction_min"],
                "guide_batch_backend": summary.metadata["guide_batch_backend"],
                "target_backend": summary.metadata.get("target_backend", ""),
                "proposal_backend": summary.metadata.get("proposal_backend", ""),
                "collective_rn": summary.metadata.get("collective_rn"),
            }
            for seed, summary in zip(seeds, seed_summaries, strict=True)
        ],
    }


def _run_seed_summaries(
    case: TrappedCase,
    controls: DMCRunControls,
    seeds: list[int],
    density_grid: np.ndarray,
    *,
    worker_count: int,
    progress: ProgressBar | None,
    checkpoint_dir: Path | None,
    checkpoint_every_steps: int | None,
    resume_seed_checkpoints: bool,
    collective_rn: CollectiveRNControls | None,
    guide_family: str,
) -> tuple[list[DMCStreamingSummary], int]:
    return run_seed_batch(
        seeds,
        worker_count=worker_count,
        progress=progress,
        submit_seed=lambda executor, seed, progress_queue: executor.submit(
            _run_seed_worker,
            case,
            controls,
            seed,
            density_grid,
            progress_queue,
            checkpoint_dir,
            checkpoint_every_steps,
            resume_seed_checkpoints,
            collective_rn,
            guide_family,
        ),
        run_serial_seed=lambda seed: run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=progress,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_steps=checkpoint_every_steps,
            resume=resume_seed_checkpoints,
            collective_rn=collective_rn,
            guide_family=guide_family,
        ),
    )


def _run_seed_worker(
    case: TrappedCase,
    controls: DMCRunControls,
    seed: int,
    density_grid: np.ndarray,
    progress_queue: Any | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every_steps: int | None = None,
    resume_seed_checkpoints: bool = False,
    collective_rn: CollectiveRNControls | None = None,
    guide_family: str = DEFAULT_GUIDE_FAMILY,
) -> tuple[int, DMCStreamingSummary]:
    worker_progress = QueuedProgress(progress_queue) if progress_queue is not None else None
    try:
        return seed, run_streaming_seed(
            case,
            controls,
            seed,
            density_grid=density_grid,
            progress=worker_progress,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_steps=checkpoint_every_steps,
            resume=resume_seed_checkpoints,
            collective_rn=collective_rn,
            guide_family=guide_family,
        )
    finally:
        if worker_progress is not None:
            worker_progress.flush()


def controls_to_dict(controls: DMCRunControls) -> dict[str, float | int | str | bool]:
    values: dict[str, float | int | str | bool] = {
        "dt": controls.dt,
        "walkers": controls.walkers,
        "burn_tau": controls.burn_tau,
        "production_tau": controls.production_tau,
        "burn_in_steps": controls.burn_in_steps,
        "production_steps": controls.production_steps,
        "store_every": controls.store_every,
        "grid_extent": controls.grid_extent,
        "n_bins": controls.n_bins,
        "local_step_method": controls.local_step_method,
    }
    if not np.isclose(controls.ess_resample_fraction, 0.35):
        values["ess_resample_fraction"] = controls.ess_resample_fraction
    if controls.relative_alpha is not None:
        values["relative_alpha"] = controls.relative_alpha
    if controls.contact_beta is not None:
        values["contact_beta"] = controls.contact_beta
    if controls.response_lambda is not None:
        values["response_lambda"] = controls.response_lambda
    return values


def _guide_family_name(guide: object) -> str:
    if isinstance(guide, FixedGuideQuadraticResponse):
        return f"quadratic-response({_guide_family_name(guide.base_guide)})"
    if isinstance(guide, ContactCorrectedReducedTGHardRodGuide):
        return "contact-corrected-reduced-tg"
    if isinstance(guide, GapHCorrectedHardRodGuide):
        return "gap-h-corrected"
    if isinstance(guide, ReducedTGHardRodGuide):
        return "reduced-tg"
    return type(guide).__name__


def dmc_run_config(
    *,
    run_kind: str,
    cases: list[str],
    seeds: list[int],
    controls: DMCRunControls,
    collective_rn: CollectiveRNControls | None = None,
    parallel_workers: int | None,
    checkpoint_every_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "run_kind": run_kind,
        "cases": cases,
        "seeds": seeds,
        "controls": controls_to_dict(controls),
        "collective_rn": None if collective_rn is None else collective_rn.to_metadata(),
        "parallel_workers": parallel_workers,
        "checkpoint_every_steps": checkpoint_every_steps,
    }


def write_summary(output_dir: Path, payload: dict[str, Any]) -> Path:
    path = ensure_dir(output_dir) / "summary.json"
    write_json(path, payload)
    return path


def write_case_table(output_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None
    fields = [
        "case_id",
        "seed_count",
        "case_parameterization",
        "rod_length_ho",
        "mixed_energy",
        "mixed_energy_seed_stderr",
        "rms_radius",
        "rms_radius_seed_stderr",
        "density_integral",
        "density_relative_l2",
        "lda_total_energy",
        "energy_dmc_minus_lda",
        "lda_rms_radius",
        "rms_dmc_minus_lda",
        "lost_out_of_grid_sample_count_total",
        "collective_rn_enabled",
        "guide_family",
        "resolved_guide_family",
        "guide_batch_backend",
        "target_backend",
        "proposal_backend",
    ]
    path = ensure_dir(output_dir) / "case_table.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def write_dmc_run_artifacts(
    output_dir: Path,
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    run_name: str,
    config: dict[str, Any],
    command: list[str] | None,
    extra_artifacts: list[Path] | None = None,
) -> dict[str, Path]:
    summary_path = write_summary(output_dir, payload)
    table_path = write_case_table(output_dir, rows)
    artifacts = [summary_path]
    if table_path is not None:
        artifacts.append(table_path)
    if extra_artifacts:
        artifacts.extend(extra_artifacts)
    manifest_path = write_run_manifest(
        output_dir,
        run_name=run_name,
        config=config,
        artifacts=artifacts,
        schema_version=str(payload["schema_version"]),
        provenance=build_run_provenance(command),
    )
    return {
        "summary": summary_path,
        "manifest": manifest_path,
        **({"case_table": table_path} if table_path is not None else {}),
    }


def load_completed_case_rows(
    checkpoint_path: Path,
    *,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not checkpoint_path.exists():
        return {}
    checkpoint = read_checkpoint(checkpoint_path)
    if expected_config is not None:
        expected = config_fingerprint(expected_config)
        observed = checkpoint.get("config_fingerprint")
        if observed != expected:
            raise ValueError("checkpoint config fingerprint does not match requested run")
    return {
        str(row["case_id"]): row
        for row in checkpoint.get("completed_cases", [])
        if isinstance(row, dict) and "case_id" in row
    }


def write_case_checkpoint(
    checkpoint_path: Path,
    *,
    status: str,
    config: dict[str, Any],
    completed_cases: list[dict[str, Any]],
    pending_cases: list[str],
) -> Path:
    return write_checkpoint(
        checkpoint_path,
        {
            "status": status,
            "config_fingerprint": config_fingerprint(config),
            "run_config": config,
            "completed_cases": completed_cases,
            "pending_cases": pending_cases,
        },
    )


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _stderr(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def dmc_total_steps(
    controls: DMCRunControls,
    *,
    seed_count: int,
    raw_validation: bool = False,
) -> int:
    multiplier = 2 if raw_validation else 1
    return multiplier * seed_count * (controls.burn_in_steps + controls.production_steps)


def dmc_progress_bar(
    *,
    controls: DMCRunControls,
    seed_count: int,
    label: str,
    enabled: bool,
    raw_validation: bool = False,
):
    return progress_bar(
        total=dmc_total_steps(controls, seed_count=seed_count, raw_validation=raw_validation),
        label=label,
        enabled=enabled,
    )
