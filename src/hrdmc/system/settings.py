from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from hrdmc.system.geometry import HarmonicTrap, OpenLineHardRodSystem
from hrdmc.system.units import HO_TRAP_OMEGA, harmonic_oscillator_unit_metadata
from hrdmc.theory.lda import lda_density_profile, lda_rms_radius

HO_CASE_RE = re.compile(r"^N(?P<n>\d+)_A(?P<A>[0-9.]+)$")
THESIS_CASE_ORDER = (
    "N10_A0",
    "N10_A0.1",
    "N10_A1",
    "N10_A10",
    "N20_A0",
    "N20_A0.1",
    "N20_A1",
    "N20_A10",
)

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
    drift_limiter: str = "none"
    relative_alpha: float | None = None
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
        if self.drift_limiter not in {"none", "umrigar"}:
            raise ValueError("drift_limiter must be 'none' or 'umrigar'")
    @property
    def burn_in_steps(self) -> int:
        return max(1, round(self.burn_tau / self.dt))
    @property
    def production_steps(self) -> int:
        return max(1, round(self.production_tau / self.dt))

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
    }
    if not np.isclose(controls.ess_resample_fraction, 0.35):
        values["ess_resample_fraction"] = controls.ess_resample_fraction
    if controls.drift_limiter != "none":
        values["drift_limiter"] = controls.drift_limiter
    if controls.relative_alpha is not None:
        values["relative_alpha"] = controls.relative_alpha
    return values

def dmc_run_config(
    *,
    run_kind: str,
    cases: list[str],
    seeds: list[int],
    controls: DMCRunControls,
    parallel_workers: int | None,
    checkpoint_every_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "run_kind": run_kind,
        "cases": cases,
        "seeds": seeds,
        "controls": controls_to_dict(controls),
        "parallel_workers": parallel_workers,
        "checkpoint_every_steps": checkpoint_every_steps,
    }
