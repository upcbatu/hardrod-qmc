from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hrdmc.sampling.initial_conditions import initial_walkers_with_metadata
from hrdmc.system.settings import TrappedCase, build_case_geometry
from hrdmc.theory.lda import lda_density_profile, lda_rms_radius
from hrdmc.theory.tonks_girardeau import trapped_tg_rms_radius
from hrdmc.validation.sampler_equivalence.models import density_bin_edges

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class VMCInitialBatch:
    positions: FloatArray
    metadata: dict[str, Any]
def prepare_vmc_initial_batch(
    case: TrappedCase,
    *,
    walkers: int,
    seed: int,
) -> VMCInitialBatch:
    """Create independent LDA-scale lattice walkers for one sampler seed."""
    if walkers < 1:
        raise ValueError("VMC initialization requires at least one walker")
    rng = np.random.default_rng(seed)
    system, _trap = build_case_geometry(case)
    target_rms = _target_initial_rms(case)
    initial = initial_walkers_with_metadata(
        system,
        walkers,
        rng,
        initialization_mode="lda-rms-lattice",
        target_initial_rms=target_rms,
        init_width_log_sigma=0.0,
    )
    return VMCInitialBatch(
        positions=np.asarray(initial.positions, dtype=float),
        metadata={
            **initial.metadata,
            "initializer_seed": int(seed),
            "target_initial_rms": target_rms,
        },
    )
def _target_initial_rms(case: TrappedCase) -> float:
    if case.rod_length == 0.0:
        return trapped_tg_rms_radius(case.n_particles, case.omega)
    system, trap = build_case_geometry(case)
    edges = density_bin_edges(case, bins=4_000)
    grid = 0.5 * (edges[:-1] + edges[1:])
    profile = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    return lda_rms_radius(profile, center=trap.center)
