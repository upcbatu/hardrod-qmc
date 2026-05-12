from __future__ import annotations

import numpy as np

from hrdmc.systems.open_line import OpenLineHardRodSystem
from hrdmc.wavefunctions import ReducedTGHardRodGuide
from hrdmc.workflows.dmc.rn_block_initial_conditions.controls import RNInitializationControls
from hrdmc.workflows.dmc.rn_block_initial_conditions.lattice import (
    InitialWalkerBatch,
    initial_walkers_with_metadata,
)
from hrdmc.workflows.dmc.rn_block_initial_conditions.preburn import breathing_preburn_walkers


def prepare_initial_walkers(
    system: OpenLineHardRodSystem,
    guide: ReducedTGHardRodGuide,
    walkers: int,
    rng: np.random.Generator,
    *,
    controls: RNInitializationControls,
    target_initial_rms: float | None = None,
) -> InitialWalkerBatch:
    controls.validate()
    initial = initial_walkers_with_metadata(
        system,
        walkers,
        rng,
        initialization_mode=controls.mode,
        target_initial_rms=target_initial_rms,
        init_width_log_sigma=controls.init_width_log_sigma,
    )
    positions, preburn_metadata = breathing_preburn_walkers(
        initial.positions,
        guide,
        rng,
        steps=controls.breathing_preburn_steps,
        log_step=controls.breathing_preburn_log_step,
    )
    metadata = dict(initial.metadata)
    metadata.update(preburn_metadata)
    return InitialWalkerBatch(positions=positions, metadata=metadata)
