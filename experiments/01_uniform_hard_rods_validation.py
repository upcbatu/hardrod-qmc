from __future__ import annotations

import json
from pathlib import Path

from hrdmc.io.artifacts import ensure_dir, write_json
from hrdmc.systems.hard_rods import HardRodSystem
from hrdmc.theory import (
    excluded_length,
    hard_rod_energy_per_particle,
    hard_rod_finite_ring_energy_per_particle,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = ensure_dir(repo_root / "results" / "homogeneous_validation")

    n_particles = 21
    rod_length = 0.5
    packing_fractions = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70]

    rows = []
    for eta in packing_fractions:
        density = eta / rod_length
        length = n_particles / density
        system = HardRodSystem(
            n_particles=n_particles,
            length=length,
            rod_length=rod_length,
        )
        finite_energy = hard_rod_finite_ring_energy_per_particle(
            n_particles=system.n_particles,
            length=system.length,
            rod_length=system.rod_length,
        )
        thermodynamic_energy = hard_rod_energy_per_particle(system.density, system.rod_length)
        rows.append(
            {
                "n_particles": system.n_particles,
                "length": system.length,
                "density": system.density,
                "rod_length": system.rod_length,
                "packing_fraction": system.packing_fraction,
                "excluded_length": excluded_length(
                    system.n_particles,
                    system.length,
                    system.rod_length,
                ),
                "finite_energy_per_particle": finite_energy,
                "thermodynamic_energy_per_particle": thermodynamic_energy,
                "finite_minus_thermodynamic": finite_energy - thermodynamic_energy,
            }
        )

    summary = {
        "status": "homogeneous hard-rod theory validation table",
        "units": "hbar^2/(2m)=1",
        "rows": rows,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
