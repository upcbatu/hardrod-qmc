from __future__ import annotations

from hrdmc.theory import lda_density_profile, lda_rms_radius
from hrdmc.workflows.dmc.rn_block import RNCase, RNRunControls, build_case_objects, make_grid


def test_grid_extent_scales_with_lda_rms_for_weak_trap() -> None:
    case = RNCase(n_particles=8, rod_length=0.5, omega=0.00944)
    controls = RNRunControls(
        dt=0.00125,
        walkers=16,
        tau_block=0.01,
        rn_cadence_tau=0.005,
        burn_tau=1.0,
        production_tau=1.0,
        store_every=10,
        grid_extent=20.0,
        n_bins=240,
    )

    grid = make_grid(controls, case)
    system, trap, _guide, _target, _proposal = build_case_objects(case)
    lda = lda_density_profile(
        grid,
        trap.values(grid),
        n_particles=float(system.n_particles),
        rod_length=system.rod_length,
    )
    rms = lda_rms_radius(lda, center=trap.center)

    assert max(abs(grid[0]), abs(grid[-1])) >= 6.0 * rms * 0.99
    assert abs(lda.integrated_particles - system.n_particles) < 1e-6
