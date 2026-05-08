# Current Repository State

This repository is currently at a **homogeneous validation plus initial trapped diagnostic stage**. It is not yet a complete trapped-system QMC benchmark code.

## Implemented

- homogeneous one-dimensional hard-rod geometry on a periodic ring;
  files: `src/hrdmc/systems/hard_rods.py`
- homogeneous hard-rod EOS, finite ring energy, chemical potential, chemical-potential inversion, and LDA support;
  files: `src/hrdmc/theory/`
- Jastrow-based trial-wavefunction implementation for the ring scaffold;
  files: `src/hrdmc/wavefunctions/jastrow.py`
- initial homogeneous VMC workflow;
  files: `src/hrdmc/monte_carlo/vmc.py`, `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`
- homogeneous ring validation benchmark over particle numbers and packing fractions;
  files: `experiments/01_uniform_hard_rods_validation.py`
- local-energy validation for the all-pair reduced hard-rod trial;
  files: `src/hrdmc/wavefunctions/jastrow.py`, `src/hrdmc/estimators/local_energy.py`
- observable implementations for `g(r)`, `S(k)`, and periodic density `n(x)`;
  files: `src/hrdmc/estimators/`
- open-line trapped hard-rod geometry, harmonic trap potential, and trapped diagnostic trial state;
  files: `src/hrdmc/systems/open_line.py`, `src/hrdmc/systems/external_potential.py`, `src/hrdmc/wavefunctions/trapped.py`
- non-periodic density estimation plus raw and relative density L2 comparison;
  files: `src/hrdmc/estimators/density.py`, `src/hrdmc/analysis/metrics.py`
- trapped VMC smoke experiment with VMC-versus-LDA density diagnostics;
  files: `experiments/02_trapped_vmc_smoke.py`
- trapped VMC diagnostic grid over `N = 4, 8` and `omega = 0.05, 0.10, 0.20`;
  files: `experiments/03_trapped_vmc_diagnostic_grid.py`
- trapped VMC seed-stability diagnostic for replicate spread checks;
  files: `experiments/04_trapped_vmc_seed_stability.py`, `src/hrdmc/analysis/stability.py`
- trapped VMC alpha-scan diagnostic and cloud-radius observables;
  files: `experiments/05_trapped_vmc_alpha_scan.py`, `src/hrdmc/estimators/cloud.py`
- blocking analysis and bias/MSE utilities;
  files: `src/hrdmc/analysis/blocking.py`, `src/hrdmc/analysis/metrics.py`

## Not Yet Implemented

- benchmark-tier expansion beyond the initial VMC diagnostic label;
  expected area: `src/hrdmc/analysis/` or result metadata
- QMC/DMC versus LDA error and failure-map analysis across parameter sweeps;
  expected area: `src/hrdmc/analysis/`
- production DMC engine;
  file: `src/hrdmc/monte_carlo/dmc.py`
- trapped QMC benchmark versus LDA orchestration workflow;
  expected area: `experiments/`

The current DMC layer is a result contract and data-structure seam. It does not yet produce production DMC trajectories, so DMC outputs should be treated as candidate references until validated.

## Validation Status

The current tests cover ring/open-line geometry, harmonic trap values, estimator output shapes, theory formulas, LDA normalization, and analysis utilities.
files: `tests/`

The VMC run is currently an end-to-end integration check for the homogeneous scaffold.
files: `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`

The homogeneous validation benchmark checks the exact all-pair trial local energy against finite-`N` ring references and can be run with `make validate-ring`.
files: `experiments/01_uniform_hard_rods_validation.py`

The trapped smoke run is a VMC diagnostic and can be run with `make smoke-trap`. The trapped diagnostic grid can be run with `make diagnose-trap-grid`. The seed-stability diagnostic can be run with `make diagnose-trap-seeds`. The alpha scan can be run with `make scan-trap-alpha`. They check sampled density, LDA normalization on the same grid, density L2 differences, RMS radius, and replicate spread. They should not be treated as final trapped benchmarks.
files: `experiments/02_trapped_vmc_smoke.py`, `experiments/03_trapped_vmc_diagnostic_grid.py`, `experiments/04_trapped_vmc_seed_stability.py`, `experiments/05_trapped_vmc_alpha_scan.py`

Benchmark interpretation and remaining validation checks are maintained in `docs/validation/README.md`.

It should not be read as trapped-system validation or final DMC validation.
