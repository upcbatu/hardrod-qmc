# Current Repository State

This repository is currently at a **promoted RN-DMC candidate package stage**.
The `src/hrdmc` package exposes the original VMC scaffold, a generic DMC
contract package, and an RN-corrected
collective-block DMC candidate.

## Implemented

- homogeneous one-dimensional hard-rod geometry on a periodic ring;
  files: `src/hrdmc/systems/hard_rods.py`
- reduced-coordinate hard-rod length identity;
  files: `src/hrdmc/systems/reduced.py`
- homogeneous hard-rod EOS, finite ring energy, chemical potential, chemical-potential inversion, and LDA support;
  files: `src/hrdmc/theory/`
- Jastrow-based trial-wavefunction implementation for the ring scaffold;
  files: `src/hrdmc/wavefunctions/jastrow.py`
- initial homogeneous VMC workflow;
  files: `src/hrdmc/monte_carlo/vmc.py`, `experiments/vmc/smoke.py`, `experiments/vmc/configs/smoke.json`
- homogeneous ring validation benchmark over particle numbers and packing fractions;
  files: `experiments/validation/homogeneous_ring.py`
- homogeneous finite-`a` exact ring validation grid including `N = 64`;
  files: `experiments/validation/homogeneous_ring_exact_grid.py`
- local-energy validation for the all-pair reduced hard-rod trial;
  files: `src/hrdmc/wavefunctions/jastrow.py`, `src/hrdmc/estimators/local_energy.py`
- observable implementations for `g(r)`, `S(k)`, and periodic density `n(x)`;
  files: `src/hrdmc/estimators/`
- open-line trapped hard-rod geometry, harmonic trap potential, and trapped diagnostic trial state;
  files: `src/hrdmc/systems/open_line.py`, `src/hrdmc/systems/external_potential.py`, `src/hrdmc/wavefunctions/trapped.py`
- non-periodic density estimation plus raw and relative density L2 comparison;
  files: `src/hrdmc/estimators/density.py`, `src/hrdmc/analysis/metrics.py`
- trapped VMC smoke experiment with VMC-versus-LDA density diagnostics;
  files: `experiments/vmc/trapped_smoke.py`
- trapped VMC diagnostic grid over `N = 4, 8` and `omega = 0.05, 0.10, 0.20`;
  files: `experiments/vmc/trapped_diagnostic_grid.py`
- trapped VMC seed-stability diagnostic for replicate spread checks;
  files: `experiments/vmc/trapped_seed_stability.py`, `src/hrdmc/analysis/stability.py`
- trapped VMC alpha-scan diagnostic and cloud-radius observables;
  files: `experiments/vmc/trapped_alpha_scan.py`, `src/hrdmc/estimators/cloud.py`
- blocking analysis and bias/MSE utilities;
  files: `src/hrdmc/analysis/blocking.py`, `src/hrdmc/analysis/metrics.py`
- an RN-corrected collective-block DMC candidate with injected system,
  guide, target-kernel, and proposal-kernel owners;
  files: `src/hrdmc/monte_carlo/dmc/rn_block/`
- an RN-block DMC smoke workflow for checking package wiring;
  files: `experiments/dmc/rn_block/smoke.py`
- streaming RN-block summary mode for mean, radius, and density accumulation
  without retaining raw snapshots;
  files: `src/hrdmc/monte_carlo/dmc/rn_block/engine.py`,
  `src/hrdmc/analysis/streaming.py`
- RN-block validation, single-case, and grid runners backed by the package API;
  files: `experiments/dmc/rn_block/validate_streaming.py`,
  `experiments/dmc/rn_block/single_case.py`, `experiments/dmc/rn_block/grid.py`
- RN-block single-case and grid runners can parallelize independent seeds with
  `--parallel-workers`; `0` chooses an automatic seed-count policy capped at 6;
- generic seed-batch execution, worker-to-parent progress propagation, and
  canonical artifact routing are owned outside experiment scripts;
  files: `src/hrdmc/runners/`, `src/hrdmc/artifacts/`
- RN-block workflow composition is owned by package workflow modules rather
  than command-line experiment scripts;
  files: `src/hrdmc/workflows/dmc/`
- trapped finite-`a` RN-DMC stationarity grid with R-hat, effective-sample,
  accounting, and warning/NO-GO diagnostics;
  files: `experiments/dmc/rn_block/trapped_stationarity_grid.py`,
  `src/hrdmc/analysis/chain_diagnostics.py`
- exact trapped Tonks-Girardeau harmonic validation for RN-block DMC in the
  zero-rod-length limit;
  files: `experiments/dmc/rn_block/exact_tg_trap.py`
- compact RN-DMC candidate tables and a result manifest;
  files: `docs/tables/`, `tests/fixtures/rn_results_manifest.json`

## Not Yet Implemented

- QMC/DMC versus LDA failure-map workflow;
  expected area: `src/hrdmc/analysis/` and `experiments/`
- release-grade citation metadata and archived result bundle;
  expected area: repository root plus external archive at release time

The current DMC layer is a contract package plus a candidate RN-block
implementation. The compact RN-DMC tables should be treated as validation
summaries, not as final release artifacts.

## Validation Status

The current tests cover ring/open-line geometry, harmonic trap values,
estimator output shapes, theory formulas, LDA normalization, and analysis
utilities.
files: `tests/`

The VMC run is currently an end-to-end integration check for the homogeneous scaffold.
files: `experiments/vmc/smoke.py`, `experiments/vmc/configs/smoke.json`

The homogeneous validation benchmark checks the exact all-pair trial local energy against finite-`N` ring references and can be run with `make validate-ring`.
files: `experiments/validation/homogeneous_ring.py`

The extended homogeneous exact ring grid checks the same finite-`N` hard-rod
identity across a wider `N` and packing-fraction grid, including `N = 64`, and
can be run with `make validate-ring-grid`.
files: `experiments/validation/homogeneous_ring_exact_grid.py`

The RN-DMC exact trapped validation checks the zero-length hard-core
Tonks-Girardeau harmonic limit against \(E_0=N^2\omega/\sqrt{2}\) and can be
run with `make validate-rn-exact`.
files: `experiments/dmc/rn_block/exact_tg_trap.py`

The trapped RN-DMC stationarity grid checks finite-`a` trapped runs for
R-hat, effective independent samples, stationarity, density accounting, and
finite/valid sample accounting. It can be run with
`make validate-rn-trapped-stationarity`. Its output is a statistical-control
diagnostic, not a final paper benchmark by itself.
files: `experiments/dmc/rn_block/trapped_stationarity_grid.py`

The trapped smoke run is a VMC diagnostic and can be run with `make smoke-trap`. The trapped diagnostic grid can be run with `make diagnose-trap-grid`. The seed-stability diagnostic can be run with `make diagnose-trap-seeds`. The alpha scan can be run with `make scan-trap-alpha`. They check sampled density, LDA normalization on the same grid, density L2 differences, RMS radius, and replicate spread. They should not be treated as final trapped benchmarks.
files: `experiments/vmc/trapped_smoke.py`, `experiments/vmc/trapped_diagnostic_grid.py`, `experiments/vmc/trapped_seed_stability.py`, `experiments/vmc/trapped_alpha_scan.py`

Benchmark interpretation and remaining validation checks are maintained in `docs/validation/README.md`.

RN-DMC method notes are maintained in `docs/RN_BLOCK_DMC.md`. Compact
candidate-result tables are maintained in `docs/tables/`.

This repo state should not be read as a final released DMC package until
long-run examples and release metadata are in place.
