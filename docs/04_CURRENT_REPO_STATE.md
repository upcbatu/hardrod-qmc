# Current Repository State

This repository is currently at a **homogeneous hard-rod VMC scaffold stage**. It is not yet a trapped-system QMC benchmark code.

## Implemented

- homogeneous one-dimensional hard-rod geometry on a periodic ring;
  files: `src/hrdmc/systems/hard_rods.py`
- finite-`N` and thermodynamic homogeneous hard-rod reference energies;
  files: `src/hrdmc/systems/hard_rods.py`
- Jastrow-based trial-wavefunction implementation for the ring scaffold;
  files: `src/hrdmc/wavefunctions/jastrow.py`
- initial homogeneous VMC workflow;
  files: `src/hrdmc/monte_carlo/vmc.py`, `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`
- observable implementations for `g(r)`, `S(k)`, and periodic density `n(x)`;
  files: `src/hrdmc/estimators/`
- blocking analysis, bias/MSE utilities, and estimator-family support code.
  files: `src/hrdmc/analysis/blocking.py`, `src/hrdmc/analysis/metrics.py`, `src/hrdmc/analysis/cost_accuracy.py`, `src/hrdmc/analysis/estimator_families.py`

## Not Yet Implemented

- open-line trapped hard-rod geometry;
  expected area: `src/hrdmc/systems/`
- harmonic trap potential;
  expected area: `src/hrdmc/systems/external_potential.py`
- trapped initial configurations satisfying hard-rod exclusion;
  expected area: `src/hrdmc/systems/`
- non-periodic density-profile estimation for trapped coordinates;
  expected area: `src/hrdmc/estimators/density.py`
- homogeneous hard-rod equation-of-state utilities beyond the current energy methods;
  expected area: `src/hrdmc/systems/` or `src/hrdmc/analysis/`
- LDA implementation for trapped hard rods;
  expected area: `src/hrdmc/analysis/`
- trapped VMC smoke experiment;
  expected area: `experiments/`
- production DMC engine;
  file: `src/hrdmc/monte_carlo/dmc.py`
- trapped QMC benchmark versus LDA comparison workflow;
  expected area: `experiments/`

## Reframed Infrastructure

The estimator-family code remains useful for labeling VMC, mixed DMC, extrapolated, and pure estimates. It is now support infrastructure. The main thesis comparison is a trapped benchmark of excluded-volume LDA accuracy and failures, not estimator cost-accuracy ranking.

The current DMC layer is a result contract and data-structure seam. It does not yet produce production DMC trajectories.

## Validation Status

The current tests are basic checks for ring geometry, estimator output shapes, and analysis utilities.
files: `tests/test_hard_rods.py`, `tests/test_estimators.py`, `tests/test_blocking.py`

The VMC run is currently an end-to-end integration check for the homogeneous scaffold.
files: `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`

It should not be read as trapped-system validation or final DMC validation.
