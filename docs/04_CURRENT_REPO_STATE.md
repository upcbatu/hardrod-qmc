# Current Repository State

This repository is currently at a **VMC-based implementation stage**, not yet a complete DMC benchmark code.

## Implemented

- one-dimensional hard-rod geometry and reference energies;
  files: `src/hrdmc/systems/hard_rods.py`
- Jastrow-based trial-wavefunction implementation;
  files: `src/hrdmc/wavefunctions/jastrow.py`
- initial VMC workflow;
  files: `src/hrdmc/monte_carlo/vmc.py`, `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`
- observable implementations for `g(r)`, `S(k)`, and `n(x)`;
  files: `src/hrdmc/estimators/`
- blocking analysis, bias/MSE utilities, and estimator-family definitions.
  files: `src/hrdmc/analysis/blocking.py`, `src/hrdmc/analysis/metrics.py`, `src/hrdmc/analysis/cost_accuracy.py`, `src/hrdmc/analysis/estimator_families.py`

## Not Yet Implemented

- production DMC engine;
  file: `src/hrdmc/monte_carlo/dmc.py`
- mixed-estimator production from real DMC runs;
  related files: `src/hrdmc/monte_carlo/dmc.py`, `src/hrdmc/analysis/estimator_families.py`
- pure-estimator production through forward walking;
  related files: `src/hrdmc/monte_carlo/dmc.py`, `src/hrdmc/analysis/estimator_families.py`
- full extrapolated-estimator benchmark workflow;
  related files: `src/hrdmc/analysis/estimator_families.py`, `experiments/02_estimator_cost_accuracy.py`
- final density-sweep and cost-accuracy benchmark scripts.
  files: `experiments/01_uniform_hard_rods_validation.py`, `experiments/02_estimator_cost_accuracy.py`

## Validation Status

The current tests are basic checks for geometry, estimator output shapes, and analysis utilities.
files: `tests/test_hard_rods.py`, `tests/test_estimators.py`, `tests/test_blocking.py`

The VMC run is currently used only as an end-to-end integration check.
files: `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`

It should not be read as final scientific validation of the thesis workflow.
