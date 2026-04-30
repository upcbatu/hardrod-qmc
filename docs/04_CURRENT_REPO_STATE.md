# Current Repository State

This repository is currently at a **homogeneous hard-rod VMC scaffold stage**. It is not yet a complete trapped-system QMC benchmark code.

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
- blocking analysis and bias/MSE utilities.
  files: `src/hrdmc/analysis/blocking.py`, `src/hrdmc/analysis/metrics.py`

## Not Yet Implemented

- open-line trapped hard-rod geometry;
  expected area: `src/hrdmc/systems/`
- harmonic trap potential;
  expected area: `src/hrdmc/systems/external_potential.py`
- trapped initial configurations satisfying hard-rod exclusion;
  expected area: `src/hrdmc/systems/`
- non-periodic density-profile estimation for trapped coordinates;
  expected area: `src/hrdmc/estimators/density.py`
- benchmark-tier labeling for trapped data;
  expected area: `src/hrdmc/analysis/` or result metadata
- QMC versus LDA error and failure-map analysis;
  expected area: `src/hrdmc/analysis/`
- trapped VMC smoke experiment;
  expected area: `experiments/`
- production DMC engine;
  file: `src/hrdmc/monte_carlo/dmc.py`
- trapped QMC benchmark versus LDA orchestration workflow;
  expected area: `experiments/`

The current DMC layer is a result contract and data-structure seam. It does not yet produce production DMC trajectories, so DMC outputs should be treated as candidate references until validated.

## Validation Status

The current tests cover ring geometry, estimator output shapes, theory formulas, LDA normalization, and analysis utilities.
files: `tests/`

The VMC run is currently an end-to-end integration check for the homogeneous scaffold.
files: `experiments/00_smoke_vmc.py`, `experiments/configs/smoke.json`

The homogeneous validation benchmark checks the exact all-pair trial local energy against finite-`N` ring references and can be run with `make validate-ring`.
files: `experiments/01_uniform_hard_rods_validation.py`

It should not be read as trapped-system validation or final DMC validation.
