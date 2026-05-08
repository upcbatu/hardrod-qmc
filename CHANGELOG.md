# Changelog

## 2026-05-08

### Added

- Added normalization/reference checks for homogeneous `g(r)`, `S(k)`, and periodic density estimators.
- Added histogram-edge density integration, LDA boundary containment checks, and relative density L2 diagnostics.

### Changed

- Updated validation notes to distinguish initial homogeneous observable normalization checks from later literature-curve comparisons.
- Removed invalid blocking-analysis reporting from the homogeneous smoke output.
- Clarified homogeneous validation and structure-factor source-map wording.

## 2026-04-30

### Changed

- Reframed the thesis direction around trapped hard-rod QMC benchmarks of excluded-volume LDA.
- Split architecture ownership so `systems/` owns geometry and constraints, while `theory/` owns EOS, excluded-volume mapping, and LDA predictions.
- Updated proposal, architecture, workflow, source-map, and current-state docs to mirror the new owner boundaries.
- Updated the homogeneous VMC smoke path to read ring reference energies from `theory/`.

### Added

- Added `src/hrdmc/theory/` for hard-rod EOS, chemical-potential inversion, finite ring energy, and LDA density/energy predictions.
- Added package-level README files documenting owner boundaries.
- Added a runnable homogeneous validation table experiment.
- Added a homogeneous ring validation benchmark with all-pair trial local-energy checks and a `make validate-ring` target.
- Added tests for hard-rod theory formulas and LDA normalization.
- Added tests for the all-pair local-energy estimator against finite-`N` ring references.
- Added `docs/validation/` for thesis-facing validation notes.
- Added open-line trapped hard-rod geometry, harmonic trap support, trapped density diagnostics, and a trapped VMC smoke entrypoint.
- Added a trapped VMC diagnostic grid over particle number and trap strength.
- Added a trapped VMC seed-stability diagnostic for replicate spread checks.

### Removed

- Removed the old estimator-cost benchmark path and estimator-family analysis helpers.
- Removed pure-estimator and forward-walking infrastructure from the active architecture.
