# Changelog

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
- Added tests for hard-rod theory formulas and LDA normalization.

### Removed

- Removed the old estimator-cost benchmark path and estimator-family analysis helpers.
- Removed pure-estimator and forward-walking infrastructure from the active architecture.
