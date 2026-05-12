# Changelog

## 2026-05-12

### Added

- Added breathing-aware RN-DMC initialization controls for LDA-RMS lattice/logspread starts and optional guide-Metropolis breathing preburn.
- Added configurable fixed-scale RN collective proposal mixtures for weak-trap stationarity probes.

### Changed

- Reduced the public RN-DMC release runner surface and tightened initialization metadata typing.
- Updated trapped RN-block blocking-plateau detection to use Flyvbjerg-Petersen/pyblock-compatible standard-error uncertainty instead of relying only on a fixed relative-spread cutoff.
- Raised the stationarity plateau gate to ignore under-sampled coarsest blocking points while keeping full blocking curves in artifacts.

## 2026-05-11

### Added

- Added RN-block DMC modules with shared DMC population, guide, result, and streaming owners.
- Added finite-`a` trapped RN-DMC workflows with stationarity gates, run manifests, checkpointing, and progress reporting.
- Added exact homogeneous and trapped validation entrypoints for RN-DMC release checks.
- Added weighted estimators, time-series diagnostics, chain diagnostics, and artifact/provenance helpers.
- Added cloud run scripts for reproducible VM campaigns.
- Added a canonical RN-block DMC method note and completed bibliography entries
  for DMC kernels, RN transition components, and stationarity diagnostics.

### Changed

- Reorganized experiments into validation, VMC, and DMC method folders.
- Moved DMC ownership from a flat module into method-specific and shared DMC packages.
- Reworked trapped stationarity checks to separate hygiene failures, correlated-data uncertainty, and stationarity failures.
- Updated documentation and equation source mapping for RN-block DMC, LDA comparison, and release gate semantics.

### Removed

- Removed tracked generated VMC result summaries from the source tree.
- Removed numbered top-level VMC experiment scripts in favor of domain folders.

## 2026-05-09

### Added

- Added trapped hard-rod local-energy diagnostics with kinetic, trap, and total energy components.
- Added finite-difference and invalid-configuration tests for the trapped local-energy estimator.
- Added total-energy alpha-scan reporting, including an energy-vs-alpha plot and bracketing diagnostics.
- Added a lower-alpha trapped VMC energy bracket packet for the current diagnostic trial.
- Added an upper-side trapped VMC energy alpha scan with longer chains to test whether the diagnostic minimum is bracketed.

### Changed

- Updated trapped alpha-scan summaries to keep `sampled_potential_energy_mean` as a labeled harmonic-trap-energy alias while using `sampled_total_energy_mean` for energy diagnostics.

## 2026-05-08

### Added

- Added normalization/reference checks for homogeneous `g(r)`, `S(k)`, and periodic density estimators.
- Added histogram-edge density integration, LDA boundary containment checks, and relative density L2 diagnostics.
- Added trapped cloud-radius estimators and an alpha-scan diagnostic for the trapped VMC trial.

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
