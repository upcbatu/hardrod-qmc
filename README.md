# Trapped Hard-Rod QMC Thesis Scaffold

This repository is a computational-physics thesis scaffold for trapped one-dimensional hard-rod bosons.

The homogeneous hard-rod system on a ring is kept as a controlled QMC validation benchmark, because its excluded-volume mapping gives known reference energies and wavefunction structure. The main thesis target is the trapped system: compute benchmark observables with QMC, with DMC as the target production method and VMC as a baseline, then map where an excluded-volume local-density approximation succeeds or fails.

## Start here

- [docs/00_PROPOSAL.md](docs/00_PROPOSAL.md)
  Short thesis proposal.
- [docs/01_ARCHITECTURE_RATIONALE.md](docs/01_ARCHITECTURE_RATIONALE.md)
  Why the code is organized the way it is.
- [docs/02_HIGH_LEVEL_WORKFLOW.md](docs/02_HIGH_LEVEL_WORKFLOW.md)
  High-level workflow from homogeneous validation to trapped-system comparison.
- [docs/03_EQUATION_SOURCE_MAP.md](docs/03_EQUATION_SOURCE_MAP.md)
  Which equations and method ideas come from which papers.
- [docs/04_CURRENT_REPO_STATE.md](docs/04_CURRENT_REPO_STATE.md)
  What is already implemented, what the tests cover, and what is still missing.
- [docs/05_TIMELINE_AND_NAVIGATION.md](docs/05_TIMELINE_AND_NAVIGATION.md)
  Tentative calendar and navigation note.

## Code layout

```text
src/hrdmc/systems/        physical geometry, constraints, and potentials
src/hrdmc/wavefunctions/  trial states
src/hrdmc/monte_carlo/    VMC and DMC
src/hrdmc/estimators/     observables from coordinate data
src/hrdmc/theory/         homogeneous EOS, excluded-volume mapping, LDA
src/hrdmc/analysis/       errors, uncertainty, and failure maps
src/hrdmc/io/             JSON / NPZ outputs
src/hrdmc/plotting/       figures
experiments/              runnable entrypoints
tests/                    regression tests
data/                     external/reference inputs, usually untracked
results/                  generated experiment outputs, usually untracked
notebooks/                inspection and figure drafting
```

## Current status

- homogeneous periodic hard-rod geometry is implemented
- exact homogeneous ring reference energies and hard-rod EOS utilities are implemented in `theory/`
- a working VMC smoke pipeline exists for the homogeneous scaffold
- observable estimators for `g(r)`, `S(k)`, and ring-based `n(x)` exist
- DMC is scaffolded, not production-ready
- trapped open-line geometry, harmonic trapping, and benchmark failure-map workflows are the next implementation targets
