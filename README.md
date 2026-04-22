# Hard-Rod DMC Thesis Scaffold

This repository is a computational-physics thesis scaffold for benchmarking Quantum Monte Carlo estimator families on the one-dimensional hard-rod Bose gas.

## Start here

- [docs/00_PROPOSAL.md](docs/00_PROPOSAL.md)
  Short thesis proposal.
- [docs/01_ARCHITECTURE_RATIONALE.md](docs/01_ARCHITECTURE_RATIONALE.md)
  Why the code is organized the way it is.
- [docs/02_HIGH_LEVEL_WORKFLOW.md](docs/02_HIGH_LEVEL_WORKFLOW.md)
  High-level workflow from benchmark setup to estimator comparison.
- [docs/03_EQUATION_SOURCE_MAP.md](docs/03_EQUATION_SOURCE_MAP.md)
  Which equations and method ideas come from which papers.
- [docs/04_CURRENT_REPO_STATE.md](docs/04_CURRENT_REPO_STATE.md)
  What is already implemented, what the tests cover, and what is still missing.
- [docs/05_TIMELINE_AND_NAVIGATION.md](docs/05_TIMELINE_AND_NAVIGATION.md)
  Tentative calendar and navigation note.

## Code layout

```text
src/hrdmc/systems/        physical model and benchmark formulas
src/hrdmc/wavefunctions/  trial states
src/hrdmc/monte_carlo/    VMC and DMC
src/hrdmc/estimators/     observables from coordinate data
src/hrdmc/analysis/       uncertainty and cost metrics
src/hrdmc/io/             JSON / NPZ outputs
src/hrdmc/plotting/       figures
experiments/              runnable entrypoints
```

## Current status

- hard-rod geometry is implemented
- a working VMC smoke pipeline exists
- observable estimators for `g(r)`, `S(k)`, and `n(x)` exist
- DMC is scaffolded for the thesis-level implementation
