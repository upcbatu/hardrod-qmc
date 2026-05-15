# Trapped Hard-Rod QMC Thesis Codebase

This repository is a computational-physics thesis codebase for trapped one-dimensional hard-rod bosons.

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
- [docs/dmc/method.md](docs/dmc/method.md)
  Canonical RN-DMC method, gate, and claim-boundary note.
- [docs/validation/README.md](docs/validation/README.md)
  Validation notes for benchmark status, interpretation, and remaining checks.

## Code layout

```text
src/hrdmc/systems/        physical geometry, constraints, potentials, kernels
src/hrdmc/wavefunctions/  VMC trials and DMC guides
src/hrdmc/monte_carlo/    VMC plus DMC contracts and implementations
src/hrdmc/estimators/     observables from coordinate data
src/hrdmc/theory/         homogeneous EOS, chemical potential, LDA
src/hrdmc/analysis/       errors, uncertainty, and failure maps
src/hrdmc/runners/        generic seed-batch execution and progress plumbing
src/hrdmc/workflows/      method workflow composition above engines
src/hrdmc/artifacts/      canonical result routing
src/hrdmc/io/             JSON / NPZ outputs
src/hrdmc/plotting/       figures
experiments/anchors/      exact and analytic anchor entrypoints
experiments/dmc/rn_block/ RN-block DMC release entrypoints
tests/                    regression tests
data/                     external/reference inputs, usually untracked
results/                  generated experiment outputs, usually untracked
notebooks/                inspection and figure drafting
```

## Development Checks

Install development tooling with `python3 -m pip install -e ".[dev]"`.

- `make lint` runs `ruff check`.
- `make typecheck` runs `pyright` with the repository `pyproject.toml` settings.
- `make check` runs both lint and type checks.
- `make test` runs lint and unit tests.

## Current status

- homogeneous periodic hard-rod geometry is implemented
- exact homogeneous ring reference energies and hard-rod EOS utilities are implemented in `theory/`
- the anchor validation entrypoints compare exact/analytic references against
  the public engine and estimator surfaces through thin CLIs
- VMC remains available in the package as scaffold code, but development VMC
  experiment scripts are no longer part of the public experiment surface
- observable estimators for local energy, `g(r)`, `S(k)`, and ring-based `n(x)` exist
- the DMC layer now has a generic contract package and an RN-block
  candidate implementation under `src/hrdmc/monte_carlo/dmc/rn_block/`
- the active trapped-system progress report uses the RN-block workflow under
  `experiments/dmc/rn_block/` plus transported forward walking under
  `src/hrdmc/estimators/pure/forward_walking/`; older bootstrap DMC notes
  should not be read as the active status for that report
- compact validation tables, gated RN runners, and release-quality
  documentation are now present; release metadata and archived result bundles
  are still pending
