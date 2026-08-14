# Microscopic Description of Trapped Hard Rods

## What the code computes

This repository computes ground-state observables for strictly one-dimensional
hard rods in a harmonic trap. The production calculations use
importance-sampled diffusion Monte Carlo (DMC), with transported forward
walking for coordinate observables. Exact Tonks--Girardeau and two-body
solutions provide validation anchors, and an excluded-volume local-density
approximation (LDA) provides the analytic comparison. The reported matrix
covers particle numbers 10 and 20 at rod lengths 0, 0.1, 1, and 10 in
harmonic-oscillator units.

The outputs include mixed energies, pure cloud radii and density profiles,
DMC--LDA comparisons, and numerical allowances from time-step,
walker-population, and forward-walking checks. The manuscript contains the
derivations and physical interpretation. These notes document the
implementation, conventions, and reproduction procedure.

## Installation

Python 3.10 or newer is required. From a clone of the repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dmc]"
```

Install `.[dmc,dev]` instead when developing the repository.

## Reproducing the eight-case result

The compact summaries and final figures are tracked under `results/`; their
scope is described in [results/README.md](results/README.md). The following
command reassembles the eight thesis cases from the archived result bundle.
The full simulation bundle is not tracked, so a clean clone must first receive
the directory
`results/dmc/final_matrix/thesis_5seed_all_optimized_v1/` and its
`thesis_5seed_all_optimized_v1_supplement/N20_A10_r2_tau15` supplement.

```bash
OUT="$(mktemp -d "${TMPDIR:-/tmp}/hardrod-final-matrix.XXXXXX")"
PYTHONPATH=src python3 experiments/dmc/local/assemble_final_matrix.py \
  --source-root results/dmc/final_matrix/thesis_5seed_all_optimized_v1 \
  --output-root "$OUT" \
  --r2-supplement \
    N20_A10=results/dmc/final_matrix/thesis_5seed_all_optimized_v1_supplement/N20_A10_r2_tau15 \
  --retrospective-energy-cases N10_A0.1
```

On success, the command reports an accepted 8/8 matrix. See
[docs/reproducing.md](docs/reproducing.md) for the complete table, figure, and
Hellmann--Feynman reproduction chain and for every untracked prerequisite.

## Re-running the VMC validation

The tracked VMC packet checks two outstanding method objectives: consistency
of local and gradient kinetic-energy estimators, and agreement between an
independent random-walk Metropolis sampler and the production MALA transition
with branching disabled. The bounded validation cases are the exact
Tonks--Girardeau case `N10_A0` and the finite-diameter case `N10_A1`.

Both 512-walker packets are recorded under `results/vmc/validation/` with
status `accepted_with_warnings`. The warnings are non-blocking time-series
alerts; all required convergence, estimator-consistency, and sampler-
equivalence checks pass. The fixed sampler controls are tracked under
`data/vmc_sampler_choices/`. Exact rerun commands and the interpretation of
the packet are given in [docs/reproducing.md](docs/reproducing.md).

## Packages

The source layout has the root `hrdmc` namespace and eleven owner subpackages.
Command-line entry points are thin programs under `experiments/`.

| Package | Ownership |
|---|---|
| `hrdmc` | Empty root namespace; consumers import directly from owner modules. |
| `artifacts` | Artifact paths, schemas, manifests, progress, and terminal summaries. |
| `estimators` | Mixed, variational, Hellmann--Feynman, and forward-walking observables. |
| `plotting` | Reusable styles and diagnostic figure builders. |
| `production` | DMC benchmark, stationarity, and final-matrix packet production. |
| `sampling` | Initial conditions, MALA transitions, DMC/VMC engines, and population control. |
| `statistics` | Physics-independent diagnostics, equivalence tests, and numerical fits. |
| `system` | Hard-rod geometry, run settings, guide registry, and code units. |
| `theory` | Tonks--Girardeau, finite-diameter two-body, EOS, and LDA references. |
| `trial` | Trial/importance-sampling guides and their numerical kernels. |
| `uncertainty` | Time-step, population, stationarity, and forward-walking allowances. |
| `validation` | Exact anchors and branching-free sampler-equivalence checks. |

Package boundaries and dependency rules are documented in
[docs/architecture.md](docs/architecture.md). Code-unit conventions are in
[docs/units.md](docs/units.md).

## Development checks

After installing `.[dmc,dev]`:

- `make check` runs lint, type, structure, dead-code, import, test, public-surface,
  and whitespace checks.
- `make check-science` runs the fixed-seed DMC and published
  Hellmann--Feynman regression checks.
- `make report-duplicates` reports duplicate code without failing the build.
- `make security` runs the network-dependent dependency audit.
- `make clean` removes local caches and empty generated directories.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the software or its preferred thesis
citation. [CITATION.bib](CITATION.bib) records the papers whose methods are
implemented by the code; it is not the software citation.

## License

This project is distributed under the [MIT License](LICENSE).
