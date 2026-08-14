# Architecture

The library contains the root namespace and eleven owner subpackages. The root
`hrdmc` package re-exports nothing, so imports identify the module that owns an
operation. Programs under `experiments/` parse command-line arguments and call
the corresponding library modules.

| Package | Owns |
|---|---|
| `hrdmc` | The empty root namespace. |
| `artifacts` | Artifact layout, JSON schemas, manifests, progress, and bounded terminal output. |
| `estimators` | Observable construction from sampled states, including forward walking and energy response. |
| `plotting` | Reusable plotting style and diagnostic packet figures. |
| `production` | DMC calculations that write benchmark, stationarity, and matrix packets. |
| `sampling` | Markov transitions, DMC/VMC engines, population control, initial states, and seed batches. |
| `statistics` | Physics-independent time-series diagnostics, comparisons, and numerical fits; no artifact I/O. |
| `system` | Hamiltonian geometry and parameters, case settings, guide selection, and stored-unit metadata. |
| `theory` | Exact and analytic references: Tonks--Girardeau, finite-diameter two-body, EOS, and LDA. |
| `trial` | The trial/importance-sampling function and its evaluation kernels. |
| `uncertainty` | Assessment of recorded packets and emission of numerical allowances. |
| `validation` | Exact-anchor comparisons and independent-sampler equivalence checks. |

## Naming rule

A scientific or computational subject receives its own directory once it has
at least three cohesive modules or about one thousand lines. Within that
directory, filenames may describe roles such as `run.py`, `outputs.py`, or
`assessment.py`; the directory supplies the subject. At the root of a stage
package such as `sampling/`, `statistics/`, or `estimators/`, each filename
must instead name its subject. Repeated filename prefixes do not replace a
subject directory.

Source modules are limited to 600 lines, callables to 150 lines, and top-level
symbols to 25.

## Production and uncertainty

`production/` and `uncertainty/` meet only through finished artifacts:

| Side | Reads | Writes |
|---|---|---|
| `production/` | Case settings and sampling engines | DMC benchmark, stationarity, and matrix packets |
| `uncertainty/` | Manifest-bound finished packets | Time-step, population, stationarity, and forward-walking allowances |

Production performs the stochastic calculation and writes its packets.
Uncertainty reads those packets and assigns numerical allowances; it neither
runs DMC nor replaces central observables.
One narrow code dependency crosses this boundary:
`uncertainty/timestep/assessment.py` reuses
`production.matrix.assembly.load_final_matrix_energy_selection` to read and
verify a completed matrix assembly. The dependency is a finished-artifact
reader; it does not invoke production.

## Enforced layering

`make structure` checks the dependency restrictions defined in
`pyproject.toml`:

- `statistics` must not import `production`, `uncertainty`, `validation`, or
  `plotting`.
- `theory` must not import `production`, `uncertainty`, or `validation`.
- `system` must not import `production`, `uncertainty`, or `validation`.
- `trial` must not import `production`, `uncertainty`, or `validation`.

These restrictions prevent numerical primitives and physical definitions from
depending on orchestration or reporting code.
