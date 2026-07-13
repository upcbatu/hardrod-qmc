# Architecture and Rationale

This repository is organized around a simple principle:

> physical systems, trial states, Monte Carlo sampling, coordinate observables, theory approximations, comparison analysis, artifact IO, and plotting are separate owners.

The revised thesis direction needs this boundary. The homogeneous hard-rod ring is a validation benchmark, the trapped hard-rod gas is the main sampled many-body target, the excluded-volume LDA is a theory-layer approximation, and the final thesis product is an analysis-layer accuracy and failure map.

## 1. Owner Boundaries

### `systems/`

Purpose:
define physical geometry and Hamiltonian ingredients.

This module owns only:

- ring geometry;
- open-line trapped geometry;
- hard-core constraints;
- boundary conventions;
- external potentials;
- reduced-coordinate geometry identities such as `L' = L - N a`.
- DMC target propagator interfaces tied to a system Hamiltonian.
- system-owned target/proposal transition densities and samplers;
- system-side hot array kernels under `systems/kernels/`.

It does not own the homogeneous equation of state, chemical potential,
excluded-volume LDA, trial/guide amplitudes, estimator reducers, benchmark
error logic, or wavefunction log-amplitude mathematics. `systems/` owns
Green-kernel and coordinate-map mathematics tied to the physical system.

### `wavefunctions/`

Purpose:
define trial states and guides used by the Monte Carlo layer.

This module owns:

- diagnostic trial amplitude or log-amplitude evaluation under `trials/`;
- DMC guide classes and guide parameters under `guides/`;
- guide-owned hot array kernels under `kernels/`;
- DMC guide derivatives when a guide is used for importance sampling;
- local-energy ingredients derived from the guide;
- rejection of invalid configurations;
- tunable trial-state parameters;
- trial forms used directly in VMC and as DMC input.

`wavefunctions/api.py` defines the guide interface consumed by DMC engines.
Engines depend on the protocol, not on concrete guide classes or kernel
modules.

Proposal densities, optional collective-move correction weights, population
control, coordinate observable reducers, and numerical-check logic are owned by
their respective DMC, estimator, and analysis layers.

For production DMC, a guide controls the importance-sampling drift,
local-energy variance, and practical projection quality. The DMC engine consumes
a guide protocol with:

```text
log_value
grad_log_value
lap_log_value
local_energy
validity
```

### `monte_carlo/`

Purpose:
generate sampled data and define result contracts.

This module owns:

- VMC engines;
- DMC engines when implemented;
- walker/result contracts;
- population-control support;

DMC is the target production method, but numerical interpretation remains
separate from sample generation. Trapped results record the method, estimator,
and validation status needed to interpret each observable.

The generic DMC contract remains a shared support layer inside the `dmc/`
package. The default importance-sampled engine lives at:

```text
src/hrdmc/monte_carlo/dmc/local/
```

It owns local drift-diffusion, branching, population control, checkpoints,
streaming summaries, and DMC transport events. Optional collective proposals
and their change-of-measure correction live separately at:

```text
src/hrdmc/monte_carlo/dmc/collective_rn/
```

The system/Hamiltonian layer owns the physical system and any target short-time
transition density:

```text
K_sys(x_new | x_old, tau)
```

The guide owns trial-wavefunction derivatives and local energy. The optional
collective extension owns its proposal:

```text
Q_theta(x_new | x_old)
```

and the ratio bookkeeping:

```text
log K_sys - log Q_theta
```

This keeps optional collective transport from duplicating Hamiltonian physics.

Monte Carlo engines call guide and transition interfaces only. Optional numba
availability and guide/proposal formulas remain outside the engine logic.

### `numerics/`

Purpose:
centralize optional numerical backend plumbing.

This module owns:

- optional numba availability detection;
- `njit` fallback helpers;
- backend label helpers for artifacts.

It does not own physics formulas, kernels, sampling, estimators, or diagnostic
status rules.
Formula owners keep hot array kernels under their own packages, such as
`systems/kernels/` or `wavefunctions/kernels/`, and import backend helpers from
`numerics/`.

### `estimators/`

Purpose:
compute observables from sampled coordinates.

This module owns:

- pair distribution function `g(r)`;
- static structure factor `S(k)`;
- density profile `n(x)`;
- future trapped observables that are direct functions of sampled coordinates.

LDA solving belongs to `theory/`. Trapped density estimation consumes sampled
coordinates on an open-line grid without periodic wrapping.

### `theory/`

Purpose:
own analytic and semi-analytic approximations.

This module owns:

- homogeneous hard-rod EOS;
- chemical potential;
- excluded-volume EOS and LDA formulas that use the reduced length supplied by
  `systems/`;
- chemical-potential inversion;
- LDA normalization;
- LDA density and energy predictions.

The theory layer produces reference predictions. Comparisons against sampled
data belong to `analysis/`.

### `analysis/`

Purpose:
compare sampled observables against references.

This module owns:

- QMC/DMC versus LDA errors;
- bulk-versus-edge diagnostics;
- finite-`N` trends;
- uncertainty summaries;
- failure maps;
- compact run summaries.

LDA normalization and EOS formulas remain in `theory/`.

### `io/`

Purpose:
serialize and load run artifacts.

This module owns:

- JSON and NPZ artifact handling;
- lightweight schema conversion;
- result metadata serialization;
- file-format boundaries between experiments, analysis, and plotting.

It does not own physics formulas, sampling behavior, comparison metrics, or
figure layout.

### `runners/`

Purpose:
execute independent run batches.

This module owns seed-batch dispatch, bounded parallel-worker integration,
worker-to-parent progress propagation, and serial fallback when multiprocessing
is unavailable.

It does not own physics systems, DMC algorithms, observables, or scientific
validation decisions.

### `workflows/`

Purpose:
compose package engines into reproducible scientific workflows.

Workflow modules sit above engines and below command-line experiments. They own
case parsing, default controls, system-guide-kernel composition, and
method-specific summaries.

They do not own primitive DMC transition kernels, low-level estimators, or CLI
argument parsing.

### `artifacts/`

Purpose:
define canonical result routes.

Artifact layout belongs here so scripts use canonical result folders.
The tracked convention is:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```

### `experiments/`

Purpose:
orchestrate concrete runs through thin CLI entrypoints.

Experiments are grouped by domain:

```text
experiments/anchors/
experiments/dmc/local/
```

This directory contains thin anchor and local DMC entrypoints. Local DMC is the
default; collective RN transport is enabled only by an explicit runner option.
Temporary probes and obsolete tuning scripts are not kept as user commands.

Reusable runner engines, method workflows, statistics logic, and scientific
equations belong to the package owners above.

### `plotting/`

Purpose:
generate figures only.

Plotting renders outputs from experiments and analysis without owning physics
or comparison logic.

### `data/`

Purpose:
hold external or generated input data.

Raw/reference data stay separate from generated results. Large data stays out
of git unless it is tiny and required for tests.

### `results/`

Purpose:
hold generated experiment outputs.

Experiment outputs include metadata with parameters, seeds, sample counts, and
runtime. Source ownership remains in the package layers above.

### `notebooks/`

Purpose:
support inspection and figure drafting.

Notebook logic moves into `src/` or `experiments/` before it becomes part of
the thesis workflow.

## 2. Main Thesis Flow

```text
ring validation
  -> validates geometry, trial state, MC, energy references

trapped system
  -> main sampled many-body target

excluded-volume LDA
  -> theory-layer approximation built from homogeneous EOS

QMC/DMC vs LDA
  -> analysis-layer benchmark and failure map

Lieb-Liniger probe
  -> optional isolated extension outside the main trapped-hard-rod path
```

The operational flow is:

```text
homogeneous validation -> trapped benchmark -> theory/LDA prediction -> analysis/failure map
```

## 3. Current Implementation Status

At present, the repository contains:

- homogeneous ring hard-rod geometry in `systems/`;
- homogeneous EOS, finite ring energy, chemical-potential inversion, and LDA support in `theory/`;
- a Jastrow-like VMC implementation;
- observable estimators for `g(r)`, `S(k)`, and ring-based `n(x)`;
- blocking and support metric utilities;
- an initial homogeneous VMC diagnostic experiment;
- a local importance-sampled DMC engine plus an optional collective RN
  scheduled-move extension.

The trapped system also has a VMC diagnostic path. Systematic failure-map runs
and the remaining high-\(A\) numerical validation are ongoing.

Repository material includes compact examples, validation fixtures, historical
summary tables, and one DMC method document. Exploratory notebooks, progress
logs, and full raw run bundles remain outside the package API.

## 4. Summary

The repository evolves toward this progression:

```text
systems geometry -> theory predictions -> sampled observables -> analysis failure map
```

This keeps the excluded-volume LDA as a scientific approximation layer,
separate from analysis helpers, and keeps EOS ownership out of `systems/`.
