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

It does not own the homogeneous equation of state, chemical potential,
excluded-volume LDA, or benchmark error logic.

### `wavefunctions/`

Purpose:
define trial states and guides used by the Monte Carlo layer.

This module owns:

- trial amplitude or log-amplitude evaluation;
- DMC guide derivatives when a trial state is used for importance sampling;
- local-energy ingredients derived from the guide;
- rejection of invalid configurations;
- tunable trial-state parameters;
- trial forms used directly in VMC and as DMC input.

The current trial forms are ring-oriented. Trapped calculations may need additional one-body trap factors or supervisor-provided trial forms.

For production DMC, a "wavefunction" is not just an initializer. It controls
the importance-sampling drift, local-energy variance, and practical projection
quality. The RN-DMC release path should therefore expose a guide protocol
with:

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

DMC is the target production method, but the architecture must not assume DMC is already a trusted reference. Until it is validated, trapped results should be labeled by benchmark tier: VMC diagnostic, DMC candidate, or external/group reference if available.

The generic DMC contract should remain a shared support layer inside the
`dmc/` package. Specific DMC engines should live in subpackages. The
RN-corrected collective-block engine candidate is therefore promoted as:

```text
src/hrdmc/monte_carlo/dmc/rn_block/
```

and not by turning the DMC contract layer into a monolith.

The RN-block subpackage must not own the physical system. Its config should own
algorithmic proposal parameters only. The system/Hamiltonian layer should own
the target short-time transition density:

```text
K_sys(x_new | x_old, tau)
```

The guide should own trial-wavefunction derivatives and local energy. RN-block
should own only the collective proposal:

```text
Q_theta(x_new | x_old)
```

and the ratio bookkeeping:

```text
log K_sys - log Q_theta
```

This prevents the collective block code from silently duplicating system
physics and playing a separate dynamics game.

### `estimators/`

Purpose:
compute observables from sampled coordinates.

This module owns:

- pair distribution function `g(r)`;
- static structure factor `S(k)`;
- density profile `n(x)`;
- future trapped observables that are direct functions of sampled coordinates.

It should not know how LDA is solved. Trapped density estimation must also avoid periodic wrapping.

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

The theory layer produces reference predictions. It does not compare them against sampled data.

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

It should not solve the LDA normalization or own EOS formulas.

### `io/`

Purpose:
serialize and load run artifacts.

This module owns:

- JSON and NPZ artifact handling;
- lightweight schema conversion;
- result metadata serialization;
- file-format boundaries between experiments, analysis, and plotting.

It should not own physics formulas, sampling behavior, comparison metrics, or figure layout.

### `runners/`

Purpose:
execute independent run batches.

This module owns seed-batch dispatch, bounded parallel-worker integration,
worker-to-parent progress propagation, and serial fallback when multiprocessing
is unavailable.

It should not own physics systems, DMC algorithms, observables, or paper
classification rules.

### `workflows/`

Purpose:
compose package engines into reproducible scientific workflows.

Workflow modules sit above engines and below command-line experiments. They own
case parsing, default controls, system-guide-kernel composition, and
method-specific summaries.

They should not own primitive DMC transition kernels, low-level estimators, or
CLI argument parsing.

### `artifacts/`

Purpose:
define canonical result routes.

Artifact layout belongs here so scripts do not invent ad hoc result folders.
The tracked convention is:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```

### `experiments/`

Purpose:
orchestrate concrete runs through thin CLI entrypoints.

Experiments are grouped by domain:

```text
experiments/validation/
experiments/dmc/rn_block/
```

Only release-facing validation and RN-block DMC entrypoints belong here. Private
diagnostic probes, tuning scans, and abandoned signal scripts are intentionally
kept out of the public experiment surface.

They should not contain reusable runner engines, method workflows, statistics
logic, or scientific equations.

### `plotting/`

Purpose:
generate figures only.

Plotting should render outputs from experiments and analysis without owning physics or comparison logic.

### `tests/`

Purpose:
verify implemented behavior.

Tests should cover owner contracts, numerical formulas, shape conventions, and regression behavior. Tests should not become hidden experiments.

### `data/`

Purpose:
hold external or generated input data.

Raw/reference data should stay separate from generated results. Large data should remain out of git unless it is tiny and required for tests.

### `results/`

Purpose:
hold generated experiment outputs.

Experiment outputs should include metadata with parameters, seeds, sample counts, and runtime. Results are artifacts, not source ownership.

### `notebooks/`

Purpose:
support inspection and figure drafting.

Notebook logic should be promoted into `src/` or `experiments/` before it becomes part of the thesis workflow.

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
  -> optional isolated extension, not part of the main trapped-hard-rod path
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
- an initial homogeneous VMC smoke experiment;
- a DMC contract package plus an RN-block candidate implementation.

The trapped system now has an initial VMC diagnostic path. Benchmark-tier
expansion, systematic failure-map workflows, and release-grade RN-DMC examples
are not yet complete.

Release material should include compact examples, validation fixtures, archived
run artifacts, and one canonical method document. Exploratory notebooks,
progress logs, and full raw run bundles should not be part of the package API.

## 4. Summary

The repository should evolve toward this progression:

```text
systems geometry -> theory predictions -> sampled observables -> analysis failure map
```

This keeps the excluded-volume LDA as a scientific approximation layer, not an analysis helper, and prevents `systems/` from accumulating EOS ownership.
