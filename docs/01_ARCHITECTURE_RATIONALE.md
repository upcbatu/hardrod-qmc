# Architecture and Rationale

This repository is organized around a simple principle:

> physical systems, trial states, Monte Carlo sampling, coordinate observables, theory approximations, and comparison analysis are separate owners.

The revised thesis direction needs this boundary. The homogeneous hard-rod ring is a validation benchmark, the trapped hard-rod gas is the main sampled many-body target, the excluded-volume LDA is a theory-layer approximation, and the final thesis product is an analysis-layer accuracy and failure map.

## 1. Owner Boundaries

### `systems/`

Purpose:
define physical geometry and Hamiltonian ingredients.

This module owns only:

- ring geometry;
- open-line trapped geometry once added;
- hard-core constraints;
- boundary conventions;
- external potentials.

It does not own the homogeneous equation of state, chemical potential, excluded-volume LDA, or benchmark error logic.

### `wavefunctions/`

Purpose:
define trial states used by the Monte Carlo layer.

This module owns:

- trial amplitude or log-amplitude evaluation;
- rejection of invalid configurations;
- tunable trial-state parameters;
- trial forms used directly in VMC and as DMC input.

The current trial forms are ring-oriented. Trapped calculations may need additional one-body trap factors or supervisor-provided trial forms.

### `monte_carlo/`

Purpose:
generate sampled data and define result contracts.

This module owns:

- VMC engines;
- DMC engines when implemented;
- walker/result contracts;
- population-control support;

DMC is the target production method, but the architecture must not assume DMC is already a trusted reference. Until it is validated, trapped results should be labeled by benchmark tier: VMC diagnostic, DMC candidate reference, or external/group reference if available.

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
- excluded-volume mapping;
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

### `experiments/`

Purpose:
orchestrate concrete runs.

Experiments combine systems, wavefunctions, Monte Carlo, estimators, theory predictions, analysis summaries, IO, and plots. They should not own scientific equations.

### `plotting/`

Purpose:
generate figures only.

Plotting should render outputs from experiments and analysis without owning physics or comparison logic.

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
- an initial DMC result contract and support structures.

The trapped system, harmonic trap, trapped density estimator, benchmark-tier labeling, and production DMC engine are not yet complete.

## 4. Summary

The repository should evolve toward this progression:

```text
systems geometry -> theory predictions -> sampled observables -> analysis failure map
```

This keeps the excluded-volume LDA as a scientific approximation layer, not an analysis helper, and prevents `systems/` from accumulating EOS ownership.
