# Architecture and Rationale

This repository is organized around a simple principle:

> physical definitions, trial states, Monte Carlo sampling, observables, and analysis references are kept separate, while experiment scripts combine them into concrete runs.

The revised thesis direction makes this separation more important. The homogeneous hard-rod ring is a validation benchmark, while the trapped hard-rod gas is the main physics target. The same codebase must support both without mixing periodic-ring assumptions into trapped observables.

## 1. Why the Repository Is Split Into Layers

### `systems/`

Purpose:
define physical systems and reference formulas.

This module currently owns:

- homogeneous one-dimensional hard rods on a periodic ring;
- periodic boundary conditions;
- hard-core exclusion;
- derived quantities such as density, packing fraction, and excluded length;
- exact homogeneous reference energies.

It still needs to own:

- open-line trapped hard rods;
- harmonic trap potential support;
- trapped initial configurations;
- homogeneous equation-of-state helpers for LDA.

### `wavefunctions/`

Purpose:
define the trial state `Psi_T` used by the Monte Carlo layer.

This module owns:

- trial amplitude or log-amplitude evaluation;
- rejection of invalid configurations;
- tunable trial-state parameters;
- trial forms used directly in VMC and as DMC input.

The current trial forms are ring-oriented. Trapped calculations may need additional one-body trap factors or supervisor-provided trial forms.

### `monte_carlo/`

Purpose:
generate coordinate and energy data from a physical system and a trial state.

This module owns:

- `vmc.py` for the current Metropolis VMC implementation;
- `dmc.py` for the DMC result contract, walker data structures, population-control support, and optional forward-walking support.

For the revised thesis, VMC is a smoke and diagnostic path. DMC is the intended production path for trapped ground-state observables.

### `estimators/`

Purpose:
compute observables from sampled coordinates.

This module currently owns:

- pair distribution function `g(r)`;
- static structure factor `S(k)`;
- periodic density profile `n(x)`.

It still needs to separate periodic and open-line density conventions. Trapped density estimation must not wrap coordinates onto a ring.

### `analysis/`

Purpose:
turn raw observables into scientific comparisons.

This module currently owns:

- blocking-aware uncertainty estimates;
- bias and variance bookkeeping;
- mean-squared error utilities;
- estimator-family labels and combinations.

It still needs to own or host:

- homogeneous hard-rod chemical potential;
- chemical-potential inversion;
- trapped LDA normalization;
- QMC benchmark versus LDA comparison summaries.

The estimator-family code remains useful support infrastructure, but it no longer defines the thesis endpoint.

## 2. Why This Structure Fits the Revised Thesis

The thesis has two different physical roles:

- homogeneous hard rods on a ring validate the numerical machinery;
- trapped hard rods provide the main benchmark of excluded-volume LDA accuracy and failures.

Separating systems, estimators, and analysis prevents a ring-specific assumption from leaking into the trapped workflow. For example, `S(k)` on a ring and `n(x)` in a trap have different coordinate conventions, even if both are computed from sampled coordinates.

The LDA comparison also benefits from separation:

- `systems/` provides the homogeneous equation of state;
- `analysis/` evaluates the LDA normalization and comparison metrics;
- `estimators/` computes benchmark observables;
- `experiments/` assembles parameter sweeps.

## 3. Current Implementation Status

At present, the repository contains:

- homogeneous ring hard-rod geometry;
- exact homogeneous energy references;
- a Jastrow-like VMC implementation;
- observable estimators for `g(r)`, `S(k)`, and ring-based `n(x)`;
- blocking and support metric utilities;
- an initial homogeneous VMC smoke experiment;
- an initial DMC result contract and support structures.

The trapped system, harmonic trap, LDA implementation, and production DMC engine are not yet complete.

## 4. Planned DMC Integration

The final DMC implementation is expected to plug into `src/hrdmc/monte_carlo/dmc.py` and produce a result object containing:

- coordinate snapshots;
- local energies;
- weights;
- run metadata;
- ancestry information only when pure-estimator support is needed.

Once such a result exists, the trapped observable layer should compute density profiles, energies, and cloud-size diagnostics from DMC data without depending on DMC internals.

## 5. Summary

The repository should evolve toward this progression:

```text
homogeneous validation -> trapped system -> benchmark observables -> excluded-volume LDA -> failure map
```

Estimator-family and cost-analysis utilities remain available, but the main thesis comparison is trapped benchmark observables against the excluded-volume LDA reference.
