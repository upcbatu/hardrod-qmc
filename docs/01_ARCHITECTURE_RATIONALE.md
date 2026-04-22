# Architecture and Rationale

This repository is organized around a simple principle:

> the physical model, the trial state, the Monte Carlo methods, the observables, and the statistical analysis are kept separate, while experiment scripts are responsible for combining them into concrete runs.

The purpose of this separation is practical. The thesis is expected to evolve from an initial VMC implementation to a more complete DMC and forward-walking benchmark. A modular layout makes that transition easier to inspect, extend, and validate.

## 1. Why the repository is split into layers

The code is divided into five scientific roles.

### `systems/`

Purpose:
define the physical benchmark and its reference quantities.

This module owns:

- the one-dimensional hard-rod geometry;
- periodic boundary conditions on a ring;
- the hard-core exclusion rule;
- derived physical quantities such as density, packing fraction, and excluded length;
- reference hard-rod energies used for validation;
- optional external potentials for later extensions.

### `wavefunctions/`

Purpose:
define the trial state `Psi_T` used by the Monte Carlo layer.

This module owns:

- evaluate the trial amplitude or its logarithm;
- reject invalid configurations;
- expose tunable trial-state parameters;
- provide the trial forms used directly in VMC and as input to DMC.

### `monte_carlo/`

Purpose:
generate Monte Carlo data from the physical model and the trial state.

This module owns:

- `vmc.py` for the present Metropolis VMC implementation;
- `dmc.py` for the DMC interface together with the associated walker, branching, and pure-estimator support logic.

Its outputs are the sampled configurations and run metadata passed to the observable and analysis layers.

### `estimators/`

Purpose:
compute observables from sampled particle coordinates.

This module owns:

- pair distribution function `g(r)`;
- static structure factor `S(k)`;
- density profile `n(x)`.

The key design choice is that these estimators do not depend on whether the data came from VMC, mixed DMC, or forward-walking pure estimation.

### `analysis/`

Purpose:
turn raw estimator outputs into uncertainty, accuracy, and cost comparisons.

This module owns:

- estimator-family labels and constructions such as VMC, mixed, extrapolated, and pure;
- blocking-aware uncertainty estimates;
- bias and variance bookkeeping;
- mean-squared error;
- runtime-weighted cost metrics.

This is the layer that supports the central thesis question: which estimator family is preferable once both accuracy and computational cost are taken into account?

## 2. Why this structure is useful for the thesis

The thesis is not only about obtaining hard-rod observables. It is also about comparing estimator families:

- VMC;
- mixed DMC;
- extrapolated estimators;
- pure estimators via forward walking.

That comparison becomes clearer when the roles are separated:

- `systems/` defines the benchmark physics;
- `wavefunctions/` defines the trial state;
- `monte_carlo/` generates the Monte Carlo data;
- `estimators/` convert coordinates into observables;
- `analysis/` builds estimator families and compares them statistically.

This makes it possible to improve the DMC implementation later without rewriting the observable definitions, and to compare estimator families without mixing physical definitions with method-specific code.

## 3. Current implementation status

At present, the repository already contains:

- a hard-rod geometry implementation;
- a Jastrow-like VMC implementation;
- observable estimators for `g(r)`, `S(k)`, and `n(x)`;
- blocking and cost-analysis utilities;
- an initial integrated VMC experiment;
- an initial DMC structure for the thesis-level implementation.

The DMC production engine is not yet complete. The repository is designed so that a more mature DMC implementation can be added without changing the overall benchmark structure.

## 4. Why the `monte_carlo/` package is structured this way

The repository keeps the sampling algorithms in one place:

- `vmc.py` contains the Metropolis sampling workflow used for the current VMC runs;
- `dmc.py` contains the DMC-facing data structures and the support needed for walker evolution, branching, and pure-estimator work.

This keeps the method layer readable while leaving the observable and analysis layers unchanged as the DMC implementation matures.

## 5. Planned DMC integration

The final DMC implementation is expected to plug into `src/hrdmc/monte_carlo/dmc.py` and produce a result object containing:

- coordinate snapshots;
- local energies;
- weights;
- ancestry information for pure estimators;
- run metadata.

Once such a result object exists, the observable layer can remain unchanged:

- the same `g(r)` code can act on DMC coordinates;
- the same `S(k)` code can act on DMC coordinates;
- the estimator-family layer can label those results as mixed or pure;
- the extrapolated estimator can be formed later from compatible VMC and mixed DMC outputs.

This is the main reason the method and observable layers are kept separate.

## 6. Summary

The repository is structured to support a controlled progression:

```text
physical model -> trial state -> Monte Carlo method -> observable -> statistical comparison
```

We use the hard-rod system as a controlled benchmark, generate data with VMC and later DMC, compute observables consistently, and then compare estimator families in terms of bias, variance, and computational cost.
