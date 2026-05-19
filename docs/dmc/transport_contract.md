# RN Transport Event Contract

This document defines the public seam between the RN-block DMC engine and
transported auxiliary-variable forward-walking estimators. The engine
emits algorithmic transport events. Estimator formulas consume the events in
`estimators/`; experiments only orchestrate.

## Scope

The transport stream is for pure-coordinate estimator development. RN dynamics,
resampling, diagnostic status, local-energy evaluation, and the numba guide
backend stay owned by the DMC engine.

Raw descendant-count forward walking is a genealogy-collapse diagnostic. Paper
coordinate claims use the transported auxiliary estimator with its own
checks.

## Event Fields

Each `RNTransportEvent` contains:

```text
step_id
production_step_id
block_id
positions
local_energy_per_walker
r2_rb_per_walker
log_weights_pre_resample
log_weights_post_resample
parent_indices
resampled
weight_gauge_shift
convention
```

`local_energy_per_walker` is included for future energy control variates.
`positions` and `local_energy_per_walker` are the post-step population after
optional resampling for that DMC step. `log_weights_pre_resample` is the
gauge-shifted weight vector immediately before the optional resampling
operation. `log_weights_post_resample` defines the normalized estimator weights
for the emitted positions; it is zero after resampling and nonzero during
weighted no-resample windows.

`r2_rb_per_walker` is the optional engine-side COM Rao-Blackwell R2 payload.
When the analytic COM variance is unavailable, this field is `None`.
When present, the convention is:

```text
r_i = x_i - center - mean_j(x_j - center)
r2_rb = mean_i r_i^2 + Var(COM)
```

The estimator chooses raw R2 or RB R2 while preserving the COM convention.

## Branching And Weight Convention

The current transport convention is:

```text
weight_convention = post_step_normalized_log_weights
parent_convention = post_resample_parent_indices
gauge_convention  = log_weights_pre_resample_are_recentered_max_subtracted
```

The engine emits one event per DMC step when a transport observer is attached.
The parent map scope is one DMC step:

```text
snapshot_alignment = on_every_dmc_step
parent_map_scope   = single_dmc_step
```

If future artifacts store only fixed-step snapshots, the engine emits a
composed parent map from one stored snapshot to the next. Estimators do not
reconstruct intermediate resampling bookkeeping from hidden engine details.

Global log-weight gauge shifts cancel in normalized estimator averages and act
as bookkeeping gauges. Transported auxiliary variables are averaged with the
normalized `log_weights_post_resample` values at block completion.

Lag zero is the identity anchor: it is assembled as the block average of each
DMC step's instantaneous normalized weighted observable. Longer lags use the
transported auxiliary value and the final post-step normalized weights.

## Collection Mode

The current estimator contract reserves:

```text
collection_mode = single_point
```

For any requested lag greater than zero, `single_point` mode requires
`block_size_steps = 1`. Multi-step collection with lagged outputs is reserved
for `sliding_window`; otherwise early samples in the collection block would
have a longer forward length than later samples.

Future sliding-window collection can use the same output schema:

```text
collection_mode = sliding_window
```

## Required Invariants

Pure-estimator artifacts report the transport invariant checks used for that
run:

```text
transport_invariant_tests_passed = [
  "lag0_identity",
  "deterministic_parent_map",
  "weight_gauge_shift_cancellation",
  "composed_parent_map_associativity"
]
```

For paper-coordinate claims, the transported auxiliary estimator additionally
needs plateau, sufficient block count, sufficient walker-weight ESS,
density-accounting when density is requested, and population checks.

## R2/RMS Semantics

R2 is the primary transported observable. The paper RMS radius is:

```text
paper_rms_radius = sqrt(aggregated_pure_r2)
```

The mean of per-configuration square roots, if ever emitted, is named
`mean_instantaneous_rms`; the paper RMS radius remains
`sqrt(aggregated_pure_r2)`.

## Estimator Surface

The public estimator implementation is split by responsibility:

```text
src/hrdmc/estimators/pure/forward_walking/
  assembly.py
  config.py
  contributions.py
  diagnostics.py
  results.py
  transported.py
```

It supports transported auxiliary FW for:

```text
r2 / paper RMS
density                  # bin density; integral gives particle count
pair_distance_density    # bin density; integral gives pair count
structure_factor
```

Raw descendant counting remains a genealogy diagnostic only. Periodic-ring
`g(r)` normalization is not reused for trapped open-line pair distances unless
the geometry and normalization contract explicitly support it.
