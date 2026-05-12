# RN Transport Event Contract

This document defines the public seam between the RN-block DMC engine and
transported auxiliary-variable forward-walking estimators. The engine
emits algorithmic transport events. Estimator formulas consume the events in
`estimators/`; experiments only orchestrate.

## Scope

The transport stream is for pure-coordinate estimator development. It must not
change RN dynamics, resampling, gates, local-energy evaluation, or the numba
guide backend.

Raw descendant-count forward walking is diagnostic only. It can report
genealogy collapse, but it cannot authorize paper coordinate claims.

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
When the engine is not given the analytic COM variance, this field is `None`.
When present, the convention is:

```text
r_i = x_i - center - mean_j(x_j - center)
r2_rb = mean_i r_i^2 + Var(COM)
```

The estimator chooses raw R2 or RB R2; it does not redefine the COM convention.

## Branching And Weight Convention

The current transport convention is:

```text
weight_convention = post_step_normalized_log_weights
parent_convention = post_resample_parent_indices
gauge_convention  = log_weights_pre_resample_are_recentered_max_subtracted
```

The engine emits one event per DMC step when a transport observer is attached.
The parent map scope is therefore one DMC step:

```text
snapshot_alignment = on_every_dmc_step
parent_map_scope   = single_dmc_step
```

If future artifacts store only fixed-step snapshots, the engine must emit a
composed parent map from one stored snapshot to the next. Estimators must not
reconstruct intermediate resampling bookkeeping from hidden engine details.

Global log-weight gauge shifts are not physical factors. They cancel in
normalized estimator averages and must not be multiplied into transported
auxiliary variables. At block completion, transported auxiliary values are
averaged with normalized `log_weights_post_resample`.

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

Pure-estimator artifacts must report the transport invariant checks used for
that run:

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

The mean of per-configuration square roots, if ever emitted, must be named
`mean_instantaneous_rms` and must not be used as the paper RMS radius.

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
