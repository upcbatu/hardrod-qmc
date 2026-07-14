# DMC Transport Event Contract

This document defines the public seam between the local DMC engine and
transported auxiliary forward-walking estimators. The engine emits algorithmic
transport events; estimator formulas consume them under `estimators/`; command
line experiments only orchestrate the workflow. The same event contract applies
when an optional scheduled collective RN move is enabled.

## Scope

The DMC engine owns state evolution, branching weights, resampling, and parent
indices. Forward walking owns auxiliary-variable transport, lag aggregation,
genealogy diagnostics, plateau checks, and coordinate-observable assembly.
Raw descendant counts are retained only as a genealogy diagnostic.

## Event Fields

Each `DMCTransportEvent` contains:

```text
step_id
production_step_id
scheduled_move_count
positions
local_energy_per_walker
log_weights_pre_resample
log_weights_post_resample
parent_indices
resampled
weight_gauge_shift
convention
```

`positions` and `local_energy_per_walker` describe the post-step population
after optional resampling. `production_step_id` is `None` during burn-in.
`scheduled_move_count` is metadata and does not change parent-map scope.

`log_weights_pre_resample` is the gauge-shifted vector immediately before the
optional resampling operation. `log_weights_post_resample` defines normalized
estimator weights for the emitted positions; it is zero after resampling and
can be nonzero during weighted no-resample windows.

The estimator can derive a center-of-mass Rao-Blackwellized \(R^2\) contribution
directly from `positions`:

```text
r_i = x_i - center - mean_j(x_j - center)
r2_rb = mean_i r_i^2 + Var(COM)
```

The trapped density estimator can use the same exact factorization: it
transports relative positions and convolves them with the known harmonic COM
ground-state density. This integrates out sampled COM jitter without changing
the DMC target.

## Branching and Parent Convention

The current convention is:

```text
weight_convention = post_step_normalized_log_weights
parent_convention = post_resample_parent_indices
gauge_convention  = log_weights_pre_resample_are_recentered_max_subtracted
snapshot_alignment = on_every_dmc_step
parent_map_scope    = single_dmc_step
```

The engine emits one event per DMC step when an observer is attached. The parent
map must therefore be consumed at every event. A scheduled collective move is
still followed by the same step-level branching/resampling event and does not
create a hidden parent-map boundary.

Global log-weight gauge shifts cancel in normalized averages. Lag zero is the
identity anchor: it is assembled from the instantaneous normalized observable
at that observable's collection cadence. R2 is collected every event. The
density direct reference and lag-zero stream share one profile evaluated only
on `density_collection_stride_steps` events. Longer lags transport the
auxiliary variable through the recorded parent indices and use the final
post-step weights.

## Collection Modes

The current single-point mode requires `block_size_steps = 1` whenever a
positive lag is requested. This gives every contribution the same forward
length. Sliding-window collection uses overlapping auxiliaries and the same
event contract.

## Required Invariants

Forward-walking artifacts record the transport checks used for the run:

```text
lag0_identity
deterministic_parent_map
weight_gauge_shift_cancellation
composed_parent_map_associativity
```

Reported lagged results additionally need finite block statistics, sufficient
walker-weight support, sufficient independent source-family support, and
density accounting when density is requested. Each density lag used in the
aggregate decision must integrate to the configured particle count within the
absolute accounting tolerance; a truncated grid therefore fails closed.

For a reported equal-weight average of independent seeds, the benchmark
workflow evaluates genealogy at that aggregate decision level. Unsupported
later lags remain diagnostics and do not veto an earlier contiguous lag window
that retains aggregate source-family support. Publication plateau selection
requires a prefix of at least three supported positive lags, starting at the
shortest configured positive lag; support cannot disappear and then re-enter
the decision window.

## R2 and RMS Semantics

\(R^2\) is the primary transported observable. The derived RMS radius is

```text
rms_radius = sqrt(aggregated_pure_r2)
```

Its serialized uncertainty is `rms_radius_stderr`. A mean of
per-configuration square roots, if emitted for diagnostics, is a different
quantity and must be named separately.

## Estimator Surface

The implementation lives under:

```text
src/hrdmc/estimators/pure/forward_walking/
```

It supports transported auxiliary forward walking for:

```text
r2 and rms_radius
density                  # integral gives particle count
pair_distance_density    # integral gives pair count
structure_factor
```

Periodic-ring `g(r)` normalization is not reused for trapped open-line pair
distances unless the geometry and normalization explicitly match.
