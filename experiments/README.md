# Experiments

This directory contains thin command-line entrypoints only.

Reusable execution, method composition, statistics, estimators, and artifact
routing belong under `src/hrdmc/`.
Experiment scripts do not own physical formulas, guide formulas, estimator
reducers, Monte Carlo state evolution, or diagnostic-status rules.

## Layout

```text
experiments/anchors/          exact or analytic anchor entrypoints
experiments/dmc/local/        local DMC experiment entrypoints
```

Anchor scripts are thin public CLIs. Reusable anchor workflows, diagnostic
checks, comparisons, and artifact writers live under
`src/hrdmc/workflows/anchors/`.
This includes analytic theory diagnostics such as the hard-rod LDA cubic
inversion and small-\(a\) expansion packet.

Report-facing trapped density figures use enough histogram bins to
resolve finite-\(N\) shell peaks. The active public DMC and anchor CLIs
default to 800 density bins; exploratory commands can override
this when only scalar diagnostics are needed.

Result artifacts include machine-readable numerical statuses. Reports translate
them into plain language: consistent with the current checks, unresolved by the
current diagnostics, or requiring a stated follow-up calculation.

The tracked experiment commands contain exact anchors and local DMC runners.
Local DMC is the default; runners that expose collective RN moves require an
explicit option. Temporary probes and obsolete tuning scripts are not kept as
user entrypoints.

Generated outputs use canonical routes under:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```
