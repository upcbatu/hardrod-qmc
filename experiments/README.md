# Experiments

This directory contains thin command-line entrypoints only.

Reusable execution, method composition, statistics, estimators, and artifact
routing belong under `src/hrdmc/`.
Experiment scripts do not own physical formulas, guide formulas, estimator
reducers, Monte Carlo state evolution, or diagnostic-status rules.

## Layout

```text
experiments/anchors/          exact or analytic anchor entrypoints
experiments/dmc/rn_block/     RN-block DMC release entrypoints
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

Internal result schemas can contain machine-readable status strings.
Supervisor-facing notes translate them into plain language: accepted under the
current numerical checks, unresolved by the current diagnostics, or reserved
for systematic follow-up.

The public experiment surface contains release-facing anchors and RN-block
DMC entrypoints. Development probes, tuning scans, and historical signal runs
stay outside this surface.

Generated outputs use canonical routes under:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```
