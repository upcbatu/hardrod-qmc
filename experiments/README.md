# Experiments

This directory contains thin command-line entrypoints only.

Reusable execution, method composition, statistics, estimators, and artifact
routing belong under `src/hrdmc/`.
Experiment scripts must not own physical formulas, guide formulas, estimator
reducers, Monte Carlo state evolution, or gate rules.

## Layout

```text
experiments/anchors/          exact or analytic anchor entrypoints
experiments/dmc/rn_block/     RN-block DMC release entrypoints
```

Anchor scripts are thin public CLIs. Reusable anchor workflows, gates,
comparisons, and artifact writers live under `src/hrdmc/workflows/anchors/`.

Development probes, tuning scans, and historical signal runs are intentionally
not part of the public experiment surface.

Generated outputs should use canonical routes under:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```
