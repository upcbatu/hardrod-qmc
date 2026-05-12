# Experiments

This directory contains thin command-line entrypoints only.

Reusable execution, method composition, statistics, estimators, and artifact
routing belong under `src/hrdmc/`.

## Layout

```text
experiments/validation/       exact or analytic validation entrypoints
experiments/dmc/rn_block/     RN-block DMC release entrypoints
```

Development probes, tuning scans, and historical signal runs are intentionally
not part of the public experiment surface.

Generated outputs should use canonical routes under:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```
