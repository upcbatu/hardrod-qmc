# Experiments

This directory contains thin command-line entrypoints only.

Reusable execution, method composition, statistics, estimators, and artifact
routing belong under `src/hrdmc/`.

## Layout

```text
experiments/vmc/              VMC diagnostic entrypoints
experiments/validation/       exact or analytic validation entrypoints
experiments/dmc/rn_block/     RN-block DMC entrypoints
```

Generated outputs should use canonical routes under:

```text
results/<physics-layer>/<method-family>/<workflow-name>/
```
