# RN-DMC Gate Audit

This note records the release gate semantics for trapped hard-rod RN-DMC runs.

## What Changed

The stationarity gate now separates three questions:

- hygiene: density accounting, finite local energies, valid retained samples,
  and RN weight control;
- correlated-data uncertainty: blocking plateau and conservative standard
  errors;
- stationarity: autocorrelation-adjusted slope, first/second-half drift,
  first/last-quarter drift, and four-block drift.

The spread statistic is still reported, but it is not a standalone veto. It is a
range-of-block-means diagnostic. It becomes a veto only when paired with a
trend, cumulative-drift, or first/last-block failure.

## New Artifacts

`experiments/dmc/rn_block/trapped_stationarity_grid.py` writes per-seed trace
artifacts when `--no-write` is not used:

- `seed_traces/<case>/<case>_seed<seed>_trace.csv`
- `seed_traces/<case>/<case>_seed<seed>_trace.npz`
- `seed_traces/<case>/blocking_<case>_seed<seed>_<metric>.csv`

Trace columns include energy, RMS radius, R2 radius, local-energy variance,
ESS fraction, log-weight span, proposal rejection fractions, retained fraction,
and RN log-density correction summaries.

## New Case Table Fields

The case table now includes:

- `old_case_gate`
- `hygiene_gate`
- `final_classification`
- `blocking_plateau_energy`
- `blocking_plateau_rms`
- `blocked_zscore_max_energy`
- `blocked_zscore_max_rms`
- `robust_zscore_max_energy`
- `robust_zscore_max_rms`
- `ess_fraction_min`
- `log_weight_span_max`
- `rn_weight_status`
- conservative standard errors for energy and RMS

## Classification Boundary

Allowed classifications are diagnostic labels, not paper claims:

- `PASS_CANDIDATE`
- `WEAK_TRAP_WARNING`
- `MIXED_OBSERVABLE_WARNING`
- `NO_GO_NO_BLOCKING_PLATEAU`
- `NO_GO_STATIONARITY`
- `SPREAD_VETO_NO_GO`
- `RN_WEIGHT_NO_GO`
- `HYGIENE_NO_GO`

No trapped finite-`a` result becomes a final benchmark unless timestep,
population, stationarity, hygiene, and estimator-bias boundaries are all
resolved. Energy is a mixed DMC energy. Density and RMS remain mixed observables
unless a pure estimator or forward-walking layer is added.

## Sweep Entry Point

Use the gate-audit sweep for a compact dt/population first pass:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 experiments/dmc/rn_block/gate_audit_sweep.py \
  --progress \
  --parallel-workers 5
```

Outputs:

- `results/dmc/rn_block/gate_audit_sweep/rn_dmc_sweep_summary.json`
- `results/dmc/rn_block/gate_audit_sweep/rn_dmc_sweep_table.csv`
