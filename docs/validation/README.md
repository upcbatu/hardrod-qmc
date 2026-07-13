# Validation Notes

This note records the validation status of the numerical workflow. It is written as a thesis-facing summary: what was tested, what reference was used, what the result means, and what remains to be validated before using the code for trapped-system conclusions.

## Current Trapped DMC Status

The homogeneous ring checks below remain the baseline validation layer. The
active trapped-system workflow uses local DMC under `experiments/dmc/local/`,
the exact anchor entrypoints under `experiments/anchors/`, and the transported
forward-walking estimator under
`src/hrdmc/estimators/pure/forward_walking/`. Collective RN moves are optional
and disabled by default.

Each row records stationarity, population, density-accounting, timestep, and
observable-specific estimator evidence. Optional collective-RN runs additionally
record their change-of-measure weight diagnostics.

## 1. Purpose

The thesis uses the homogeneous hard-rod gas on a ring as a controlled benchmark before moving to trapped one-dimensional hard rods. The homogeneous hard-rod problem has an exact excluded-volume mapping, which provides a clean validation point for the numerical pipeline.

The homogeneous validation layer checks the ring pipeline:

```text
periodic hard-rod geometry
  -> exact all-pair hard-rod trial wavefunction
  -> Metropolis VMC sampling
  -> local-energy estimator
  -> exact finite-N ring energy
```

Historical trapped VMC diagnostics were used during development. They are not
kept as current user commands.

## 2. Homogeneous Ring Benchmark

### Reference Result

For `N` hard rods of length `a` on a ring of length `L`, the excluded-volume mapping gives the reduced length

```text
L' = L - N a.
```

The finite-`N` reference energy per particle is

```text
E_N / N = (1 / 2N) sum_i (2 pi I_i / L')^2,
```

with energy measured in the homogeneous scale `hbar^2/(m length^2)`. This finite-size expression is the pass/fail reference for the present benchmark.

The thermodynamic equation of state,

```text
e_HR(rho) = pi^2 rho^2 / [6 (1 - a rho)^2],
```

is printed in the run output as contextual information. The finite-`N`
validation target is the reduced-length energy formula above.

### Numerical Test

Command:

```bash
PYTHONPATH=src python3 experiments/anchors/homogeneous_ring.py
```

Output artifact:

```text
results/homogeneous_validation/summary.json
```

The benchmark currently runs a small grid:

| Parameter | Values |
| --- | --- |
| `N` | 4, 8, 16 |
| `a rho` | 0.10, 0.30, 0.50 |
| `a` | 0.5 |

For each point, the code samples configurations with the all-pair hard-rod trial wavefunction and evaluates the local kinetic energy. This trial is the exact homogeneous-ring benchmark form, so the local energy is constant and equal to the finite-`N` reference energy.

### Result

The benchmark passed for all 9 cases.

| Quantity | Result |
| --- | ---: |
| cases passed | 9 / 9 |
| maximum absolute error in `E/N` | 1.7763568394002505e-15 |
| valid snapshot fraction | 1.0 in every case |
| acceptance-rate range | approximately 0.83 to 0.87 |

The energy error is at floating-point roundoff level. This indicates that the periodic hard-rod geometry, the exact all-pair trial wavefunction, and the local-energy formula are mutually consistent for the homogeneous benchmark. Sampler-convergence checks are handled separately, because the exact all-pair local energy is constant for every valid sampled configuration.

## 3. Interpretation

This validation establishes a controlled starting point for the code. It supports the use of the homogeneous ring as a benchmark for checking basic geometry, exclusion constraints, sampling, and energy-estimator consistency.

Trapped-system validation is handled by the trapped TG anchor, the finite-\(a\)
\(N=2\) anchor, and the DMC stationarity/forward-walking checks listed
below. The trapped workflow also requires propagation, timestep, population,
and estimator checks before it is used as the production reference.

## 4. Remaining Validation Steps

Before thesis-level trapped-system conclusions, the following checks remain.

### Homogeneous Observables

- Initial normalization checks now cover the periodic density integral, the `g(r)` finite-`N` unique-pair sum rule, and a lattice reference for `S(k)`.
- Later comparison target: homogeneous `g(r)` and `S(k)` shapes against literature curves or trusted reference data where available.

### Trapped Geometry and Sampling

- Open-line hard-rod geometry without periodic wrapping has an initial implementation.
- The harmonic trapping potential has an initial implementation.
- Trapped initial configurations are checked against the hard-core constraint.
- Development-only trapped VMC diagnostic/grid/seed/alpha scans are intentionally not
  part of the public experiment surface.

### Trapped Density and LDA

- A non-periodic trapped density estimator has an initial implementation.
- LDA normalization is checked on the same spatial grid used for sampled density profiles.
- Remaining target: systematic density profiles, total energies, edge behavior, and finite-`N` trends between sampled data and excluded-volume LDA.

### DMC Readiness

- Use `experiments/anchors/exact_validation_packet.py` for the canonical exact
  validation packet.
- Use `experiments/anchors/finite_a_n2_reference_packet.py` for the finite-`a`
  trapped N=2 deterministic reference packet before extrapolating trust to
  larger trapped systems.
- Use `experiments/anchors/exact_tg_trap.py` for the zero-rod-length
  trapped TG harmonic validation.
- Use `experiments/dmc/local/trapped_stationarity_grid.py` for finite-`a`
  trapped local-DMC stationarity diagnostics. Enable collective RN moves only
  for an explicit comparison.
- Report coordinate observables only with the corresponding forward-walking or
  energy-response validation evidence.

Those checks precede the main thesis comparison:

```text
trapped QMC/DMC observables versus excluded-volume LDA predictions
```
