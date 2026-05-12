# Validation Notes

This note records the validation status of the numerical workflow. It is written as a thesis-facing summary: what was tested, what reference was used, what the result means, and what remains to be validated before using the code for trapped-system conclusions.

## 1. Purpose

The thesis uses the homogeneous hard-rod gas on a ring as a controlled benchmark before moving to trapped one-dimensional hard rods. This is useful because the homogeneous hard-rod problem has an exact excluded-volume mapping. It therefore provides a clean validation point for the numerical pipeline.

The current validation does not attempt to test the trapped system or the local-density approximation. It checks the homogeneous ring pipeline:

```text
periodic hard-rod geometry
  -> exact all-pair hard-rod trial wavefunction
  -> Metropolis VMC sampling
  -> local-energy estimator
  -> exact finite-N ring energy
```

Historical trapped VMC diagnostics were used during development. They are not
part of the public release-facing experiment surface.

## 2. Homogeneous Ring Benchmark

### Reference Result

For `N` hard rods of length `a` on a ring of length `L`, the excluded-volume mapping gives the reduced length

```text
L' = L - N a.
```

The finite-`N` reference energy per particle is

```text
E_N / N = (1 / N) sum_i (2 pi I_i / L')^2,
```

in units where `hbar^2/(2m)=1`. This finite-size expression is the pass/fail reference for the present benchmark.

The thermodynamic equation of state,

```text
e_HR(rho) = pi^2 rho^2 / [3 (1 - a rho)^2],
```

is also printed in the run output, but only as contextual information. It is not used as the finite-`N` validation target.

### Numerical Test

Command:

```bash
PYTHONPATH=src python3 experiments/validation/homogeneous_ring.py
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

For each point, the code samples configurations with the all-pair hard-rod trial wavefunction and evaluates the local kinetic energy. Since this trial is the exact homogeneous-ring benchmark form, the local energy should be constant and equal to the finite-`N` reference energy.

### Result

The benchmark passed for all 9 cases.

| Quantity | Result |
| --- | ---: |
| cases passed | 9 / 9 |
| maximum absolute error in `E/N` | 1.7763568394002505e-15 |
| valid snapshot fraction | 1.0 in every case |
| acceptance-rate range | approximately 0.83 to 0.87 |

The energy error is at floating-point roundoff level. This indicates that the periodic hard-rod geometry, the exact all-pair trial wavefunction, and the local-energy formula are mutually consistent for the homogeneous benchmark. It is not a strong standalone validation of sampler convergence, because the exact all-pair local energy is constant for every valid sampled configuration.

## 3. Interpretation

This validation establishes a controlled starting point for the code. It supports the use of the homogeneous ring as a benchmark for checking basic geometry, exclusion constraints, sampling, and energy-estimator consistency.

It does not establish that the trapped-system calculations are correct. The trapped problem removes translational invariance, introduces an external potential, and requires non-periodic density observables. Those pieces must be validated separately.

It also does not validate DMC as a production reference. DMC should remain labeled as a candidate production method until its propagation, time-step behavior, and estimator behavior are checked.

## 4. Remaining Validation Steps

Before using the code for thesis-level trapped-system conclusions, the following checks should be completed.

### Homogeneous Observables

- Initial normalization checks now cover the periodic density integral, the `g(r)` finite-`N` unique-pair sum rule, and a lattice reference for `S(k)`.
- Still useful later: compare homogeneous `g(r)` and `S(k)` shapes against literature curves or trusted reference data where available.

### Trapped Geometry and Sampling

- Open-line hard-rod geometry without periodic wrapping has an initial implementation.
- The harmonic trapping potential has an initial implementation.
- Trapped initial configurations are checked against the hard-core constraint.
- Development-only trapped VMC smoke/grid/seed/alpha scans are intentionally not
  part of the public experiment surface.

### Trapped Density and LDA

- A non-periodic trapped density estimator has an initial implementation.
- LDA normalization is checked on the same spatial grid used for sampled density profiles.
- Still needed: systematic density profiles, total energies, edge behavior, and finite-`N` trends between sampled data and excluded-volume LDA.

### DMC Readiness

- Use `experiments/dmc/rn_block/exact_tg_trap.py` for the zero-rod-length
  trapped TG harmonic validation.
- Use `experiments/dmc/rn_block/trapped_stationarity_grid.py` for finite-`a`
  trapped RN-DMC stationarity and gate diagnostics.
- Coordinate observables remain claim-limited unless the corresponding
  estimator gate is explicitly closed.

Only after those checks should the project move to the main thesis comparison:

```text
trapped QMC/DMC observables versus excluded-volume LDA predictions
```
