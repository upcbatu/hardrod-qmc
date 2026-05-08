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

An initial trapped VMC diagnostic path has also been added. It is useful for checking geometry, density-estimator, and LDA-grid consistency, but it is not yet a production trapped benchmark.

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
PYTHONPATH=src python3 experiments/01_uniform_hard_rods_validation.py
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
- A trapped VMC smoke test exists with explicit `VMC diagnostic` benchmark-tier metadata.

The current trapped smoke command is:

```bash
PYTHONPATH=src python3 experiments/02_trapped_vmc_smoke.py
```

It writes:

```text
results/trapped_vmc_smoke/summary.json
results/trapped_vmc_smoke/density_profiles.npz
```

The smoke test currently reports sampled density, LDA density on the same grid, LDA normalization, raw density L2 difference, and relative density L2 difference. Sampled histogram density normalization is checked with bin widths rather than trapezoidal integration over bin centers. A representative dry-run with `N=4`, `a=0.5`, and `omega=0.2` gave:

| Quantity | Result |
| --- | ---: |
| valid snapshot fraction | 1.0 |
| sampled density integral | 4.0 |
| LDA integrated particles | 4.0 |
| acceptance rate | about 0.88 |
| density L2 difference | diagnostic only; value depends on chain length and trial settings |
| relative density L2 difference | diagnostic only; value depends on chain length and trial settings |

This is a diagnostic result only. It should be used to catch implementation and convention errors, not as a thesis-level claim about LDA accuracy.

### Trapped Density and LDA

- A non-periodic trapped density estimator has an initial implementation.
- LDA normalization is checked on the same spatial grid used for sampled density profiles.
- Still needed: systematic density profiles, total energies, edge behavior, and finite-`N` trends between sampled data and excluded-volume LDA.

### DMC Readiness

- Validate DMC time-step dependence.
- Validate population-control behavior.
- Decide which observables need mixed, extrapolated, or pure-estimator treatment before using DMC as the main production reference.

## 5. Current Diagnostic Grid

The trapped smoke path has been promoted into a controlled diagnostic grid:

```text
N = 4, 8
a = 0.5
omega = 0.05, 0.10, 0.20
```

Command:

```bash
PYTHONPATH=src python3 experiments/03_trapped_vmc_diagnostic_grid.py
```

Output:

```text
results/trapped_vmc_grid/summary.json
results/trapped_vmc_grid/*_density_profiles.npz
```

For each point, the output includes:

- sampled density profile;
- LDA density profile on the same grid;
- density L2 difference;
- relative density L2 difference;
- valid snapshot fraction;
- acceptance rate;
- benchmark-tier metadata.

This remains a diagnostic grid. It checks plumbing, normalization, and rough trends, but it does not yet establish LDA accuracy.

## 6. Next Step

The first stability check has been added as a seed-replicate diagnostic for a selected trapped case.

Command:

```bash
PYTHONPATH=src python3 experiments/04_trapped_vmc_seed_stability.py
```

Output:

```text
results/trapped_vmc_seed_stability/summary.json
results/trapped_vmc_seed_stability/*_density_profiles.npz
```

It reports replicate mean, sample standard deviation, standard error, and spread for acceptance rate, density-normalization error, sampled potential energy, and VMC-versus-LDA density L2 error.

This remains a `VMC diagnostic` check. Its purpose is to decide whether the trapped VMC settings are stable enough to inspect before moving toward DMC.

## 7. Next Step

The next technical step is to make the diagnostic grid numerically meaningful enough to decide what must be improved before DMC:

- check density-profile convergence with longer VMC chains;
- inspect whether the diagnostic trial parameter should be optimized;
- add simple plots for \(n_{\mathrm{VMC}}(x)\) versus \(n_{\mathrm{LDA}}(x)\);
- keep all outputs labeled as `VMC diagnostic`.

Only after those checks should the project move to the main thesis comparison:

```text
trapped QMC/DMC observables versus excluded-volume LDA predictions
```
