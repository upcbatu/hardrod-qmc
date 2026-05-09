# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 4
- omega: 0.1
- rod_length: 0.5
- seeds: 5101, 5102, 5103, 5104
- steps: 12000
- burn_in: 2000
- thinning: 20
- grid_extent: 40.0
- n_bins: 240

## Diagnostic Checks

- summary status: completed
- completed replicates: 28/28
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.06e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.0132
- max relative-density-L2 seed spread by alpha: 0.223

## Energy Diagnostic

- lowest sampled total energy alpha: 0.6
- lowest sampled total energy: 1.28605 +/- 0.017
- bracketed energy minimum: yes; the lowest sampled energy is inside the scanned alpha range
- max total-energy stderr by alpha: 0.0788
- max total-energy seed spread by alpha: 0.349
- sampled_potential_energy_mean is harmonic trap energy only; use sampled_total_energy_mean for the VMC local-energy diagnostic.

## Replicate Summary

| alpha | total energy | kinetic energy | trap energy | acceptance | relative density L2 | sampled RMS radius | RMS radius error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 1.31141 +/- 0.032 | 0.483581 +/- 0.051 | 0.827832 +/- 0.082 | 0.947042 +/- 0.0016 | 0.345921 +/- 0.027 | 6.41022 +/- 0.32 | 0.65273 +/- 0.32 |
| 0.50 | 1.32152 +/- 0.032 | 0.63794 +/- 0.071 | 0.683582 +/- 0.059 | 0.941042 +/- 0.0028 | 0.331225 +/- 0.03 | 5.83094 +/- 0.24 | 0.0734519 +/- 0.24 |
| 0.55 | 1.33634 +/- 0.029 | 0.657058 +/- 0.023 | 0.679282 +/- 0.041 | 0.937729 +/- 0.0029 | 0.33626 +/- 0.0089 | 5.82018 +/- 0.17 | 0.0626922 +/- 0.17 |
| 0.60 | 1.28605 +/- 0.017 | 0.72717 +/- 0.036 | 0.558876 +/- 0.022 | 0.935688 +/- 0.0026 | 0.328848 +/- 0.023 | 5.28295 +/- 0.11 | -0.474535 +/- 0.11 |
| 0.70 | 1.3564 +/- 0.05 | 0.823945 +/- 0.099 | 0.532456 +/- 0.052 | 0.932479 +/- 0.0026 | 0.329968 +/- 0.021 | 5.14113 +/- 0.25 | -0.616353 +/- 0.25 |
| 0.80 | 1.40289 +/- 0.058 | 0.948454 +/- 0.071 | 0.454437 +/- 0.013 | 0.928146 +/- 0.0022 | 0.392218 +/- 0.022 | 4.76518 +/- 0.071 | -0.992312 +/- 0.071 |
| 1.00 | 1.71881 +/- 0.079 | 1.36453 +/- 0.11 | 0.354277 +/- 0.03 | 0.914521 +/- 0.0024 | 0.489033 +/- 0.051 | 4.19658 +/- 0.18 | -1.56091 +/- 0.18 |

## Interpretation

This is a VMC diagnostic alpha scan.
The total-energy curve is a diagnostic variational surface for the current trial, not a production benchmark.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum unless the minimum is bracketed and uncertainties support it.
It does not validate LDA accuracy or DMC readiness.
