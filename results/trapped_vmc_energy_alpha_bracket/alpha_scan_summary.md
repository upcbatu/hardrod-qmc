# Trapped VMC Alpha-Scan Diagnostic

## Run Controls

- N: 4
- omega: 0.1
- rod_length: 0.5
- seeds: 5101, 5102, 5103, 5104
- steps: 6000
- burn_in: 1000
- thinning: 20
- grid_extent: 40.0
- n_bins: 240

## Diagnostic Checks

- summary status: completed
- completed replicates: 28/28
- minimum valid snapshot fraction: 1
- max |sampled density integral error|: 3.11e-14
- max |LDA integrated particles error|: 2.18e-10
- max acceptance seed spread by alpha: 0.0347
- max relative-density-L2 seed spread by alpha: 0.39

## Energy Diagnostic

- lowest sampled total energy alpha: 0.5
- lowest sampled total energy: 1.29421 +/- 0.029
- bracketed energy minimum: no; the lowest sampled energy is on the upper-alpha boundary
- max total-energy stderr by alpha: 0.529
- max total-energy seed spread by alpha: 2.28
- sampled_potential_energy_mean is harmonic trap energy only; use sampled_total_energy_mean for the VMC local-energy diagnostic.

## Replicate Summary

| alpha | total energy | kinetic energy | trap energy | acceptance | relative density L2 | sampled RMS radius | RMS radius error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.08 | 2.21015 +/- 0.25 | 0.213996 +/- 0.037 | 1.99616 +/- 0.28 | 0.955792 +/- 0.0046 | 0.577374 +/- 0.044 | 9.89895 +/- 0.78 | 4.14146 +/- 0.78 |
| 0.12 | 2.55888 +/- 0.53 | 0.24085 +/- 0.087 | 2.31803 +/- 0.61 | 0.954375 +/- 0.0076 | 0.671439 +/- 0.079 | 10.4611 +/- 1.5 | 4.70359 +/- 1.5 |
| 0.16 | 1.62213 +/- 0.2 | 0.312299 +/- 0.057 | 1.30983 +/- 0.24 | 0.9505 +/- 0.0061 | 0.573481 +/- 0.087 | 7.99786 +/- 0.71 | 2.24038 +/- 0.71 |
| 0.20 | 1.68421 +/- 0.14 | 0.282166 +/- 0.026 | 1.40204 +/- 0.15 | 0.953042 +/- 0.0045 | 0.59954 +/- 0.054 | 8.3348 +/- 0.46 | 2.57731 +/- 0.46 |
| 0.25 | 1.48448 +/- 0.12 | 0.363679 +/- 0.088 | 1.1208 +/- 0.2 | 0.952542 +/- 0.0058 | 0.53031 +/- 0.083 | 7.39353 +/- 0.68 | 1.63604 +/- 0.68 |
| 0.35 | 1.4049 +/- 0.065 | 0.45844 +/- 0.066 | 0.946464 +/- 0.13 | 0.944875 +/- 0.0044 | 0.504888 +/- 0.064 | 6.82833 +/- 0.48 | 1.07084 +/- 0.48 |
| 0.50 | 1.29421 +/- 0.029 | 0.653193 +/- 0.083 | 0.641019 +/- 0.06 | 0.942208 +/- 0.002 | 0.442362 +/- 0.019 | 5.64284 +/- 0.26 | -0.114646 +/- 0.26 |

## Interpretation

This is a VMC diagnostic alpha scan.
The total-energy curve is a diagnostic variational surface for the current trial, not a production benchmark.
The scan can guide density/radius diagnostics.
It does not select a production variational optimum unless the minimum is bracketed and uncertainties support it.
It does not validate LDA accuracy or DMC readiness.
