# RN-Block DMC Method And Gate Semantics

This document is the professor-facing description of the trapped hard-rod
RN-block DMC corridor implemented in `src/hrdmc`. It explains what the method
does, what the output means, and which claims are allowed. The formula audit and
source bibliography remain in
[../03_EQUATION_SOURCE_MAP.md](../03_EQUATION_SOURCE_MAP.md).

## Physical Problem

The target system is a one-dimensional open-line hard-rod gas in a harmonic
trap,

$$
H =
-\sum_i \partial_{x_i}^2
+ \frac12\omega^2\sum_i (x_i-x_0)^2
+ V_{\mathrm{HR}},
$$

with open-line hard-core constraints

$$
x_{i+1}-x_i \ge a.
$$

The code uses units \(\hbar^2/(2m)=1\). The hard-rod geometry is owned by
`systems/`; guide and local-energy formulas are owned by `wavefunctions/`;
DMC propagation is owned by `monte_carlo/dmc/`; statistical interpretation is
owned by `analysis/`.

## Why RN-Block DMC

Pure local drift-diffusion moves equilibrate the weak-trap cloud width slowly.
RN-block DMC keeps the standard full-coordinate DMC walker state but adds a
collective proposal for slow cloud-size motion. The proposal is corrected by a
Radon-Nikodym log-weight ratio, so changing the proposal does not change the
Hamiltonian being sampled.

For a local DMC step,

$$
x' = x + \Delta t\,F(x) + \sqrt{2\Delta t}\,\eta,
\qquad
F=\nabla \log\Psi_T.
$$

The local energy weight contribution is

$$
\Delta\log w_E =
-\Delta t
\left[
\frac{E_L(x')+E_L(x)}{2}-E_{\mathrm{ref}}
\right].
$$

For an RN collective block with implemented proposal density \(Q(x'\mid x)\)
and target kernel \(K(x'\mid x)\),

$$
\Delta\log w_{\mathrm{RN}}
=
\log K(x'\mid x)-\log Q(x'\mid x).
$$

When the guide ratio is part of the importance-sampled convention, the logged
increment also includes

$$
\log\Psi_T(x')-\log\Psi_T(x).
$$

This is a change-of-measure correction. It is not a new physical approximation.
The production question is whether the proposal is close enough to the target
that RN weights remain controlled.

## Current Candidate Corridor

The current weak-trap candidate corridor is:

```text
dt = 0.00125
RN cadence tau = 0.01
safe010 fixed multiscale collective proposal
LDA-RMS logspread initialization
optional breathing preburn before production
```

The initialization and preburn only reduce transient breathing mismatch before
production. They are not production estimators and do not change gate logic.

## Observables

The RN-DMC engine directly reports the mixed Hamiltonian energy. This is the
primary energy corridor.

Coordinate observables are different:

- `R2` is the mean squared cloud radius from sampled coordinates.
- `RMS` is reported as \(\sqrt{R2}\).
- density is a coordinate histogram.

These are mixed-coordinate observables unless a pure-estimator layer closes its
own gate. Paper-grade coordinate claims require either:

- a pure estimator such as transported auxiliary forward walking; or
- a Hellmann-Feynman energy-response estimator for trap \(R2/RMS\).

Mixed coordinate observables remain diagnostic until that estimator gate is
closed.

## Gate Split

The stationarity artifact separates hard methodology failures from precision
warnings.

Hard methodology failures are:

- non-finite or invalid retained samples;
- density accounting failure;
- `RN_WEIGHT_NO_GO`;
- R-hat failure;
- effective independent sample failure;
- explicit trace-stationarity failure.

No error-estimation method can override these failures.

Precision warnings are different. Blocking plateau detection can fail because
the block curve is noisy or the coarsest block levels have too few blocks. If
the methodology gate is clean, the workflow computes three correlated-trace
standard-error estimates:

- Sokal integrated-autocorrelation window;
- Geyer initial positive/monotone sequence;
- flat-top HAC long-run variance.

The reported standard error is conservative:

$$
\mathrm{SE}_{\mathrm{reported}}
=
\max(
\mathrm{SE}_{\mathrm{seed}},
\mathrm{SE}_{\mathrm{blocking}},
\mathrm{SE}_{\mathrm{correlated}}
).
$$

If at least two correlated-error estimators agree within their estimated
one-sigma uncertainty, the metric reports `TRIANGULATED_2_OF_3`. If they do not
agree but remain finite, the artifact keeps the larger error and reports a
precision warning. Missing blocking plateau can therefore become
`TRIANGULATED_PRECISION_WARNING`, but only after all hard methodology gates
pass.

This is not gate softening: the warning remains visible and the error bar is
inflated.

## Interpreting A Run

A clean energy candidate needs:

```text
density_accounting_clean = true
valid_finite_clean = true
rn_weight_status != RN_WEIGHT_NO_GO
Rhat below threshold
N_eff above threshold
trace stationarity clean
finite conservative energy error
```

If those pass but blocking plateau is absent, the run can still be a controlled
energy candidate with `RN_TRAPPED_STATIONARITY_PRECISION_WARNING`.

If R-hat, N_eff, RN weights, hygiene, density accounting, or explicit
trace-stationarity fail, the run remains NO-GO.

For \(R2/RMS/density\), passing the mixed RN-DMC gate is not enough for a paper
coordinate benchmark. A pure-coordinate estimator or energy-response estimator
must also pass.

## Claim Boundary

Current RN-block outputs are candidate/reference-tier artifacts until the
specific observable and parameter region pass the relevant timestep,
population, stationarity, RN-weight, density-accounting, and estimator gates.

Exact homogeneous and trapped limiting tests validate conventions and limiting
behavior. They do not by themselves prove every finite-rod trapped run is a
paper-level benchmark.

## Core References

- Foulkes, Mitas, Needs, and Rajagopal, Rev. Mod. Phys. 73, 33 (2001).
- Umrigar, Nightingale, and Runge, J. Chem. Phys. 99, 2865 (1993).
- Karlin and McGregor, Pacific J. Math. 9, 1141 (1959).
- Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989).
- Sokal, Functional Integration, NATO ASI Series B 361, 131-192 (1997).
- Geyer, Statistical Science 7, 473-483 (1992).
- Andrews, Econometrica 59, 817-858 (1991).
- Politis and Romano, Journal of Time Series Analysis 16, 67-103 (1995).
- Vehtari, Gelman, Simpson, Carpenter, and Buerkner, Bayesian Analysis 16,
  667-718 (2021).
- Mazzanti, Astrakharchik, Boronat, and Casulleras, Phys. Rev. Lett. 100,
  020401 (2008).
