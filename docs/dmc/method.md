# RN-Block DMC Method And Numerical Checks

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
-\frac12\sum_i \partial_{x_i}^2
+ \frac12\omega^2\sum_i (x_i-x_0)^2
+ V_{\mathrm{HR}},
$$

with open-line hard-core constraints

$$
x_{i+1}-x_i \ge a.
$$

The trapped code stores the dimensionless variables
\(q=x/a_{\rm ho}\) and \(\widetilde E=E/(\hbar\omega)\), with
\(a_{\rm ho}=\sqrt{\hbar/(m\omega)}\). Substitution gives
\(\partial_x=a_{\rm ho}^{-1}\partial_q\) and the prefactors reduce to
\[
  \frac{\hbar}{m\omega a_{\rm ho}^2}=1,
  \qquad
  \frac{m\omega a_{\rm ho}^2}{\hbar}=1.
\]
This gives the dimensionless oscillator
\(-\frac12\partial_q^2+\frac12 q^2\). The hard-core boundary becomes
\(q_{i+1}-q_i\ge A=a/a_{\rm ho}\). The scan coordinates are therefore \(N\)
and \(A\). Changing the physical trap frequency at fixed rod diameter changes
the dimensionless value \(A\), because \(a_{\rm ho}\) changes. The hard-rod geometry is owned by `systems/`; guide and
local-energy formulas are owned by `wavefunctions/`; DMC propagation is owned
by `monte_carlo/dmc/`; statistical interpretation is owned by `analysis/`.

## Why RN-Block DMC

Pure local drift-diffusion moves equilibrate the trapped cloud width slowly.
RN-block DMC keeps the standard full-coordinate DMC walker state but separates
the proposal used to move walkers from the target kernel used in the
Radon-Nikodym correction. Here "target kernel" means the transition kernel in
the change-of-measure correction, while the physical Hamiltonian remains the
trapped hard-rod Hamiltonian above. For finite-\(a\), \(N>2\) trapped runs, the
current gap-\(h\)-product target is a candidate numerical kernel construction
rather than an exact many-body propagator; exact anchors, timestep checks,
RN-weight diagnostics, stationarity checks, and pure-estimator checks quantify
its reliability.

For a local DMC step,

$$
x' = x + \Delta t\,F(x) + \sqrt{\Delta t}\,\eta,
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

This is a change-of-measure correction for the sampled move. It changes the
statistical weight of the proposal path; it does not change the physical
Hamiltonian. RN-assisted rows are compared against local-DMC baselines and
cadence checks.

## Current Candidate Corridor

The active trapped-system report uses:

```text
proposal = gap-h-transform
target   = gap-h-product
guide    = reduced-TG
system   = open-line trapped hard rods
```

For \(N=2\), the COM plus one-gap target is checked against a deterministic
finite-\(a\) reference. For \(N>2\), the same family is a candidate
RN-assisted workflow compared against the local-DMC baseline and
cadence/timestep checks. Initialization and preburn reduce transient breathing
mismatch before production. They are not production estimators and do not
change the numerical decision logic.

## Observables

The RN-DMC engine directly reports the mixed Hamiltonian energy. This is the
primary energy corridor.

Coordinate observables are different:

- `R2` is the mean squared cloud radius from sampled coordinates.
- `RMS` is reported as \(\sqrt{R2}\).
- density is a coordinate histogram.

These are mixed-coordinate observables unless a pure-estimator layer is also
checked. Paper-grade coordinate claims require either:

- a pure estimator such as transported auxiliary forward walking; or
- a Hellmann-Feynman energy-response estimator for trap \(R2/RMS\).

Mixed coordinate observables remain diagnostic until that estimator check is
closed.

## Numerical Check Split

The stationarity artifact separates hard methodology failures from precision
warnings.

Hard methodology failures are:

- non-finite or invalid retained samples;
- density accounting failure;
- RN weight-control failure;
- R-hat failure;
- effective independent sample failure;
- explicit trace-stationarity failure.

No error-estimation method can override these failures.

Precision warnings are separate. Blocking plateau detection can fail because
the block curve is noisy or the coarsest block levels have too few blocks. If
the methodology checks are clean, the workflow computes three correlated-trace
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
precision warning. Missing blocking plateau is reported as
`TRIANGULATED_PRECISION_WARNING`, but only after all hard methodology checks
pass.

The warning remains visible and the error bar is inflated.

## Interpreting A Run

A clean energy candidate needs:

```text
density_accounting_clean = true
valid_finite_clean = true
RN weight status controlled
Rhat below threshold
N_eff above threshold
trace stationarity clean
finite conservative energy error
```

If those pass but blocking plateau is absent, the run can still be a controlled
energy candidate with `RN_TRAPPED_STATIONARITY_PRECISION_WARNING`.

If R-hat, N_eff, RN weights, finite/valid sample checks, density accounting, or explicit
trace-stationarity fail, the run remains unresolved.

For \(R2/RMS/density\), a paper coordinate benchmark requires the mixed RN-DMC
checks plus a completed pure-coordinate estimator or energy-response estimator
check.

## Claim Boundary

Current RN-block outputs are candidate/reference-tier artifacts until the
specific observable and parameter region passes the relevant timestep,
population, stationarity, RN-weight, density-accounting, and estimator checks.

Exact homogeneous and trapped limiting tests validate conventions and limiting
behavior. Finite-rod trapped rows still require the observable-specific checks
above before paper-level use.

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
