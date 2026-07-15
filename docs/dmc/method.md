# Trapped Hard-Rod DMC Method and Numerical Checks

This document describes the trapped hard-rod DMC implementation in
`src/hrdmc`: the default local trajectory, the optional collective
Radon-Nikodym extension, the estimators, and the numerical evidence attached to
reported results. Formula ownership and literature references are mapped in
[../03_EQUATION_SOURCE_MAP.md](../03_EQUATION_SOURCE_MAP.md).

## Physical Problem

The target system is a one-dimensional open-line hard-rod gas in a harmonic
trap,

$$
H =
-\frac12\sum_i \partial_{x_i}^2
+ \frac12\omega^2\sum_i(x_i-x_0)^2
+ V_{\mathrm{HR}},
$$

with ordered hard-core constraints

$$
x_{i+1}-x_i \ge a.
$$

The trapped code uses \(q=x/a_{\rm ho}\),
\(\widetilde E=E/(\hbar\omega)\), and
\(A=a/a_{\rm ho}\), where
\(a_{\rm ho}=\sqrt{\hbar/(m\omega)}\). The dimensionless Hamiltonian is
\(-\frac12\sum_i\partial_{q_i}^2+\frac12\sum_iq_i^2\) inside the hard-rod
domain. Production scans are therefore indexed by \(N\) and \(A\).

## Default Local DMC Trajectory

The default implementation is
`src/hrdmc/monte_carlo/dmc/local/`. It does not construct or require a
collective proposal.

For the default Metropolis-corrected drift-diffusion step (MALA), a walker at
\(X\) proposes

$$
X' = X + \Delta\tau\,\nabla\log\Psi_T(X)
+ \sqrt{\Delta\tau}\,\eta,
\qquad \eta\sim\mathcal N(0,I).
$$

The Metropolis acceptance probability includes both the guide-amplitude ratio
and the forward/reverse Gaussian proposal densities. Invalid hard-rod proposals
and Metropolis rejections leave the previous valid walker in place. The engine
records acceptance, invalid-proposal, Metropolis-rejection, drift, and mobility
telemetry.

After the local move, branching adds

$$
\Delta\log w_E =
-\Delta\tau
\left[
\frac{E_L(X)+E_L(X')}{2}-E_{\rm ref}
\right].
$$

Weights are recentered for numerical stability. The population is resampled
when its weight effective sample size crosses the configured fraction of the
walker count. Resampling indices and pre/post-resampling weights are retained
in the transport event needed by forward walking.

## Optional Collective RN Move

`src/hrdmc/monte_carlo/dmc/collective_rn/` implements a scheduled collective
proposal. It is disabled by default and is injected into the local engine only
when a workflow explicitly requests it.

For collective proposal density \(Q(Y\mid X)\) and selected target transition
density \(K(Y\mid X)\), the additional log-weight increment is

$$
\Delta\log w_{\rm RN}
= \log K(Y\mid X)-\log Q(Y\mid X).
$$

When the importance-sampled convention includes the guide ratio, the increment
also contains

$$
\log\Psi_T(Y)-\log\Psi_T(X).
$$

This is a change-of-measure correction for that scheduled proposal. It does not
replace the Hamiltonian or redefine the local DMC trajectory. Runs using the
extension report its cadence, proposal/target metadata, event count, and
weight-increment diagnostics separately from the local-step telemetry.

## Guide Function

The guide owns `log_value`, `grad_log_value`, `lap_log_value`, `local_energy`,
and validity checks. It changes importance-sampling efficiency and local-energy
fluctuations but not the target Hamiltonian. Guide parameters must therefore be
recorded, and a changed guide requires renewed timestep, population, and
stationarity checks in the affected parameter region.

## Observables

The mixed local-energy estimator provides the DMC energy. Coordinate
observables require additional care because the mixed distribution contains the
guide:

- \(R^2\) is the mean squared cloud radius;
- `rms_radius` is \(\sqrt{R^2}\), formed after aggregating \(R^2\);
- density is the open-line particle histogram;
- pair-distance density and \(S(k)\) are optional vector observables.

The repository provides two independent routes for coordinate quantities:

- transported auxiliary forward walking for \(R^2\), density, pair-distance
  density, and \(S(k)\);
- a Hellmann-Feynman energy response for trap \(R^2\), with RMS obtained only
  after differentiating the energy.

Direct weighted coordinate averages remain mixed-estimator diagnostics. They
are not substituted for the forward-walking or energy-response result.

## Numerical Checks

A finite-\(A\) many-body result is interpreted together with the following
evidence:

- finite and valid retained samples;
- density normalization when density is reported;
- local-step acceptance, invalid-proposal rate, drift scale, and mobility;
- log-weight span, weight ESS, resampling frequency, and source-family
  genealogy;
- agreement across independent seeds, including the serialized split-R-hat
  diagnostic and an effective sample count;
- stable traces after burn-in and conservative correlated-error estimates;
- a timestep comparison in the same physical parameter region;
- a walker-population comparison when population sensitivity is plausible;
- forward-walking lag stability and genealogy support for each reported
  coordinate observable;
- agreement with exact or deterministic anchors where available.

Blocking, Sokal, Geyer, and flat-top HAC estimates describe precision; they do
not repair invalid samples, chain disagreement, population collapse, or a
failed timestep comparison. When several valid error estimates are available,
the workflow reports the conservative maximum and retains the individual
diagnostics.

## Reading an Artifact

Benchmark packets store physical estimates under `estimates`. The packet keeps
the mixed energy, forward-walking coordinate results, LDA references, seed-level
summaries, stationarity diagnostics, and method metadata distinct. Optional
collective RN metadata is absent or disabled for an ordinary local-DMC run.

A numerical status is a compact summary of the recorded checks, not a new
physical quantity. The underlying values, thresholds, traces, and seed results
remain the evidence used to interpret a parameter case.

The final-matrix assembly verifies every source manifest before selecting an
observable. An observable-specific supplement must reproduce the primary
packet's case, seeds, controls, guide identity, implementation tree, and mixed
energy. Source statuses are retained even when a later matrix-level assessment
changes the selected status. Source locators are stored relative to the final
assembly directory so the result tree remains portable with its bound packets.

For a declared matrix-level energy-stationarity assessment, the slope,
first/last-quarter, and late-cumulative statistics from every seed form one
simultaneous family. If there are \(K\) two-sided statistics, the Bonferroni
critical value at confidence \(1-\alpha\) is

\[
z_{\mathrm{crit}}=\Phi^{-1}\!\left(1-\frac{\alpha}{2K}\right).
\]

The first/last-block statistic is retained in each source packet but is not
counted twice when the four-block construction makes it identical to the
first/last-quarter statistic. The assembly also requires its configured
split-R-hat and effective-sample screens. This is an operational simultaneous
screen, not a proof of stationarity and not a rank-normalized R-hat. A policy
chosen after inspecting the source matrix is serialized as `retrospective`.

The default population diagnostics are deliberately conservative operational
screens, not universal constants. Weight-ESS fractions of 0.20 (warning) and
0.10 (invalid), a log-weight-span warning at 50, a forward-walking source-family
ESS of at least 50, and a largest-family fraction no greater than 0.10 are used
to expose poorly resolved populations before a physical comparison is made.
They do not prove convergence by themselves, and a publishable result still
requires seed, timestep, population, and lag-sensitivity evidence. Every value
is serialized so a different policy cannot be inherited silently on resume.

## Core References

- Foulkes, Mitas, Needs, and Rajagopal, Rev. Mod. Phys. 73, 33 (2001).
- Umrigar, Nightingale, and Runge, J. Chem. Phys. 99, 2865 (1993).
- Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989).
- Sokal, Functional Integration, NATO ASI Series B 361, 131-192 (1997).
- Geyer, Statistical Science 7, 473-483 (1992).
- Andrews, Econometrica 59, 817-858 (1991).
- Politis and Romano, Journal of Time Series Analysis 16, 67-103 (1995).
- Vehtari, Gelman, Simpson, Carpenter, and Buerkner, Bayesian Analysis 16,
  667-718 (2021).
- Casulleras and Boronat, Phys. Rev. B 52, 3654-3661 (1995).
- Sarsa, Boronat, and Casulleras, J. Chem. Phys. 116, 5956-5962 (2002).
- Mazzanti, Astrakharchik, Boronat, and Casulleras, Phys. Rev. Lett. 100,
  020401 (2008).
