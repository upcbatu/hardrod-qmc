# RN-Block DMC Method Note

This document is the release-facing method note for the RN-corrected
collective-block DMC implementation. The formula audit remains in
[03_EQUATION_SOURCE_MAP.md](03_EQUATION_SOURCE_MAP.md); this note explains how
the method pieces fit together.

## Method Role

The RN-block engine is a candidate DMC production method for trapped one-
dimensional hard rods. It is not the source of the hard-rod model, the trap
Hamiltonian, or the excluded-volume LDA. Those live in `systems/`, `theory/`,
and `wavefunctions/`.

The engine owns only:

- weighted walker propagation;
- RN-corrected collective-block transitions;
- population-control bookkeeping;
- streaming mixed observables;
- checkpoint-compatible run state.

## Local DMC Step

For one local proposal step, walkers are advanced with an importance-sampled
short-time DMC convention:

$$
x' = x + \Delta t\,F(x) + \sqrt{2\Delta t}\,\eta,
\qquad
\eta \sim \mathcal N(0, I),
$$

where \(F=\nabla\log\psi_T\) is provided by the guide. The energy weight update
is the usual short-time DMC mixed-estimator ingredient:

$$
\Delta \log w_E
=
-\Delta t
\left[
\frac{E_L(x') + E_L(x)}{2}
- E_{\mathrm{ref}}
\right].
$$

This convention is method-backed by Foulkes et al. and Umrigar, Nightingale,
and Runge. It remains subject to timestep and population-control checks.

## Collective RN Block

The weak-trap mode is slow under purely local propagation, so the RN-block
implementation adds a collective proposal over a low-dimensional coordinate
\(\theta\), such as cloud scale or breathing displacement. Let
\(Q_\theta(x'\mid x)\) be the implemented proposal density and
\(K_\theta(x'\mid x)\) be the target transition density for that same block.
The Radon-Nikodym correction is

$$
\Delta\log W_{\mathrm{RN}}
=
\log K_\theta(x'\mid x)
-
\log Q_\theta(x'\mid x).
$$

If the guide is included in the proposal/target convention, the corresponding
importance-sampled ratio is

$$
\Delta\log W_{\mathrm{RN,IS}}
=
\log K_\theta(x'\mid x)
-
\log Q_\theta(x'\mid x)
+
\log\psi_T(x')-\log\psi_T(x).
$$

This is a change-of-measure correction, not a new physical Hamiltonian. The
target/proposal terms are injected through system and guide owners, so RN-block
DMC does not own the trap model or hard-rod geometry.

## Ordered Harmonic Target

For the \(a=0\) trapped Tonks-Girardeau validation, the ordered noncrossing
harmonic transition is built from a Karlin-McGregor determinant of one-body
Mehler kernels. This is used as an exact-limit validation anchor for the engine
and does not automatically validate finite-rod trapped results.

## Statistical Gates

A finite-rod trapped run is accepted only as a controlled diagnostic when all
method gates are satisfied:

- density and finite-value hygiene pass;
- RN weights stay finite and controlled;
- independent seed traces agree under \(\hat R\);
- effective independent samples are sufficient;
- autocorrelation-adjusted stationarity checks pass;
- blocking uncertainty is reported rather than hidden.

Blocking/error estimates follow Flyvbjerg and Petersen. Chain agreement and
effective-sample diagnostics follow the modern \(\hat R\) convergence-diagnostic
literature of Vehtari et al. The spread statistic is retained as a warning
unless paired with directional drift.

## Claim Boundary

RN-block DMC can be used as a thesis benchmark only after the specific
observable and parameter region pass the relevant validation gates. Exact
homogeneous and Tonks-Girardeau trapped checks validate conventions and limiting
behavior; they do not by themselves prove every finite-rod trapped run is a
paper-level benchmark.

## Core References

- Foulkes, Mitas, Needs, and Rajagopal, Rev. Mod. Phys. 73, 33 (2001).
- Umrigar, Nightingale, and Runge, J. Chem. Phys. 99, 2865 (1993).
- Karlin and McGregor, Pacific J. Math. 9, 1141 (1959).
- Flyvbjerg and Petersen, J. Chem. Phys. 91, 461 (1989).
- Vehtari, Gelman, Simpson, Carpenter, and Buerkner, Bayesian Analysis 16, 667
  (2021).
- Mazzanti, Astrakharchik, Boronat, and Casulleras, Phys. Rev. Lett. 100,
  020401 (2008).
