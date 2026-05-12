# Equation and Method Source Map

This document is the audit map for formulas and method conventions in
`src/hrdmc`. It answers four questions for each scientific formula:

```text
What is the formula?
Where is it implemented?
What is the source basis?
What is the claim boundary?
```

It intentionally does not source-map plotting, JSON IO, CLI parsing, or config
loading, because those are software plumbing rather than physics or QMC
formulas.

## Source Status

Every entry below has one of these statuses:

- `Primary physics`: formula comes from the hard-rod/trapped-gas thesis
  literature.
- `Method paper`: formula is a standard QMC/statistical method with a cited
  method source.
- `Analytic identity`: formula is an elementary analytic identity used by the
  code; it is not a thesis physics claim by itself.
- `Repo convention`: implementation normalization or utility formula. These
  must not be presented as literature claims unless a source is added.

## Bibliography

### Thesis Physics

- `[Mazzanti2008HardRods]` F. Mazzanti, G. E. Astrakharchik, J. Boronat,
  J. Casulleras, *Ground-State Properties of a One-Dimensional System of Hard
  Rods*, **Phys. Rev. Lett. 100**, 020401 (2008).
  DOI: [10.1103/PhysRevLett.100.020401](https://doi.org/10.1103/PhysRevLett.100.020401)
- `[AstrakharchikGiorgini2002TrappedCrossover]` G. E. Astrakharchik and
  S. Giorgini, *Quantum Monte Carlo study of the three- to one-dimensional
  crossover for a trapped Bose gas*, **Phys. Rev. A 66**, 053614 (2002).
  DOI: [10.1103/PhysRevA.66.053614](https://doi.org/10.1103/PhysRevA.66.053614)
- `[Astrakharchik2005LDA]` G. E. Astrakharchik, *Local density approximation for
  a perturbative equation of state*, **Phys. Rev. A 72**, 063620 (2005).
  DOI: [10.1103/PhysRevA.72.063620](https://doi.org/10.1103/PhysRevA.72.063620)
- `[GirardeauAstrakharchik2010TrappedHardSphere]` M. D. Girardeau and
  G. E. Astrakharchik, *Wave functions of the super-Tonks-Girardeau gas and the
  trapped one-dimensional hard-sphere Bose gas*, **Phys. Rev. A 81**,
  061601(R) (2010).
  DOI: [10.1103/PhysRevA.81.061601](https://doi.org/10.1103/PhysRevA.81.061601)

### QMC And Statistics Methods

- `[Foulkes2001QMC]` W. M. C. Foulkes, L. Mitas, R. J. Needs, G. Rajagopal,
  *Quantum Monte Carlo simulations of solids*, **Rev. Mod. Phys. 73**, 33
  (2001). DOI: [10.1103/RevModPhys.73.33](https://doi.org/10.1103/RevModPhys.73.33)
- `[UmrigarNightingaleRunge1993DMC]` C. J. Umrigar, M. P. Nightingale,
  K. J. Runge, *A diffusion Monte Carlo algorithm with very small time-step
  errors*, **J. Chem. Phys. 99**, 2865 (1993).
  DOI: [10.1063/1.465195](https://doi.org/10.1063/1.465195)
- `[KarlinMcGregor1959]` S. Karlin, J. McGregor, *Coincidence probabilities*,
  **Pacific J. Math. 9**, 1141-1164 (1959).
  DOI: [10.2140/pjm.1959.9.1141](https://doi.org/10.2140/pjm.1959.9.1141)
- `[FlyvbjergPetersen1989Blocking]` H. Flyvbjerg, H. G. Petersen, *Error
  estimates on averages of correlated data*, **J. Chem. Phys. 91**, 461-466
  (1989). DOI: [10.1063/1.457480](https://doi.org/10.1063/1.457480)
- `[VehtariGelmanSimpsonCarpenterBuerkner2021Rhat]` A. Vehtari, A. Gelman,
  D. Simpson, B. Carpenter, P.-C. Buerkner, *Rank-normalization, folding, and
  localization: An improved R-hat for assessing convergence of MCMC*,
  **Bayesian Analysis 16**, 667-718 (2021).
  DOI: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
- `[Sokal1997MC]` A. D. Sokal, *Monte Carlo Methods in Statistical Mechanics:
  Foundations and New Algorithms*, in *Functional Integration*, NATO ASI
  Series B, **361**, 131-192 (1997).
  DOI: [10.1007/978-1-4899-0319-8_6](https://doi.org/10.1007/978-1-4899-0319-8_6)
- `[Geyer1992PracticalMCMC]` C. J. Geyer, *Practical Markov Chain Monte Carlo*,
  **Statistical Science 7**, 473-483 (1992).
  DOI: [10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137)
- `[Andrews1991HAC]` D. W. K. Andrews, *Heteroskedasticity and Autocorrelation
  Consistent Covariance Matrix Estimation*, **Econometrica 59**, 817-858
  (1991). DOI: [10.2307/2938229](https://doi.org/10.2307/2938229)
- `[PolitisRomano1995FlatTop]` D. N. Politis and J. P. Romano,
  *Bias-Corrected Nonparametric Spectral Estimation*, **Journal of Time Series
  Analysis 16**, 67-103 (1995).
  DOI: [10.1111/j.1467-9892.1995.tb00223.x](https://doi.org/10.1111/j.1467-9892.1995.tb00223.x)
- `[Hellmann1937Quantenchemie]` H. Hellmann, *Einführung in die
  Quantenchemie*, Franz Deuticke, Leipzig und Wien (1937).
- `[Feynman1939Forces]` R. P. Feynman, *Forces in Molecules*,
  **Phys. Rev. 56**, 340-343 (1939).
  DOI: [10.1103/PhysRev.56.340](https://doi.org/10.1103/PhysRev.56.340)

### Optional Literature

- `[LiebLiniger1963GroundState]` E. H. Lieb and W. Liniger, *Exact Analysis of
  an Interacting Bose Gas. I. The General Solution and the Ground State*,
  **Phys. Rev. 130**, 1605-1616 (1963).
  DOI: [10.1103/PhysRev.130.1605](https://doi.org/10.1103/PhysRev.130.1605)
- `[Lieb1963ExcitationSpectrum]` E. H. Lieb, *Exact Analysis of an Interacting
  Bose Gas. II. The Excitation Spectrum*, **Phys. Rev. 130**, 1616-1624 (1963).
  DOI: [10.1103/PhysRev.130.1616](https://doi.org/10.1103/PhysRev.130.1616)
- `[Astrakharchik2005Quasi1DHardRods]` G. E. Astrakharchik, J. Boronat,
  J. Casulleras, S. Giorgini, *Beyond the Tonks-Girardeau Gas: Strongly
  Correlated Regime in Quasi-One-Dimensional Bose Gases*, **Phys. Rev. Lett.
  95**, 190407 (2005).
  DOI: [10.1103/PhysRevLett.95.190407](https://doi.org/10.1103/PhysRevLett.95.190407)

## Units

The repository uses:

$$
\frac{\hbar^2}{2m}=1.
$$

Therefore kinetic local-energy terms use:

$$
T_{\mathrm{local}}
=
-\sum_i
\left[
\partial_i^2\log\Psi_T
+
\left(\partial_i\log\Psi_T\right)^2
\right].
$$

Physical-unit formulas must reinsert \(\hbar^2/(2m)\) or \(m\) explicitly.

## Estimator Response Formulas

### E1. Hellmann-Feynman Trap R2/RMS Response

Code:
[src/hrdmc/estimators/energy_response.py](src/hrdmc/estimators/energy_response.py)

Formula:

For

$$
H(\lambda)=H_0+\lambda\sum_i (x_i-x_0)^2,
\qquad
\lambda=\frac12\omega^2,
$$

Hellmann-Feynman gives

$$
\frac{dE_0}{d\lambda}
=
\left\langle\sum_i (x_i-x_0)^2\right\rangle_{\rm pure}.
$$

Therefore

$$
R_2^{\rm pure}
=
\frac1N\frac{dE_0}{d\lambda},
\qquad
R_{\rm RMS}^{\rm paper}
=
\sqrt{R_2^{\rm pure}}.
$$

Source basis:
Hellmann-Feynman theorem from `[Hellmann1937Quantenchemie]` and
`[Feynman1939Forces]`.

Claim boundary:
This only estimates trap \(R_2\)/RMS from RN-DMC energy artifacts. It is not a
density estimator. Paper-grade use requires every energy point to pass the
RN-DMC methodology gates; missing gate metadata remains diagnostic only.

## Systems And Geometry

### S1. Periodic Hard-Rod Ring

Code:
[src/hrdmc/systems/hard_rods.py](src/hrdmc/systems/hard_rods.py)

Formula:

$$
\rho = \frac{N}{L},
\qquad
\eta = \rho a,
\qquad
k_m = \frac{2\pi m}{L}.
$$

The periodic hard-core constraint is:

$$
\min_i d_i \ge a,
$$

where \(d_i\) are nearest-neighbor gaps on the ring.

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
This is geometry and boundary-condition ownership only. It does not own EOS,
LDA, or DMC benchmark claims.

### S2. Open-Line Hard-Rod Geometry

Code:
[src/hrdmc/systems/open_line.py](src/hrdmc/systems/open_line.py)

Formula:

$$
x_{i+1}-x_i \ge a
\quad
\text{after sorting open-line positions}.
$$

Source basis:
`Primary physics`, trapped one-dimensional hard-core context from
`[GirardeauAstrakharchik2010TrappedHardSphere]`.

Claim boundary:
This is geometry only. The trap, guide, LDA, and sampler are separate owners.

### S3. External Potentials

Code:
[src/hrdmc/systems/external_potential.py](src/hrdmc/systems/external_potential.py)

Formula:

$$
V_{\mathrm{harm}}(x)
=
\frac{1}{2}\omega^2(x-x_0)^2,
\qquad
V_{\mathrm{zero}}(x)=0,
$$

and the utility cosine potential is:

$$
V_{\mathrm{cos}}(x)
=
A\cos\left(\frac{2\pi x}{\lambda}\right).
$$

Source basis:
The harmonic trap is `Primary physics` / standard trapped-gas model with QMC
context from `[AstrakharchikGiorgini2002TrappedCrossover]`. The zero and cosine
potentials are `Repo convention` utilities.

Claim boundary:
The thesis trapped hard-rod path uses the harmonic trap. The cosine potential
must not be cited as a thesis physics formula unless a separate source and use
case are added.

### S4. Reduced Hard-Rod Length

Code:
[src/hrdmc/systems/reduced.py](src/hrdmc/systems/reduced.py)

Formula:

$$
L_{\mathrm{eff}} = L - Na.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
This is a geometry identity used by ring theory and ring trial states. It is
not an LDA solver and not a trapped exact solution.

### S5. System-Owned Propagator Interfaces

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula:

$$
K_{\Delta\tau}^{\mathrm{sys}}(\mathbf{Y}\mid\mathbf{X})
\approx
\langle \mathbf{Y}|
e^{-\Delta\tau \hat H_{\mathrm{sys}}}
|\mathbf{X}\rangle .
$$

Source basis:
`Method paper`, DMC short-time-kernel convention from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`.

Claim boundary:
`systems/` owns the target-kernel interface. RN-block consumes it and must not
silently rebuild Hamiltonian physics.

### S6. Harmonic Mehler Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula implemented:

$$
\gamma = \sqrt{2}\omega,
\qquad
m_\gamma = \frac{\omega}{\sqrt{2}},
$$

$$
\mu_\tau(x)
=
x_0 + \frac{x-x_0}{\cosh(\gamma\tau)},
\qquad
\sigma_\tau^2
=
\frac{\tanh(\gamma\tau)}{m_\gamma}.
$$

Source basis:
`Analytic identity`, normalized imaginary-time harmonic-oscillator transition.
QMC kernel usage is method-backed by `[Foulkes2001QMC]`.

Claim boundary:
Exact for the one-body harmonic kernel in repo units. It is not by itself an
exact many-body trapped hard-rod propagator.

### S7. Exact Ordered Harmonic Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula implemented for the zero-rod-length trapped hard-core limit:

$$
K_{\mathrm{TG,harm}}(\mathbf{y}\mid\mathbf{x};\tau)
=
\det\left[
K_{\mathrm{harm}}(y_j\mid x_i;\tau)
\right]_{i,j=1}^{N}.
$$

For the same model, the exact trapped Tonks-Girardeau ground-state energy in
repo units is:

$$
E_0
=
\sum_{n=0}^{N-1}(2n+1)\frac{\omega}{\sqrt{2}}
=
\frac{N^2\omega}{\sqrt{2}}.
$$

Source basis:
`Analytic identity`, harmonic oscillator spectrum plus the ordered
Karlin-McGregor determinant `[KarlinMcGregor1959]` and trapped hard-core/TG
mapping context from `[GirardeauAstrakharchik2010TrappedHardSphere]`.

Claim boundary:
This is an exact validation anchor only for \(a=0\) in a harmonic trap. It
does not validate finite-rod trapped benchmarks by itself.

### S8. Free Ordered Hard-Rod Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula:

$$
u_i=x_i-a\left(i-\frac{N-1}{2}\right),
$$

$$
p_\tau(z)
=
\frac{1}{\sqrt{4\pi\tau}}
\exp\left[-\frac{z^2}{4\tau}\right],
$$

$$
K_{\mathrm{free,ordered}}(\mathbf{u}'\mid\mathbf{u})
=
\det\left[p_\tau(u'_j-u_i)\right]_{i,j=1}^{N}.
$$

Source basis:
`Method paper`, `[KarlinMcGregor1959]` determinant logic for non-crossing
paths, applied to reduced hard-rod coordinates from `[Mazzanti2008HardRods]`.

Claim boundary:
Exact for free ordered/non-crossing diffusion. Trap effects are not included
until the primitive endpoint factor below.

### S9. Open Hard-Rod Trap Primitive Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula:

$$
\log K_{\mathrm{primitive}}
=
\log K_{\mathrm{free,ordered}}
-
\frac{\tau}{2}
\left[
V(\mathbf{X})+V(\mathbf{Y})
\right].
$$

Source basis:
`Method paper`, primitive short-time DMC convention from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`.

Claim boundary:
Approximate short-time target density. It requires timestep validation before
paper-level trapped benchmark claims.

## Theory

### T1. Finite-N Homogeneous Ring Energy

Code:
[src/hrdmc/theory/hard_rods.py](src/hrdmc/theory/hard_rods.py)

Formula:

$$
q_j = j-\frac{N-1}{2},
\qquad
k_j=\frac{2\pi q_j}{L_{\mathrm{eff}}},
$$

$$
\frac{E_N}{N}
=
\frac{1}{N}\sum_{j=0}^{N-1} k_j^2
=
\frac{\pi^2(N^2-1)}{3L_{\mathrm{eff}}^2}.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
This is the homogeneous validation benchmark. It is not the trapped result.

### T2. Thermodynamic Hard-Rod EOS

Code:
[src/hrdmc/theory/hard_rods.py](src/hrdmc/theory/hard_rods.py)

Formula:

$$
e_{\mathrm{HR}}(\rho)
=
\frac{\pi^2\rho^2}{3(1-a\rho)^2},
$$

$$
\epsilon_{\mathrm{HR}}(\rho)
=
\rho e_{\mathrm{HR}}(\rho)
=
\frac{\pi^2\rho^3}{3(1-a\rho)^2}.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
Thermodynamic homogeneous EOS in repo units. It is an LDA input, not a trapped
exact solution.

### T3. Hard-Rod Chemical Potential

Code:
[src/hrdmc/theory/hard_rods.py](src/hrdmc/theory/hard_rods.py)

Formula:

$$
\mu_{\mathrm{HR}}(\rho)
=
\frac{d\epsilon_{\mathrm{HR}}}{d\rho}
=
\frac{\pi^2\rho^2(3-a\rho)}
{3(1-a\rho)^3}.
$$

Source basis:
`Primary physics`, derivative of the EOS from `[Mazzanti2008HardRods]`.

Claim boundary:
The inverse chemical potential in code is numerical bisection. The bisection is
an implementation method, not a separate physics formula.

### T4. Excluded-Volume LDA

Code:
[src/hrdmc/theory/lda.py](src/hrdmc/theory/lda.py)

Formula:

$$
\mu_0
=
V_{\mathrm{trap}}(x)
+
\mu_{\mathrm{HR}}\left(n_{\mathrm{LDA}}(x)\right),
$$

with normalization:

$$
\int n_{\mathrm{LDA}}(x)\,dx = N.
$$

Total LDA energy:

$$
E_{\mathrm{LDA}}
=
\int
\left[
\epsilon_{\mathrm{HR}}\left(n_{\mathrm{LDA}}(x)\right)
+
V_{\mathrm{trap}}(x)n_{\mathrm{LDA}}(x)
\right]dx.
$$

LDA cloud radius:

$$
R_{\mathrm{LDA}}
=
\sqrt{
\frac{1}{N}
\int (x-x_0)^2 n_{\mathrm{LDA}}(x)\,dx
}.
$$

Source basis:
`Primary physics`, LDA precedent from `[Astrakharchik2005LDA]`, hard-rod EOS
from `[Mazzanti2008HardRods]`.

Claim boundary:
LDA is the theory prediction being tested. It is not QMC data and not the
benchmark.

### T5. Grid Integration And LDA Support Edges

Code:
[src/hrdmc/theory/lda.py](src/hrdmc/theory/lda.py)

Formula:

Grid integrals use trapezoidal quadrature:

$$
\int f(x)\,dx
\approx
\mathrm{trapz}(f,x).
$$

LDA support edges are threshold diagnostics:

$$
x_{\mathrm{left/right}}
=
\min/\max\{x_i: n_{\mathrm{LDA}}(x_i)>\epsilon\}.
$$

Source basis:
`Repo convention`, numerical quadrature and diagnostic support extraction.

Claim boundary:
Support edges are rough grid diagnostics. Normalization validates particle
count on the chosen grid; it does not prove LDA validity.

## Wavefunctions And Guides

### W1. Homogeneous All-Pair Reduced Hard-Rod Trial

Code:
[src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py)

Formula:

$$
y_i = x_i - ia,
\qquad
L_{\mathrm{eff}}=L-Na,
$$

$$
\Psi_T(\mathbf{x})
\propto
\prod_{i<j}
\sin^p\left[
\frac{\pi(y_j-y_i)}{L_{\mathrm{eff}}}
\right].
$$

For `power=1` in the homogeneous ring validation:

$$
E_L(\mathbf{x})=
-\frac{\nabla^2\Psi_T}{\Psi_T}
=
E_N.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
The all-pair form is controlled for homogeneous validation. It is not a trapped
exact wavefunction.

### W2. Nearest-Neighbor Ring Smoke Trial

Code:
[src/hrdmc/wavefunctions/jastrow.py](src/hrdmc/wavefunctions/jastrow.py)

Formula:

$$
\Psi_{\mathrm{NN}}
\propto
\prod_i
\sin^p\left[
\frac{\pi(g_i-a)}{L_{\mathrm{eff}}}
\right],
$$

where \(g_i\) are nearest-neighbor ring gaps.

Source basis:
`Repo convention`.

Claim boundary:
Smoke-test scaffold only. It must not be cited as a paper hard-rod trial.

### W3. Trapped VMC Diagnostic Trial

Code:
[src/hrdmc/wavefunctions/trapped.py](src/hrdmc/wavefunctions/trapped.py)

Formula:

$$
\Psi_T(\mathbf{x})
\propto
\exp\left[
-\frac{\alpha}{2}\sum_i(x_i-x_0)^2
\right]
\prod_i (x_{i+1}-x_i-a)^p.
$$

Source basis:
`Repo convention`, motivated by hard-core constraints and harmonic confinement.

Claim boundary:
VMC diagnostic trial only. It is not a final trapped benchmark wavefunction.

### W4. DMC Guide Protocol

Code:
[src/hrdmc/wavefunctions/guides.py](src/hrdmc/wavefunctions/guides.py)

Formula:

$$
\Psi_T,
\quad
\nabla\log\Psi_T,
\quad
\nabla^2\log\Psi_T,
\quad
E_L
=
\frac{\hat H\Psi_T}{\Psi_T}.
$$

Source basis:
`Method paper`, importance-sampled DMC conventions from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`.

Claim boundary:
Protocol only. It defines what a DMC guide must provide.

### W5. Reduced TG-Like Trapped DMC Guide

Code:
[src/hrdmc/wavefunctions/trapped_guides.py](src/hrdmc/wavefunctions/trapped_guides.py)

Formula:

$$
y_i=x_i-a\left(i-\frac{N-1}{2}\right),
$$

$$
\Psi_T(\mathbf{x})
\propto
\exp\left[
-\frac{\alpha}{2}\sum_i(y_i-x_0)^2
\right]
\prod_{i<j}(y_j-y_i)^p.
$$

Local energy uses:

$$
E_L
=
-\sum_i
\left[
\partial_i^2\log\Psi_T
+
\left(\partial_i\log\Psi_T\right)^2
\right]
+
V_{\mathrm{trap}}(\mathbf{x}).
$$

Source basis:
`Primary physics` context from `[Mazzanti2008HardRods]` and
`[GirardeauAstrakharchik2010TrappedHardSphere]`; `Method paper` local-energy
usage from `[Foulkes2001QMC]`.

Claim boundary:
Candidate DMC guide. Quality must be validated by variance, stationarity,
timestep, and population controls.

## Monte Carlo Engines

### M1. Metropolis VMC

Code:
[src/hrdmc/monte_carlo/vmc.py](src/hrdmc/monte_carlo/vmc.py)

Formula:

$$
A(\mathbf{x}\to\mathbf{x}')
=
\min\left[
1,
\exp\left(2[\log\Psi_T(\mathbf{x}')-\log\Psi_T(\mathbf{x})]\right)
\right].
$$

Source basis:
`Method paper`, VMC/Metropolis convention summarized in `[Foulkes2001QMC]`.

Claim boundary:
VMC is a diagnostic baseline unless separately validated for a specific claim.

### M2. DMC Contract Helpers

Code:
[src/hrdmc/monte_carlo/dmc/contracts.py](src/hrdmc/monte_carlo/dmc/contracts.py)

Formula:

$$
\tilde w_i=\frac{w_i}{\sum_j w_j}.
$$

Systematic resampling uses equally spaced cumulative-weight positions.

Source basis:
`Method paper`, population/weight-control context from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`.

Claim boundary:
Contract layer only. It is not a concrete DMC benchmark by itself.

### M3. RN-Corrected Collective-Block DMC

Code:
[src/hrdmc/monte_carlo/dmc/rn_block/](src/hrdmc/monte_carlo/dmc/rn_block/)

Local DMC proposal in repo units:

$$
\mathbf{X}'
=
\mathbf{X}
+
2\Delta\tau\nabla\log\Psi_T(\mathbf{X})
+
\sqrt{2\Delta\tau}\,\boldsymbol{\eta},
\qquad
\boldsymbol{\eta}\sim\mathcal{N}(0,I).
$$

Local weight increment:

$$
\Delta\log W_{\mathrm{local}}
=
-\Delta\tau
\left[
\frac{E_L(\mathbf{X})+E_L(\mathbf{X}')}{2}
-
E_{\mathrm{ref}}
\right].
$$

Collective block proposal:

$$
\mathbf{Y}=\Phi_q(\mathbf{X}),
\qquad
\mathbf{Y}\sim Q_\theta(\mathbf{Y}\mid\mathbf{X}).
$$

RN correction:

$$
\Delta\log W_{\mathrm{RN}}
=
\log K_{\Delta\tau}^{\mathrm{sys}}(\mathbf{Y}\mid\mathbf{X})
-
\log Q_\theta(\mathbf{Y}\mid\mathbf{X}).
$$

Importance-sampled RN correction:

$$
\Delta\log W_{\mathrm{RN,IS}}
=
\log K_{\Delta\tau}^{\mathrm{sys}}(\mathbf{Y}\mid\mathbf{X})
-
\log Q_\theta(\mathbf{Y}\mid\mathbf{X})
+
\log\Psi_T(\mathbf{Y})
-
\log\Psi_T(\mathbf{X}).
$$

Source basis:
`Method paper`, DMC/importance-sampling basis from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`; RN identity is the standard
Radon-Nikodym/importance-sampling density-ratio identity; non-crossing target
kernel component uses `[KarlinMcGregor1959]`.

Claim boundary:
Candidate engine. It is DMC, but paper-level benchmark status requires
timestep, population, stationarity, accounting, and archived run-artifact gates.

## Estimators

### E1. Density Profiles

Code:
[src/hrdmc/estimators/density.py](src/hrdmc/estimators/density.py)

Formula:

$$
n(x_b)
=
\frac{\mathrm{count\ in\ bin}\ b}{N_{\mathrm{samples}}\Delta x}.
$$

For weighted DMC samples:

$$
n(x_b)
=
\frac{1}{\Delta x}
\sum_{s,j}
\tilde w_s\,
\mathbf{1}[x_{s,j}\in b].
$$

Source basis:
`Method paper`, standard coordinate estimator convention in QMC workflows
summarized by `[Foulkes2001QMC]`.

Claim boundary:
Histogram normalization convention. Density integrates to \(N\) only when the
grid captures all particles and weights are normalized.

### E1b. Density Support Edges

Code:
[src/hrdmc/estimators/density.py](src/hrdmc/estimators/density.py)

Formula:

$$
x_{\mathrm{left/right}}
=
\min/\max\{x_b: n(x_b)>\epsilon\}.
$$

Source basis:
`Repo convention`.

Claim boundary:
Rough occupied-bin diagnostic only. It is not a physical cloud boundary unless
the density threshold, grid, and estimator tier are explicitly justified.

### E2. Pair Distribution Function

Code:
[src/hrdmc/estimators/pair_distribution.py](src/hrdmc/estimators/pair_distribution.py)

Formula:

$$
g(r_b)
=
\frac{\mathrm{pair\ count\ in\ bin}\ b}
{N_{\mathrm{samples}}N\rho\Delta r}.
$$

Source basis:
`Primary physics`, observable used in `[Mazzanti2008HardRods]`; normalization is
the repository finite-ring convention.

Claim boundary:
Mainly homogeneous validation. Trapped local pair analysis would need a separate
normalization.

### E3. Static Structure Factor

Code:
[src/hrdmc/estimators/structure_factor.py](src/hrdmc/estimators/structure_factor.py)

Formula:

$$
\rho_k=\sum_j e^{ikx_j},
\qquad
S(k)=\frac{\langle |\rho_k|^2\rangle}{N}.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Claim boundary:
Uses physical wrapped particle positions \(x_j\), not reduced coordinates.

### E4. Local Energy Estimators

Code:
[src/hrdmc/estimators/local_energy.py](src/hrdmc/estimators/local_energy.py)

Formula:

$$
E_L
=
\frac{\hat H\Psi_T}{\Psi_T}
=
-\sum_i
\left[
\partial_i^2\log\Psi_T
+
\left(\partial_i\log\Psi_T\right)^2
\right]
+
V.
$$

Source basis:
`Method paper`, local-energy convention from `[Foulkes2001QMC]`.

Claim boundary:
The homogeneous all-pair ring case is a validation benchmark. Trapped VMC local
energy is diagnostic unless backed by DMC/external validation.

### E5. Cloud Radius

Code:
[src/hrdmc/estimators/cloud.py](src/hrdmc/estimators/cloud.py)

Formula:

$$
R^2
=
\frac{1}{N}
\sum_i(x_i-x_0)^2,
\qquad
R_{\mathrm{rms}}=\sqrt{\langle R^2\rangle}.
$$

Source basis:
`Primary physics`, trapped-cloud observable context from
`[AstrakharchikGiorgini2002TrappedCrossover]`; hard-rod trapped context from
`[GirardeauAstrakharchik2010TrappedHardSphere]`.

Claim boundary:
For DMC, current weighted RMS is a mixed-distribution candidate
observable unless a pure-estimator path is added.

### E6. Weighted DMC Observables

Code:
[src/hrdmc/estimators/weighted.py](src/hrdmc/estimators/weighted.py)

Formula:

$$
\langle O\rangle_w
=
\sum_s \tilde w_s O(\mathbf{X}_s),
\qquad
N_{\mathrm{eff}}
=
\frac{1}{\sum_s \tilde w_s^2}.
$$

Source basis:
`Method paper`, weighted Monte Carlo and DMC estimator convention from
`[Foulkes2001QMC]`.

Claim boundary:
The estimator filters non-finite, invalid, and non-positive-weight samples.
Energy is a mixed estimator. Density/RMS are mixed-distribution candidate
observables unless pure estimation is implemented.

## Analysis And Validation Metrics

### A1. Blocking Standard Error

Code:
[src/hrdmc/analysis/blocking.py](src/hrdmc/analysis/blocking.py)

Formula:

Block a correlated series into block means and estimate:

$$
\mathrm{SE}
=
\frac{\mathrm{std}(\bar x_{\mathrm{block}})}
{\sqrt{N_{\mathrm{blocks}}}}.
$$

Source basis:
`Method paper`, `[FlyvbjergPetersen1989Blocking]`.

Claim boundary:
Users must inspect plateau behavior. A single blocking number is not a full
validation proof.

### A2. Correlated Monte Carlo Error Triangulation

Code:
[src/hrdmc/analysis/correlated_error.py](src/hrdmc/analysis/correlated_error.py)

Formula:

For a correlated production trace \(x_t\), the standard error of the sample
mean is estimated from the spectral density at zero:

$$
\mathrm{SE}(\bar x)
=
\sqrt{\frac{S(0)}{N}}
=
\sqrt{\frac{\sigma_x^2\,2\tau_{\mathrm{int}}}{N}}.
$$

The Sokal-window estimator uses

$$
\tau_{\mathrm{int}}
=
\frac12+\sum_{t=1}^{W}\rho_t,
\qquad
W \ge c\,\tau_{\mathrm{int}},
$$

with the repository default inherited from
[src/hrdmc/analysis/timeseries.py](src/hrdmc/analysis/timeseries.py). The
Geyer initial-sequence estimator uses autocovariance pairs

$$
\Gamma_k=\gamma_{2k}+\gamma_{2k+1},
\qquad
\tau_{\mathrm{int}}
=
-\frac12+\frac{1}{\gamma_0}\sum_{k=0}^{m}\Gamma_k,
$$

truncated at the initial positive monotone sequence. The HAC estimator uses a
flat-top lag window:

$$
\hat S(0)
=
\gamma_0+2\sum_{t=1}^{b}K(t/b)\gamma_t,
\qquad
K(u)=
\begin{cases}
1,& |u|\le 1/2,\\
2(1-|u|),& 1/2<|u|\le 1,\\
0,& |u|>1.
\end{cases}
$$

The stationarity workflow computes all three estimates per seed. If at least
two standard-error estimates overlap within their estimated one-sigma
uncertainty, the seed status is `TRIANGULATED_2_OF_3`; otherwise the seed is
reported as `DISAGREE_HONEST_LARGE`. The case standard error is conservative:

$$
\mathrm{SE}_{\mathrm{corr,case}}
=
\frac{
\left[\sum_s \mathrm{SE}_{\mathrm{corr},s}^2\right]^{1/2}
}{M},
\qquad
\mathrm{SE}_{\mathrm{reported}}
=
\max(\mathrm{SE}_{\mathrm{seed}},
     \mathrm{SE}_{\mathrm{block}},
     \mathrm{SE}_{\mathrm{corr,case}}).
$$

Source basis:
`Method paper`; Sokal integrated-autocorrelation window from `[Sokal1997MC]`,
Geyer initial-sequence variance estimation from `[Geyer1992PracticalMCMC]`,
HAC long-run variance from `[Andrews1991HAC]`, and the flat-top lag window from
`[PolitisRomano1995FlatTop]`.

Claim boundary:
This replaces "blocking plateau missing" as a hard precision veto only when the
methodology gates are already clean: finite/hard-core hygiene, density
accounting, RN-weight gate, R-hat, effective-sample, and explicit
trace-stationarity checks still hard-fail. Missing blocking plateau becomes a
precision warning with an explicitly inflated error bar; it is not a way to
manufacture a benchmark GO.

### A3. Bias And MSE

Code:
[src/hrdmc/analysis/metrics.py](src/hrdmc/analysis/metrics.py)

Formula:

$$
\mathrm{bias}=\hat\theta-\theta_{\mathrm{ref}},
\qquad
\mathrm{MSE}=\mathrm{bias}^2+\mathrm{variance}.
$$

Source basis:
`Method paper`, standard statistical error definitions used in QMC validation
workflows; QMC benchmark context from `[Foulkes2001QMC]`.

Claim boundary:
Needs a trustworthy reference \(\theta_{\mathrm{ref}}\). LDA is not the
benchmark reference.

### A4. Density L2 Metrics

Code:
[src/hrdmc/analysis/metrics.py](src/hrdmc/analysis/metrics.py)

Raw squared density L2:

$$
D_n^{(2)}
=
\int
\left|
n_{\mathrm{benchmark}}(x)
-
n_{\mathrm{LDA}}(x)
\right|^2 dx.
$$

Relative density L2:

$$
\delta n_2
=
\frac{
\left[
\int |n_{\mathrm{benchmark}}(x)-n_{\mathrm{LDA}}(x)|^2 dx
\right]^{1/2}
}{
\left[
\int |n_{\mathrm{LDA}}(x)|^2 dx
\right]^{1/2}
}.
$$

Source basis:
`Repo convention`, standard L2 norm comparison; thesis use is QMC-vs-LDA error
mapping.

Claim boundary:
This metric is mathematically valid as a normed discrepancy. It does not decide
which curve is true; that depends on benchmark tier.

### A5. Replicate Stability

Code:
[src/hrdmc/analysis/stability.py](src/hrdmc/analysis/stability.py)

Formula:

$$
\bar x = \frac{1}{M}\sum_m x_m,
\qquad
s = \sqrt{\frac{1}{M-1}\sum_m(x_m-\bar x)^2},
\qquad
\mathrm{SE}=\frac{s}{\sqrt M}.
$$

Source basis:
`Method paper`, standard replicate summary convention; QMC validation context
from `[Foulkes2001QMC]`.

Claim boundary:
Replicate spread is a diagnostic. It is not a substitute for timestep,
population, and stationarity controls.

### A6. Time-Series Stationarity Diagnostics

Code:
[src/hrdmc/analysis/timeseries.py](src/hrdmc/analysis/timeseries.py)

Formula:

The code estimates an autocorrelation function, an integrated autocorrelation
time, and an effective independent sample count:

$$
N_{\mathrm{eff}}
\approx
\frac{N}{2\tau_{\mathrm{int}}}.
$$

It then uses autocorrelation-adjusted slope and cumulative-drift diagnostics on
production traces.

For multiple independent seed traces, the stationarity report also computes the
standard potential-scale-reduction diagnostic. With within-chain variance \(W\),
between-chain variance \(B\), \(M\) chains, and \(N\) samples per chain:

$$
\widehat{\mathrm{var}}^+
=
\frac{N-1}{N}W+\frac{1}{N}B,
\qquad
\hat R
=
\sqrt{\frac{\widehat{\mathrm{var}}^+}{W}}.
$$

The implementation uses this as a conservative chain-agreement diagnostic, not
as a Bayesian posterior diagnostic.

For a fitted production-tail slope \(b\) with naive regression standard error
\(\sigma_b\), the autocorrelation-adjusted slope statistic is

$$
z_{\mathrm{slope}}
=
\frac{|b|}
{\sigma_b\sqrt{N/N_{\mathrm{eff}}}}.
$$

The first/last quarter drift uses blocking standard errors:

$$
z_{1/4}
=
\frac{
|\bar x_{\mathrm{last\ quarter}}-\bar x_{\mathrm{first\ quarter}}|
}{
\sqrt{\mathrm{SE}_{\mathrm{block}}(x_{\mathrm{first\ quarter}})^2
+\mathrm{SE}_{\mathrm{block}}(x_{\mathrm{last\ quarter}})^2}
}.
$$

The late cumulative drift diagnostic is

$$
z_{\mathrm{late}}
=
\frac{
|\bar x_{1:N}-\bar x_{1:\lfloor 0.75N\rfloor}|
}{
\mathrm{SE}_{\mathrm{block}}(x_{1:N})
}.
$$

The four-block diagnostic compares block means \(B_k\):

$$
z_{\mathrm{block}}
=
\frac{
|\bar x_{B_4}-\bar x_{B_1}|
}{
\sqrt{
\mathrm{SE}_{\mathrm{block}}(B_1)^2+
\mathrm{SE}_{\mathrm{block}}(B_4)^2
}
},
\qquad
z_{\mathrm{spread}}
=
\frac{
\max_k \bar x_{B_k}-\min_k \bar x_{B_k}
}{
\mathrm{SE}_{\mathrm{block}}(x_{1:N})
}.
$$

The stationarity gate is:

$$
z_{\mathrm{slope}}\le 2,\qquad
z_{1/4}\le 2.5,\qquad
z_{\mathrm{late}}\le 2,\qquad
z_{\mathrm{block}}\le 2.5.
$$

\(z_{\mathrm{spread}}>4\) is retained as a warning. It is a range statistic, not
a trend test, so it becomes a veto only when paired with a slope, cumulative, or
first/last-block failure.

Case-level uncertainty reporting uses a conservative standard error:

$$
\mathrm{SE}_{\mathrm{block,combined}}
=
\frac{
\left[\sum_s \mathrm{SE}_{\mathrm{block}}(x_s)^2\right]^{1/2}
}{M},
\qquad
\mathrm{SE}_{\mathrm{reported}}
=
\max\left(
\mathrm{SE}_{\mathrm{seed}},
\mathrm{SE}_{\mathrm{block,combined}},
\mathrm{SE}_{\mathrm{corr,case}}
\right),
$$

where \(s\) indexes independent seeds and \(M\) is the seed count. If any
observable has spread warnings, missing blocking plateaus, or correlated-error
inflation, the case artifact reports a precision warning rather than silently
trusting the seed standard error.

Source basis:
`Method paper`, autocorrelation/error-control logic aligned with
`[FlyvbjergPetersen1989Blocking]` and A2; chain-agreement context from
`[VehtariGelmanSimpsonCarpenterBuerkner2021Rhat]`; DMC validation need from
`[Foulkes2001QMC]`.

Claim boundary:
Gate-support analysis only. It is not a physics source and does not make DMC
correct without timestep/population/accounting validation.

## Experiment Scripts

Experiment scripts under `experiments/` are orchestration surfaces. They should
call owners above without owning formulas. If an experiment introduces a new
formula, it must be added to this source map before being used for a thesis or
paper claim.

## Current Missing Source-Map Items

No `src/hrdmc` physics/method formula is intentionally left unmapped. Generated
run scripts and archived artifacts are not covered here; package formulas must
be added to this document before they are used for thesis or paper claims.
