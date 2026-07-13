# Equation and Method Source Map

This document is the audit map for formulas and method conventions in
`src/hrdmc`. It answers four questions for each scientific formula:

```text
What is the formula?
Where is it implemented?
What is the source basis?
What has been validated, and under which assumptions?
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
  code, separate from thesis physics results.
- `Repo convention`: implementation normalization or utility formula. These
  require an added source before presentation as literature statements.

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
- `[Girardeau1960TG]` M. Girardeau, *Relationship between systems of
  impenetrable bosons and fermions in one dimension*, **J. Math. Phys. 1**,
  516-523 (1960).
  DOI: [10.1063/1.1703687](https://doi.org/10.1063/1.1703687)
- `[GirardeauWrightTriscari2001TrappedTG]` M. D. Girardeau, E. M. Wright, and
  J. M. Triscari, *Ground-state properties of a one-dimensional system of
  hard-core bosons in a harmonic trap*, **Phys. Rev. A 63**, 033601 (2001).
  DOI: [10.1103/PhysRevA.63.033601](https://doi.org/10.1103/PhysRevA.63.033601)
- `[GirardeauAstrakharchik2010TrappedHardSphere]` M. D. Girardeau and
  G. E. Astrakharchik, *Wave functions of the super-Tonks-Girardeau gas and the
  trapped one-dimensional hard-sphere Bose gas*, **Phys. Rev. A 81**,
  061601(R) (2010).
  DOI: [10.1103/PhysRevA.81.061601](https://doi.org/10.1103/PhysRevA.81.061601)

### QMC And Statistics Methods

- `[Billingsley1995Probability]` P. Billingsley, *Probability and Measure*,
  3rd ed., Wiley (1995). Used here for the Radon-Nikodym derivative and
  change-of-measure convention underlying likelihood-ratio reweighting.
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
- `[Doob1957HTransform]` J. L. Doob, *Conditional Brownian motion and the
  boundary limits of harmonic functions*, **Bull. Soc. Math. France 85**,
  431-458 (1957).
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
- `[CasullerasBoronat1995Pure]` J. Casulleras and J. Boronat,
  *Unbiased estimators in quantum Monte Carlo methods: Application to liquid
  4He*, **Phys. Rev. B 52**, 3654-3661 (1995).
  DOI: [10.1103/PhysRevB.52.3654](https://doi.org/10.1103/PhysRevB.52.3654)
- `[SarsaBoronatCasulleras2002Pure]` A. Sarsa, J. Boronat, J. Casulleras,
  *Quadratic diffusion Monte Carlo and pure estimators for atoms*,
  **J. Chem. Phys. 116**, 5956-5962 (2002).

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

## Report-Facing Source Usage Summary

This table is the compact source audit used by the report. It records each
source's role and range of applicability.

| source | used for in this repository/report | validation scope |
| --- | --- | --- |
| `[Mazzanti2008HardRods]` | hard-rod potential, reduced length \(L'=L-Na\), homogeneous exact ring reference, homogeneous EOS | homogeneous exactness and EOS input for the trapped LDA |
| `[AstrakharchikGiorgini2002TrappedCrossover]` | important trapped-QMC precedent for energies/radii and QMC-vs-LDA-style comparisons | trapped-QMC context for the comparison strategy |
| `[Astrakharchik2005LDA]` | LDA equation \(\mu=\mu_{\rm hom}(n)+V_{\rm ext}\), normalization, LDA density/energy/radius construction | smooth LDA reference built from the homogeneous EOS |
| `[Girardeau1960TG]`, `[GirardeauWrightTriscari2001TrappedTG]` | Tonks-Girardeau mapping and \(a=0\) trapped TG harmonic-trap exact-anchor logic | exact \(a=0\) impenetrable point-boson anchor |
| `[GirardeauAstrakharchik2010TrappedHardSphere]` | trapped hard-sphere/super-Tonks context and finite-\(a\), \(N=2\) COM-relative reference motivation | finite-\(a\), \(N=2\) anchor motivation |
| `[Foulkes2001QMC]` | importance-sampled DMC, local energy, mixed estimator conventions | general DMC method source |
| `[UmrigarNightingaleRunge1993DMC]` | short-time DMC, drift-diffusion, timestep-bias caution | DMC timestep and drift-diffusion method source |
| `[Billingsley1995Probability]` | Radon-Nikodym/change-of-measure \(K/Q\) weighting | proposal-to-target change-of-measure basis |
| `[KarlinMcGregor1959]` | ordered non-crossing determinant kernels | ordered-kernel basis where the assumptions match |
| `[Doob1957HTransform]` | ground-state \(h\)-transform identity for normalized gap kernels | one-gap ground-state transform identity |
| `[CasullerasBoronat1995Pure]`, `[SarsaBoronatCasulleras2002Pure]` | mixed-vs-pure estimator split and forward-walking descendants | estimator-method basis; implementation still needs lag-zero and plateau checks |
| `[FlyvbjergPetersen1989Blocking]`, `[Sokal1997MC]`, `[Geyer1992PracticalMCMC]`, `[Andrews1991HAC]`, `[PolitisRomano1995FlatTop]` | correlated-error, autocorrelation, \(N_{\rm eff}\), and error-bar triangulation | statistical precision layer |
| `[Hellmann1937Quantenchemie]`, `[Feynman1939Forces]` | optional energy-response route to pure \(R^2\) | controlled energy-derivative route for \(R^2\) |
| `[LiebLiniger1963GroundState]`, `[Lieb1963ExcitationSpectrum]`, `[Astrakharchik2005Quasi1DHardRods]` | optional background for future extensions and strongly correlated 1D context | future-extension background |

## Units

The trapped repository convention is harmonic-oscillator units:

$$
q=\frac{x}{a_{\rm ho}},
\qquad
\widetilde E=\frac{E}{\hbar\omega_{\rm trap}},
\qquad
a_{\rm ho}=\sqrt{\frac{\hbar}{m\omega_{\rm trap}}}.
$$

Therefore kinetic local-energy terms use:

$$
T_{\mathrm{local}}
=
-\frac12\sum_i
\left[
\partial_i^2\log\Psi_T
+
\left(\partial_i\log\Psi_T\right)^2
\right].
$$

Default trapped cases are parameterized by \(N\) and \(A=a/a_{\rm ho}\), with
no hidden code-frequency conversion.

## Estimator Response Formulas

### E1. Hellmann-Feynman Trap R2/RMS Response

Code:
[src/hrdmc/estimators/pure/energy_response.py](src/hrdmc/estimators/pure/energy_response.py)

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
R_{\rm RMS}
=
\sqrt{R_2^{\rm pure}}.
$$

Source basis:
Hellmann-Feynman theorem from `[Hellmann1937Quantenchemie]` and
`[Feynman1939Forces]`.

Validation scope:
This estimates trap \(R_2\)/RMS from local-DMC energy responses. It does not
estimate density. Every energy point in the finite-difference stencil needs its
own timestep, population, stationarity, and precision evidence.

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

Validation scope:
This is geometry and boundary-condition ownership only. It does not own EOS,
LDA, or DMC benchmark results.

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

Validation scope:
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
The harmonic trap is a standard trapped-gas model. 
`[AstrakharchikGiorgini2002TrappedCrossover]` is an important trapped-QMC
precedent for computing trapped-gas energies and cloud radii and comparing
them with mean-field/LDA-type descriptions in a three- to one-dimensional
crossover setting. The present hard-rod scope uses the strictly
one-dimensional finite-length hard-rod Hamiltonian and the excluded-volume LDA
built from the hard-rod equation of state. The zero and cosine potentials are
`Repo convention` utilities.

Validation scope:
The thesis trapped hard-rod path uses the harmonic trap. The cosine potential
has no thesis physics role without a separate source and use case.

### S4. Reduced Hard-Rod Length

Code:
[src/hrdmc/systems/reduced.py](src/hrdmc/systems/reduced.py)

Formula:

$$
L_{\mathrm{eff}} = L - Na.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Validation scope:
This geometry identity is used by ring theory and ring trial states. It is
separate from the LDA solver and from trapped exact solutions.

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

Validation scope:
`systems/` owns the target-kernel interface. The optional collective RN
extension consumes it through that interface instead of rebuilding Hamiltonian
physics.

### S6. Harmonic Mehler Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula implemented:

$$
\gamma = \omega,
\qquad
m_\gamma = \omega,
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

Validation scope:
Exact for the one-body harmonic kernel in harmonic-oscillator units. The
many-body trapped hard-rod propagator requires the hard-core ordering/domain
layer as well.

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
harmonic-oscillator units is:

$$
E_0
=
\sum_{n=0}^{N-1}\left(n+\frac12\right)\omega
=
\frac{N^2\omega}{2}.
$$

Source basis:
`Analytic identity`, harmonic oscillator spectrum plus the ordered
Karlin-McGregor determinant `[KarlinMcGregor1959]`, TG mapping from
`[Girardeau1960TG]`, trapped harmonic TG exact-reference context from
`[GirardeauWrightTriscari2001TrappedTG]`, and trapped hard-core/TG context from
`[GirardeauAstrakharchik2010TrappedHardSphere]`.

Validation scope:
This is an exact validation anchor only for \(a=0\) in a harmonic trap. It
validates the zero-rod-length trapped limit.

### S8. Free Ordered Hard-Rod Kernel

Code:
[src/hrdmc/systems/propagators.py](src/hrdmc/systems/propagators.py)

Formula:

$$
u_i=x_i-a\left[(i-1)-\frac{N-1}{2}\right],
\qquad i=1,\ldots,N,
$$

$$
p_\tau(z)
=
\frac{1}{\sqrt{2\pi\tau}}
\exp\left[-\frac{z^2}{2\tau}\right],
$$

$$
K_{\mathrm{free,ordered}}(\mathbf{u}'\mid\mathbf{u})
=
\det\left[p_\tau(u'_j-u_i)\right]_{i,j=1}^{N}.
$$

Source basis:
`Method paper`, `[KarlinMcGregor1959]` determinant logic for non-crossing
paths, applied to reduced hard-rod coordinates from `[Mazzanti2008HardRods]`.

Validation scope:
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

Validation scope:
Approximate short-time target density. It requires timestep validation before
use in a finite-rod trapped comparison.

### S10. Optional Collective Gap H-Transform Proposal

Code:
[src/hrdmc/systems/gap_h_transform.py](src/hrdmc/systems/gap_h_transform.py)

Formula:

For open-line coordinates define the center of mass and nearest-neighbor gaps

$$
X_{\mathrm{cm}}=\frac{1}{N}\sum_i(x_i-x_0),
\qquad
g_i=x_{i+1}-x_i.
$$

The proposal factorizes as

$$
Q_{\mathrm{gap-h}}(\mathbf{x}'\mid\mathbf{x};\tau)
=
Q_{\mathrm{cm}}(X_{\mathrm{cm}}'\mid X_{\mathrm{cm}};\tau)
\,
\prod_{i=1}^{N-1}
Q_{\mathrm{gap}}(g_i'\mid g_i;\tau),
$$

where \(Q_{\mathrm{cm}}\) is the exact harmonic COM h-transform,

$$
X_{\mathrm{cm}}'=\rho X_{\mathrm{cm}} + \sqrt{\sigma_Q^2(1-\rho^2)}\,\eta,
\qquad
\rho=e^{-\omega\tau},
\qquad
\sigma_Q^2=\frac{1}{2N\omega},
$$

and \(Q_{\mathrm{gap}}\) is the N=2 relative hard-wall harmonic h-transform
on \(g\ge a\),

$$
Q_{\mathrm{gap}}(g'\mid g;\tau)
=
\frac{
K_{\mathrm{rel,D}}^{(a)}(g'\mid g;\tau)\psi_0(g')
}{
e^{-E_0\tau}\psi_0(g)
},
$$

with

$$
H_{\mathrm{rel}}
=
-2\frac{d^2}{dg^2}
+
\frac{1}{4}\omega^2 g^2,
\qquad
g\ge a,
\qquad
\psi(a)=0.
$$

The implementation reconstructs positions from \(X'_{\mathrm{cm}}\) and the
sampled gaps. This is the sampleable proposal family \(Q\). When the optional
collective extension is enabled, its target density \(K\) is selected
separately by the workflow.

Source basis:
`Analytic identity`, harmonic COM separation, Dirichlet relative-coordinate
h-transform from `[Doob1957HTransform]`, and importance-sampled DMC proposal
logic from `[Foulkes2001QMC]` and `[UmrigarNightingaleRunge1993DMC]`.

Validation scope:
This entry maps the proposal family. For \(N=2\) it uses the finite-grid
relative h-transform. For \(N>2\), the independent nearest-neighbor gap product
is a sampling approximation. A run using this proposal additionally requires
the selected target, correction-weight, cadence, local-baseline, timestep,
population, stationarity, density-accounting, and precision checks.

### S10b. Optional Collective Gap-H Product Target

Code:
[src/hrdmc/systems/gap_h_transform.py](src/hrdmc/systems/gap_h_transform.py)

Formula:

This optional target density is built from the same COM and nearest-neighbor
gap h-transform factors. It converts the normalized h-transform product back to
a raw kernel:

$$
K_{\mathrm{product}}(\mathbf{x}'\mid\mathbf{x};\tau)
=
p_h(\mathbf{x}'\mid\mathbf{x};\tau)
\exp[-\tau E_{\mathrm{product}}]
\frac{\psi_{\mathrm{product}}(\mathbf{x})}
     {\psi_{\mathrm{product}}(\mathbf{x}')}.
$$

In implementation terms,

$$
\log K_{\mathrm{product}}
=
\log p_h
-
\tau E_{\mathrm{product}}
+
\log\psi_{\mathrm{product}}(\mathbf{x})
-
\log\psi_{\mathrm{product}}(\mathbf{x}').
$$

For \(N=2\), the COM and one-gap relative-coordinate Hamiltonian separate
exactly, so this target reduces to the deterministic finite-\(a\) reference.
For \(N>2\), it is a nearest-neighbor product target:

$$
K_{\mathrm{product}}
\approx
K_{\mathrm{cm}}
\prod_{i=1}^{N-1}K_{\mathrm{gap}}(g_i'\mid g_i).
$$

Source basis:
`Analytic identity`, N=2 COM/relative separation, Dirichlet h-transform from
`[Doob1957HTransform]`, and DMC target-kernel usage from `[Foulkes2001QMC]` /
`[UmrigarNightingaleRunge1993DMC]`.

Validation scope:
The target is exact for the finite-\(a\), \(N=2\) one-gap trapped problem. For
\(N>2\), this product target is an approximate many-body kernel and requires
cadence, timestep, and population validation before comparison use.

### S11. Gap-H-Corrected Trapped Guide

Code:
[src/hrdmc/wavefunctions/guides/gap_h.py](src/hrdmc/wavefunctions/guides/gap_h.py)
[src/hrdmc/wavefunctions/kernels/gap_h.py](src/hrdmc/wavefunctions/kernels/gap_h.py)

Formula:

The reduced TG guide uses

$$
u_i=x_i-a\left[(i-1)-\frac{N-1}{2}\right],
\qquad i=1,\ldots,N,
\qquad
\Psi_{\mathrm{TG}}
=
\exp\left[-\frac{\alpha}{2}\sum_i(u_i-x_0)^2\right]
\prod_{i<j}(u_j-u_i).
$$

For every nearest-neighbor physical gap \(g_i=x_{i+1}-x_i\), the matched
gap-H guide multiplies the TG backbone by the N=2 relative correction

$$
\Psi_{\mathrm{gapH}}
=
\Psi_{\mathrm{TG}}
\prod_{i=1}^{N-1}
\frac{h_2(g_i)}{h_{\mathrm{TG}}(g_i)},
\qquad
h_{\mathrm{TG}}(g)=(g-a)\exp\left[-\frac{\alpha}{4}(g-a)^2\right],
$$

where \(h_2(g)\) is the positive finite-grid ground state of

$$
H_{\mathrm{rel}}
=
-2\frac{d^2}{dg^2}
+\frac{1}{4}\omega^2g^2,
\qquad g\ge a,\qquad h_2(a)=0.
$$

For \(N=2\), this restores the exact COM-separated relative guide up to the
finite-grid interpolation. For \(N>2\), it is a nearest-neighbor product guide
ansatz matched to the same gap table as the gap-H proposal. Any collective
target family is selected separately by the workflow; this guide changes only
importance sampling, drift, guide ratio, and local energy and can be used with
ordinary local DMC.

Source basis:
`Analytic identity`, harmonic COM separation, N=2 Dirichlet relative-coordinate
ground state, h-transform context from `[Doob1957HTransform]`, and
importance-sampled DMC guide logic from `[Foulkes2001QMC]` and
`[UmrigarNightingaleRunge1993DMC]`.

Validation scope:
This entry maps the guide/proposal matching layer. Every use requires guide,
stationarity, population, timestep, density-accounting, and precision checks.
Runs that also enable collective transport require the target, correction
weight, cadence, and local-baseline checks described above.

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

Validation scope:
This is the homogeneous validation benchmark. Trapped results use the
open-line harmonic system.

### T2. Thermodynamic Hard-Rod EOS

Code:
[src/hrdmc/theory/hard_rods.py](src/hrdmc/theory/hard_rods.py)

Formula:

$$
e_{\mathrm{HR}}(\rho)
=
\frac{\pi^2\rho^2}{6(1-a\rho)^2},
$$

$$
\epsilon_{\mathrm{HR}}(\rho)
=
\rho e_{\mathrm{HR}}(\rho)
=
\frac{\pi^2\rho^3}{6(1-a\rho)^2}.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Validation scope:
Thermodynamic homogeneous EOS in harmonic-oscillator units. This is the LDA
input, separate from trapped exact solutions.

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
{6(1-a\rho)^3}.
$$

Source basis:
`Primary physics`, derivative of the EOS from `[Mazzanti2008HardRods]`.

Validation scope:
The inverse chemical potential in code is numerical bisection. The bisection is
an implementation method rather than a separate physics formula.

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

Validation scope:
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

Validation scope:
Support edges are rough grid diagnostics. Normalization validates particle
count on the chosen grid. LDA validity is assessed by comparison with sampled
observables.

## Wavefunctions And Guides

### W1. Homogeneous All-Pair Reduced Hard-Rod Trial

Code:
[src/hrdmc/wavefunctions/trials/jastrow.py](src/hrdmc/wavefunctions/trials/jastrow.py)

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

Validation scope:
The all-pair form is controlled for homogeneous validation. It is separate from
trapped exact wavefunctions.

### W2. Nearest-Neighbor Ring Diagnostic Trial

Code:
[src/hrdmc/wavefunctions/trials/jastrow.py](src/hrdmc/wavefunctions/trials/jastrow.py)

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

Validation scope:
Diagnostic scaffold only; no validated hard-rod interpretation is attached to
this form.

### W3. Trapped VMC Diagnostic Trial

Code:
[src/hrdmc/wavefunctions/trials/trapped.py](src/hrdmc/wavefunctions/trials/trapped.py)

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

Validation scope:
VMC diagnostic trial only; this trial form is not asserted to be exact for the
trapped hard-rod problem.

### W4. DMC Guide Protocol

Code:
[src/hrdmc/wavefunctions/api.py](src/hrdmc/wavefunctions/api.py)

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

Validation scope:
Protocol only. It defines what a DMC guide must provide.

### W5. Reduced TG-Like Trapped DMC Guide

Code:
[src/hrdmc/wavefunctions/guides/trapped_tg.py](src/hrdmc/wavefunctions/guides/trapped_tg.py)

Formula:

$$
y_i=x_i-a\left[(i-1)-\frac{N-1}{2}\right],
\qquad i=1,\ldots,N,
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
`Primary physics` context from `[Mazzanti2008HardRods]`, TG mapping from
`[Girardeau1960TG]`, and trapped hard-rod guide context from
`[GirardeauAstrakharchik2010TrappedHardSphere]`; `Method paper` local-energy
usage from `[Foulkes2001QMC]`.

Validation scope:
DMC guide ansatz. Quality must be validated by variance, stationarity,
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

Validation scope:
VMC is a diagnostic baseline unless separately validated for the intended
comparison.

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

Validation scope:
Contract layer only; concrete DMC benchmark status comes from the workflow
checks.

### M3. Local Importance-Sampled DMC

Code:
[src/hrdmc/monte_carlo/dmc/local/](src/hrdmc/monte_carlo/dmc/local/)

The default Metropolis-corrected drift-diffusion proposal in
harmonic-oscillator units is:

$$
\mathbf{X}'
=
\mathbf{X}
+
\Delta\tau\nabla\log\Psi_T(\mathbf{X})
+
\sqrt{\Delta\tau}\,\boldsymbol{\eta},
\qquad
\boldsymbol{\eta}\sim\mathcal{N}(0,I).
$$

For Gaussian transition density \(q\), the Metropolis acceptance probability is

$$
A(\mathbf X\to\mathbf X')
=
\min\left[
1,
\frac{\Psi_T(\mathbf X')^2 q(\mathbf X\mid\mathbf X')}
     {\Psi_T(\mathbf X)^2 q(\mathbf X'\mid\mathbf X)}
\right].
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

Source basis:
`Method paper`, DMC and importance-sampling basis from `[Foulkes2001QMC]` and
the drift/timestep treatment in `[UmrigarNightingaleRunge1993DMC]`.

Validation scope:
The engine supplies the local DMC trajectory and population bookkeeping.
Reported rows still need timestep, population, stationarity, accounting, and
observable-specific estimator checks.

### M4. Optional Collective RN Scheduled Move

Code:
[src/hrdmc/monte_carlo/dmc/collective_rn/](src/hrdmc/monte_carlo/dmc/collective_rn/)

An explicitly enabled scheduled extension can propose

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

This is the discrete Markov-kernel Radon-Nikodym derivative. For a trajectory
generated under proposal \(Q\), the target-kernel path measure contributes the
likelihood ratio

$$
\frac{d\mathbb{P}_K}{d\mathbb{P}_Q}
=
\prod_t
\frac{K(x_{t+1}\mid x_t)}
     {Q(x_{t+1}\mid x_t)}.
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
`[UmrigarNightingaleRunge1993DMC]`; Radon-Nikodym/change-of-measure convention
from `[Billingsley1995Probability]`. Target-specific kernel sources are mapped
by the system-kernel entries above, for example S9 for the primitive target and
S10b for the optional gap-h-product target.

Validation scope:
This extension is not part of the default trajectory. A run that enables it
must report the selected proposal and target, cadence, correction-weight
behavior, and agreement with the local-DMC baseline in addition to the ordinary
DMC checks.

## Estimators

### E1. Density Profiles

Code:
[src/hrdmc/estimators/observables/density.py](src/hrdmc/estimators/observables/density.py)

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

Validation scope:
Histogram normalization convention. Density integrates to \(N\) only when the
grid captures all particles and weights are normalized.

### E1b. Density Support Edges

Code:
[src/hrdmc/estimators/observables/density.py](src/hrdmc/estimators/observables/density.py)

Formula:

$$
x_{\mathrm{left/right}}
=
\min/\max\{x_b: n(x_b)>\epsilon\}.
$$

Source basis:
`Repo convention`.

Validation scope:
Rough occupied-bin diagnostic only. Physical cloud-boundary use requires an
explicit density threshold, grid, and validated density estimator.

### E2. Pair Distribution Function

Code:
[src/hrdmc/estimators/observables/pair_distribution.py](src/hrdmc/estimators/observables/pair_distribution.py)

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

Validation scope:
Mainly homogeneous validation. Trapped local pair analysis would need a separate
normalization.

### E3. Static Structure Factor

Code:
[src/hrdmc/estimators/observables/structure_factor.py](src/hrdmc/estimators/observables/structure_factor.py)

Formula:

$$
\rho_k=\sum_j e^{ikx_j},
\qquad
S(k)=\frac{\langle |\rho_k|^2\rangle}{N}.
$$

Source basis:
`Primary physics`, `[Mazzanti2008HardRods]`.

Validation scope:
Uses physical wrapped particle positions \(x_j\). Reduced coordinates are
reserved for the excluded-volume mapping.

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

Validation scope:
The homogeneous all-pair ring case is a validation benchmark. Trapped VMC local
energy is diagnostic unless backed by DMC/external validation.

### E5. Cloud Radius

Code:
[src/hrdmc/estimators/observables/cloud.py](src/hrdmc/estimators/observables/cloud.py)

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
`Primary physics` observable context: 
`[AstrakharchikGiorgini2002TrappedCrossover]` is the important trapped-QMC
precedent for energy/radius reporting and QMC-vs-LDA-style comparison;
`[GirardeauAstrakharchik2010TrappedHardSphere]` supplies hard-rod trapped
context.

Validation scope:
For DMC, direct weighted \(R^2\)/RMS from sampled coordinates is a
mixed-coordinate diagnostic only. Reported pure \(R^2\)/RMS must come from
Hellmann-Feynman energy response or a transported auxiliary forward-walking
estimator that passes its own check.

### E6. Weighted DMC Observables

Code:
[src/hrdmc/estimators/mixed/weighted.py](src/hrdmc/estimators/mixed/weighted.py)

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

Validation scope:
The estimator filters non-finite, invalid, and non-positive-weight samples.
Energy is the mixed Hamiltonian estimator. Direct weighted density, \(R^2\),
RMS, pair-distance density, and \(S(k)\) are mixed-coordinate diagnostics for
DMC. Reported pure coordinate observables require Hellmann-Feynman energy
response for \(R^2\)/RMS or transported auxiliary forward walking for
coordinate observables.

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

Validation scope:
Plateau behavior is part of the validation record. A single blocking number
alone is insufficient validation.

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

Validation scope:
This can replace a missing blocking plateau as the precision estimate only when
finite/hard-core validity, density accounting, R-hat, effective sample size,
and explicit trace-stationarity checks pass. A collective-RN run must also pass
its correction-weight checks. The missing plateau remains visible and the
reported error bar is explicitly inflated.

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

Validation scope:
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

Validation scope:
This metric is mathematically valid as a normed discrepancy. It does not decide
which curve is physically accurate; that depends on independent reference and
estimator validation.

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

Validation scope:
Replicate spread is a diagnostic, separate from timestep, population, and
stationarity controls.

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

The stationarity check is:

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

where \(s\) indexes independent seeds and \(M\) is the seed count. Energy-side
spread warnings, missing blocking plateaus, or correlated-error inflation mark
the energy case as a precision warning. Mixed coordinate traces have a
separate diagnostic status and do not alter the Hamiltonian energy estimate.

Source basis:
`Method paper`, autocorrelation/error-control logic aligned with
`[FlyvbjergPetersen1989Blocking]` and A2; chain-agreement context from
`[VehtariGelmanSimpsonCarpenterBuerkner2021Rhat]`; DMC validation need from
`[Foulkes2001QMC]`.

Validation scope:
Diagnostic-support analysis only. It is separate from physics sources and from
timestep/population/accounting validation.

## DMC Transport Event Contract and Transported Forward Walking

Code:
`src/hrdmc/monte_carlo/dmc/local/transport.py`,
`src/hrdmc/estimators/pure/forward_walking/`

The local DMC engine can emit one transport event per DMC step. The event is a
bookkeeping contract rather than an estimator formula:

$$
\mathcal E_t =
\left(
x_t,\ E_L(x_t),\ w_t^{\mathrm{pre}},\ w_t^{\mathrm{post}},\ p_t,\ I_t
\right),
$$

where \(p_t(j)\) is the post-resampling parent index of final walker \(j\) at
that DMC step, \(w_t^{\mathrm{pre}}\) are gauge-shifted pre-resampling log
weights, \(w_t^{\mathrm{post}}\) define the normalized estimator weights for
the emitted post-step population, and \(I_t\) stores the explicit convention
fields. Global log-weight gauge shifts cancel under normalized averages and
therefore act as bookkeeping gauges for transported auxiliary variables.

The optional COM Rao-Blackwell R2 payload is:

$$
r_i = x_i-x_0-\frac1N\sum_j (x_j-x_0),
\qquad
R^2_{\mathrm{RB}} =
\frac1N\sum_i r_i^2 + \mathrm{Var}(Q).
$$

The transported auxiliary estimator supports \(R^2\), density-bin vectors
whose integral gives particle count, pair-distance-density vectors whose
integral gives pair count, and static structure factor vectors. A single-point
collect/delay block uses:

$$
P_j \leftarrow P_{p_t(j)}
$$

at every DMC step. During the collection phase,

$$
P_j \leftarrow P_j + A_j,
\qquad
A_j = \frac1N\sum_i (x_{j,i}-x_0)^2
$$

or the corresponding vector-valued bin/Fourier observable for density,
pair-distance density, or \(S(k)\). For \(R^2\), \(A_j=R^2_{\mathrm{RB},j}\)
if the engine emits the RB payload and the estimator is configured to use it.
During the delay phase, \(P_j\) is only transported. For \(L>0\), current
`single_point` mode enforces one collected step before the delay, so every
lagged contribution has the same forward length \(L\):

$$
\widehat A_L =
\sum_{j=1}^M \tilde w^{\mathrm{post}}_j P_j.
$$

The \(L=0\) identity anchor is evaluated as the block average of instantaneous
normalized weighted mixed estimates,

$$
\widehat A_0 =
\frac1B \sum_{t=1}^B \sum_{j=1}^M
\tilde w_{t,j}^{\mathrm{post}} A_{t,j},
$$

so changing no-resample weights inside a collection block cannot break the
lag-zero mixed-estimator identity.

The RMS radius is defined after aggregating \(R^2\):

$$
R_{\mathrm{rms}} = \sqrt{\widehat{R^2}}.
$$

Per-configuration square roots are diagnostic quantities.

Source basis:
Bookkeeping contract for the transported auxiliary-variable forward-walking
approach in `[CasullerasBoronat1995Pure]` /
`[SarsaBoronatCasulleras2002Pure]`, adapted to this repository's local DMC
weighted-population engine.

Validation scope:
A reported coordinate result requires the transport stream plus lag-zero
identity, deterministic parent-map, gauge cancellation, lag stability,
sufficient block count, walker-weight support, independent source-family
support, density accounting when density is requested, and population checks.

## Experiment Scripts

Experiment scripts under `experiments/` are orchestration surfaces. They call
the owners above without owning formulas. A new experiment-level formula needs
a source-map entry before it is used in the thesis.

## Current Missing Source-Map Items

All active `src/hrdmc` physics/method formulas have source-map entries.
Generated run scripts and archived artifacts sit outside this map; package
formulas enter this document before thesis use.
