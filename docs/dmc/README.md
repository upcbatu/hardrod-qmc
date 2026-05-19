# DMC Notes

This folder contains release-facing notes for the trapped hard-rod RN-block DMC
corridor.

## Trapped Unit Convention

Production trapped cases are parameterized in harmonic-oscillator length units:

```text
N8_A0.2
```

means \(N=8\) and \(A=a/a_{\mathrm{ho}}=0.2\). The physical trap frequency is
absorbed into \(A\) after this nondimensionalization.
The engine stores dimensionless variables \(q=x/a_{\mathrm{ho}}\) and
\(\widetilde E=E/(\hbar\Omega)\), so the default trapped Hamiltonian in code
variables is \(-\frac12\nabla_q^2+\frac12 q^2+\widetilde V_{\rm HR}\).
Reported energies are in units of \(\hbar\Omega\).

Legacy case strings such as `N8_a0.5_omega0.2` are no longer accepted by the
production trapped runners. Convert them explicitly to \(A=a/a_{\mathrm{ho}}\)
before launching a new run.

- [method.md](method.md): end-to-end method flow, numerical checks, and scope
  boundary.
- [transport_contract.md](transport_contract.md): RN-block event stream
  contract for transported auxiliary forward-walking estimators.

Formula ownership and literature references remain in
[../03_EQUATION_SOURCE_MAP.md](../03_EQUATION_SOURCE_MAP.md).
