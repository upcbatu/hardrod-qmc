# DMC Notes

This folder contains release-facing notes for the trapped hard-rod RN-block DMC
corridor.

## Trapped Unit Convention

Production trapped cases are parameterized in harmonic-oscillator length units:

```text
N8_A0.2
```

means \(N=8\) and \(A=a/a_{\mathrm{ho}}=0.2\). The physical trap frequency is
not an independent production case parameter after this nondimensionalization.
Internally the engine still uses the kinetic convention \(\hbar^2/(2m)=1\).
In that convention harmonic-oscillator units correspond to
`omega_code = sqrt(2)`, and code energies are in \(\hbar\Omega/2\). Divide
reported engine energies by two to express them in \(\hbar\Omega\).

Legacy case strings such as `N8_a0.5_omega0.2` are no longer accepted by the
production trapped runners. Convert them explicitly to \(A=a/a_{\mathrm{ho}}\)
before launching a new run.

- [method.md](method.md): end-to-end method flow, gate semantics, and claim
  boundary.
- [transport_contract.md](transport_contract.md): RN-block event stream
  contract for transported auxiliary forward-walking estimators.

Formula ownership and literature references remain in
[../03_EQUATION_SOURCE_MAP.md](../03_EQUATION_SOURCE_MAP.md).
