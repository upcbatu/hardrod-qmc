# Estimators

Owner: observable reducers and estimator semantics.

Layout:

```text
observables/  coordinate kernels independent of sampling semantics
mixed/        weighted mixed reductions from DMC walker populations
pure/         pure-estimator routes such as energy response and FW
```

Shared observable kernels are not called mixed or pure by themselves. The
sampled distribution and reducer decide the estimator semantics.

This package must not own Monte Carlo propagation, theory approximations, or
benchmark comparison logic.
