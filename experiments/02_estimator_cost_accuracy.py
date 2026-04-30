"""Secondary estimator-diagnostic entrypoint.

This is no longer the main thesis workflow. The trapped hard-rod versus LDA
comparison owns the main scientific path. This script remains reserved for
support diagnostics if estimator-family comparisons become useful.

Potential diagnostics:
- VMC baseline
- mixed DMC
- extrapolated estimator
- pure / forward-walking estimator
- bias, variance, MSE, CPU cost
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Requires DMCResult production. This is secondary support infrastructure."
    )


if __name__ == "__main__":
    main()
