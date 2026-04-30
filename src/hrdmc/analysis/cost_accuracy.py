from __future__ import annotations

from dataclasses import dataclass

from hrdmc.analysis.metrics import cost_score, mean_squared_error


@dataclass(frozen=True)
class EstimatorScore:
    """Support score for estimator diagnostics.

    Source/rationale
    ----------------
    This is support infrastructure, not the main thesis objective. It combines
    the standard statistical decomposition

        MSE = bias^2 + variance

    with a CPU-time multiplier. The estimator families are motivated by the
    DMC/pure-estimator papers
    [BoronatCasulleras1995PureEstimators] and [SarsaBoronatCasulleras2002QuadraticDMC].
    """

    observable: str
    estimator: str
    estimate: float
    reference: float
    variance: float
    cpu_seconds: float

    @property
    def bias(self) -> float:
        return self.estimate - self.reference

    @property
    def mse(self) -> float:
        return mean_squared_error(self.bias, self.variance)

    @property
    def cost_score(self) -> float:
        return cost_score(self.mse, self.cpu_seconds)
