from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.cost_sensitive_policy import CostModel, expected_cost


@dataclass(frozen=True)
class StressScenario:
    name: str
    logit_shift: float = 0.0
    pd_multiplier: float = 1.0
    lgd_multiplier: float = 1.0
    balance_multiplier: float = 1.0


@dataclass(frozen=True)
class StressResult:
    scenario: str
    mean_pd: float
    p95_pd: float
    expected_loss: float
    expected_loss_per_account: float
    policy_cost_per_application: float
    approval_rate: float
    review_rate: float
    decline_rate: float


def logit(probability: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def stress_probabilities(base_pd: np.ndarray, scenario: StressScenario) -> np.ndarray:
    shifted = sigmoid(logit(np.asarray(base_pd, dtype=float)) + scenario.logit_shift)
    scaled = shifted * scenario.pd_multiplier
    return np.clip(scaled, 0.0, 1.0)


def expected_loss(
    balances: np.ndarray,
    stressed_pd: np.ndarray,
    *,
    base_lgd: float = 0.45,
    scenario: StressScenario,
) -> float:
    exposure = np.asarray(balances, dtype=float) * scenario.balance_multiplier
    lgd = np.clip(base_lgd * scenario.lgd_multiplier, 0.0, 1.0)
    return float(np.sum(exposure * stressed_pd * lgd))


def evaluate_scenario(
    *,
    y_true: np.ndarray,
    base_pd: np.ndarray,
    balances: np.ndarray,
    approve_below: float,
    decline_above: float,
    scenario: StressScenario,
    costs: CostModel = CostModel(),
    base_lgd: float = 0.45,
) -> StressResult:
    stressed = stress_probabilities(base_pd, scenario)
    policy = expected_cost(
        y_true,
        stressed,
        approve_below=approve_below,
        decline_above=decline_above,
        costs=costs,
    )
    loss = expected_loss(
        balances,
        stressed,
        base_lgd=base_lgd,
        scenario=scenario,
    )
    return StressResult(
        scenario=scenario.name,
        mean_pd=float(np.mean(stressed)),
        p95_pd=float(np.quantile(stressed, 0.95)),
        expected_loss=loss,
        expected_loss_per_account=float(loss / len(stressed)),
        policy_cost_per_application=policy.expected_cost_per_application,
        approval_rate=policy.approval_rate,
        review_rate=policy.review_rate,
        decline_rate=policy.decline_rate,
    )


def stress_matrix(
    *,
    y_true: np.ndarray,
    base_pd: np.ndarray,
    balances: np.ndarray,
    approve_below: float,
    decline_above: float,
    scenarios: list[StressScenario],
    costs: CostModel = CostModel(),
) -> pd.DataFrame:
    rows = [
        evaluate_scenario(
            y_true=y_true,
            base_pd=base_pd,
            balances=balances,
            approve_below=approve_below,
            decline_above=decline_above,
            scenario=scenario,
            costs=costs,
        )
        for scenario in scenarios
    ]
    frame = pd.DataFrame([row.__dict__ for row in rows])
    baseline = frame.loc[frame["scenario"] == "baseline", "expected_loss"]
    baseline_loss = float(baseline.iloc[0]) if not baseline.empty else float(frame["expected_loss"].iloc[0])
    frame["loss_uplift_vs_baseline"] = frame["expected_loss"] / max(baseline_loss, 1e-12) - 1.0
    return frame


def default_scenarios() -> list[StressScenario]:
    return [
        StressScenario("baseline"),
        StressScenario("mild_deterioration", logit_shift=0.20, lgd_multiplier=1.05),
        StressScenario("recession", logit_shift=0.55, lgd_multiplier=1.20, balance_multiplier=1.02),
        StressScenario("severe_downturn", logit_shift=0.90, lgd_multiplier=1.35, balance_multiplier=1.05),
    ]


if __name__ == "__main__":
    rng = np.random.default_rng(44)
    rows = 5000
    base_pd = np.clip(rng.beta(1.2, 9.0, rows), 0.005, 0.85)
    y_true = rng.binomial(1, base_pd)
    balances = rng.lognormal(mean=9.2, sigma=0.7, size=rows)

    report = stress_matrix(
        y_true=y_true,
        base_pd=base_pd,
        balances=balances,
        approve_below=0.08,
        decline_above=0.65,
        scenarios=default_scenarios(),
    )
    print(report.to_string(index=False))
