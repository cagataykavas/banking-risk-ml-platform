from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

import numpy as np

from src.decision_economics import EconomicsAssumptions


@dataclass(frozen=True)
class PolicyScenario:
    """A probability and economics stress applied before routing decisions."""

    name: str
    pd_logit_shift: float = 0.0
    pd_multiplier: float = 1.0
    assumptions: EconomicsAssumptions = field(default_factory=EconomicsAssumptions)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("scenario name must contain 1-100 characters")
        if not math.isfinite(self.pd_logit_shift) or abs(self.pd_logit_shift) > 20:
            raise ValueError("pd_logit_shift must be finite and in [-20, 20]")
        if not math.isfinite(self.pd_multiplier) or not 0 < self.pd_multiplier <= 10:
            raise ValueError("pd_multiplier must be finite and in (0, 10]")
        _validate_assumptions(self.assumptions)


@dataclass(frozen=True)
class RobustPolicyConstraints:
    max_review_rate: float = 0.30
    min_approval_rate: float = 0.20

    def __post_init__(self) -> None:
        for name in ("max_review_rate", "min_approval_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class ScenarioPolicyEvidence:
    scenario: str
    expected_value_per_application: float
    approval_rate: float
    review_rate: float
    reject_rate: float


@dataclass(frozen=True)
class RobustPolicyCandidate:
    approve_below: float
    reject_above: float
    worst_case_value_per_application: float
    mean_value_per_application: float
    maximum_regret_per_application: float
    worst_case_scenario: str
    max_review_rate: float
    min_approval_rate: float
    scenarios: tuple[ScenarioPolicyEvidence, ...]


@dataclass(frozen=True)
class RobustPolicyReport:
    accepted: bool
    reason_codes: tuple[str, ...]
    evaluated_candidates: int
    feasible_candidates: int
    scenario_names: tuple[str, ...]
    selected: RobustPolicyCandidate | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_assumptions(assumptions: EconomicsAssumptions) -> None:
    if not isinstance(assumptions, EconomicsAssumptions):
        raise TypeError("assumptions must be EconomicsAssumptions")
    nonnegative = (
        "approved_account_revenue",
        "loss_given_default",
        "manual_review_cost",
        "false_decline_opportunity_cost",
    )
    for name in nonnegative:
        value = getattr(assumptions, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    capture = assumptions.review_default_capture_rate
    if isinstance(capture, bool) or not isinstance(capture, (int, float)):
        raise TypeError("review_default_capture_rate must be numeric")
    if not math.isfinite(capture) or not 0 <= capture <= 1:
        raise ValueError("review_default_capture_rate must be finite and in [0, 1]")


def _probabilities(values: object) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("base_pd must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("base_pd must contain only finite values")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("base_pd must be in [0, 1]")
    return probabilities


def _grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain numeric values")
        number = float(value)
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name} must contain finite values in [0, 1]")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"{name} cannot be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} cannot contain duplicates")
    return tuple(sorted(parsed))


def stress_probabilities(base_pd: np.ndarray, scenario: PolicyScenario) -> np.ndarray:
    """Apply a bounded odds shift and multiplier without producing 0/0 logits."""
    epsilon = np.finfo(float).eps
    clipped = np.clip(base_pd, epsilon, 1 - epsilon)
    logits = np.log(clipped / (1 - clipped)) + scenario.pd_logit_shift
    shifted = 1 / (1 + np.exp(-logits))
    return np.clip(shifted * scenario.pd_multiplier, 0, 1)


def _evaluate(
    probabilities: np.ndarray,
    *,
    approve_below: float,
    reject_above: float,
    scenario: PolicyScenario,
) -> ScenarioPolicyEvidence:
    approved = probabilities < approve_below
    reviewed = (probabilities >= approve_below) & (probabilities < reject_above)
    rejected = probabilities >= reject_above
    assumptions = scenario.assumptions

    approve_value = assumptions.approved_account_revenue - (
        probabilities * assumptions.loss_given_default
    )
    review_value = (
        -assumptions.manual_review_cost
        + (1 - probabilities) * assumptions.approved_account_revenue
        + probabilities
        * (1 - assumptions.review_default_capture_rate)
        * (assumptions.approved_account_revenue - assumptions.loss_given_default)
    )
    reject_value = -(1 - probabilities) * assumptions.false_decline_opportunity_cost
    value = np.select(
        [approved, reviewed, rejected],
        [approve_value, review_value, reject_value],
    )
    return ScenarioPolicyEvidence(
        scenario=scenario.name,
        expected_value_per_application=float(np.mean(value)),
        approval_rate=float(np.mean(approved)),
        review_rate=float(np.mean(reviewed)),
        reject_rate=float(np.mean(rejected)),
    )


def optimize_robust_policy(
    base_pd: object,
    *,
    scenarios: Iterable[PolicyScenario],
    approve_grid: Iterable[float] | None = None,
    reject_grid: Iterable[float] | None = None,
    constraints: RobustPolicyConstraints | None = None,
) -> RobustPolicyReport:
    """Select a fixed two-threshold policy by maximin expected value.

    Feasibility is required in every scenario. Ties are resolved by lower maximum
    scenario regret, higher mean value, and then lexicographically lower thresholds.
    """
    probabilities = _probabilities(base_pd)
    scenario_rows = tuple(scenarios)
    if not scenario_rows:
        raise ValueError("at least one scenario is required")
    if any(not isinstance(item, PolicyScenario) for item in scenario_rows):
        raise ValueError("scenarios must contain PolicyScenario values")
    names = tuple(item.name for item in scenario_rows)
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")

    approve_values = _grid(
        approve_grid if approve_grid is not None else np.linspace(0.03, 0.30, 10),
        "approve_grid",
    )
    reject_values = _grid(
        reject_grid if reject_grid is not None else np.linspace(0.35, 0.80, 10),
        "reject_grid",
    )
    active_constraints = constraints or RobustPolicyConstraints()
    stressed = {
        scenario.name: stress_probabilities(probabilities, scenario) for scenario in scenario_rows
    }

    evaluated = 0
    raw_candidates: list[tuple[float, float, tuple[ScenarioPolicyEvidence, ...]]] = []
    for approve_below in approve_values:
        for reject_above in reject_values:
            if approve_below >= reject_above:
                continue
            evaluated += 1
            evidence = tuple(
                _evaluate(
                    stressed[scenario.name],
                    approve_below=approve_below,
                    reject_above=reject_above,
                    scenario=scenario,
                )
                for scenario in scenario_rows
            )
            if any(
                item.review_rate > active_constraints.max_review_rate
                or item.approval_rate < active_constraints.min_approval_rate
                for item in evidence
            ):
                continue
            raw_candidates.append((approve_below, reject_above, evidence))

    if not raw_candidates:
        return RobustPolicyReport(
            accepted=False,
            reason_codes=("NO_FEASIBLE_POLICY",),
            evaluated_candidates=evaluated,
            feasible_candidates=0,
            scenario_names=names,
            selected=None,
        )

    best_by_scenario = {
        name: max(
            next(item.expected_value_per_application for item in evidence if item.scenario == name)
            for _, _, evidence in raw_candidates
        )
        for name in names
    }
    candidates: list[RobustPolicyCandidate] = []
    for approve_below, reject_above, evidence in raw_candidates:
        values = [item.expected_value_per_application for item in evidence]
        worst_index = int(np.argmin(values))
        regrets = [
            best_by_scenario[item.scenario] - item.expected_value_per_application
            for item in evidence
        ]
        candidates.append(
            RobustPolicyCandidate(
                approve_below=approve_below,
                reject_above=reject_above,
                worst_case_value_per_application=values[worst_index],
                mean_value_per_application=mean(values),
                maximum_regret_per_application=max(regrets),
                worst_case_scenario=evidence[worst_index].scenario,
                max_review_rate=max(item.review_rate for item in evidence),
                min_approval_rate=min(item.approval_rate for item in evidence),
                scenarios=evidence,
            )
        )

    selected = min(
        candidates,
        key=lambda item: (
            -item.worst_case_value_per_application,
            item.maximum_regret_per_application,
            -item.mean_value_per_application,
            item.approve_below,
            item.reject_above,
        ),
    )
    return RobustPolicyReport(
        accepted=True,
        reason_codes=(),
        evaluated_candidates=evaluated,
        feasible_candidates=len(candidates),
        scenario_names=names,
        selected=selected,
    )


def default_policy_scenarios() -> tuple[PolicyScenario, ...]:
    return (
        PolicyScenario("baseline"),
        PolicyScenario(
            "recession",
            pd_logit_shift=0.55,
            pd_multiplier=1.05,
            assumptions=EconomicsAssumptions(
                approved_account_revenue=200.0,
                loss_given_default=3360.0,
                manual_review_cost=20.0,
                false_decline_opportunity_cost=110.0,
                review_default_capture_rate=0.72,
            ),
        ),
        PolicyScenario(
            "severe_downturn",
            pd_logit_shift=0.90,
            pd_multiplier=1.10,
            assumptions=EconomicsAssumptions(
                approved_account_revenue=180.0,
                loss_given_default=3920.0,
                manual_review_cost=24.0,
                false_decline_opportunity_cost=100.0,
                review_default_capture_rate=0.68,
            ),
        ),
    )


if __name__ == "__main__":
    rng = np.random.default_rng(91)
    probabilities = np.clip(rng.beta(1.2, 9.0, size=5000), 0.001, 0.95)
    print(optimize_robust_policy(probabilities, scenarios=default_policy_scenarios()).to_dict())
