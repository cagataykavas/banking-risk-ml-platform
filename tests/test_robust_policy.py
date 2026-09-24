from __future__ import annotations

import json

import numpy as np
import pytest

from src.decision_economics import EconomicsAssumptions
from src.robust_policy import (
    PolicyScenario,
    RobustPolicyConstraints,
    default_policy_scenarios,
    optimize_robust_policy,
    stress_probabilities,
)


def _probabilities() -> np.ndarray:
    return np.array([0.01, 0.03, 0.06, 0.10, 0.18, 0.28, 0.45, 0.72, 0.90])


def test_selects_maximin_policy_with_scenario_evidence() -> None:
    report = optimize_robust_policy(
        _probabilities(),
        scenarios=default_policy_scenarios(),
        approve_grid=[0.05, 0.10, 0.20],
        reject_grid=[0.40, 0.65, 0.85],
        constraints=RobustPolicyConstraints(max_review_rate=0.80, min_approval_rate=0.10),
    )

    assert report.accepted is True
    assert report.selected is not None
    assert report.selected.worst_case_scenario == "severe_downturn"
    assert len(report.selected.scenarios) == 3
    assert report.selected.worst_case_value_per_application == min(
        item.expected_value_per_application for item in report.selected.scenarios
    )


def test_ex_ante_value_matches_transparent_decision_economics() -> None:
    scenario = PolicyScenario(
        "baseline",
        assumptions=EconomicsAssumptions(
            approved_account_revenue=100.0,
            loss_given_default=1000.0,
            manual_review_cost=10.0,
            false_decline_opportunity_cost=50.0,
            review_default_capture_rate=0.75,
        ),
    )
    report = optimize_robust_policy(
        [0.10, 0.40, 0.90],
        scenarios=[scenario],
        approve_grid=[0.20],
        reject_grid=[0.80],
        constraints=RobustPolicyConstraints(max_review_rate=1.0, min_approval_rate=0.0),
    )
    evidence = report.selected.scenarios[0]  # type: ignore[union-attr]
    approve_value = 100.0 - 0.10 * 1000.0
    review_value = -10.0 + 0.60 * 100.0 + 0.40 * 0.25 * (100.0 - 1000.0)
    reject_value = -(1 - 0.90) * 50.0
    assert evidence.expected_value_per_application == pytest.approx(
        (approve_value + review_value + reject_value) / 3
    )


def test_feasibility_must_hold_in_every_scenario() -> None:
    report = optimize_robust_policy(
        _probabilities(),
        scenarios=[PolicyScenario("base"), PolicyScenario("stress", pd_logit_shift=1.0)],
        approve_grid=[0.20],
        reject_grid=[0.60],
        constraints=RobustPolicyConstraints(max_review_rate=0.10, min_approval_rate=0.80),
    )
    assert report.accepted is False
    assert report.reason_codes == ("NO_FEASIBLE_POLICY",)
    assert report.selected is None
    assert report.feasible_candidates == 0


def test_probability_stress_is_bounded_and_monotonic() -> None:
    base = np.array([0.0, 0.05, 0.5, 1.0])
    stressed = stress_probabilities(
        base, PolicyScenario("stress", pd_logit_shift=0.6, pd_multiplier=1.1)
    )
    assert np.all((stressed >= 0) & (stressed <= 1))
    assert np.all(np.diff(stressed) >= 0)
    assert stressed[1] > base[1]


def test_report_is_deterministic_and_json_ready() -> None:
    kwargs = {
        "scenarios": default_policy_scenarios(),
        "approve_grid": [0.05, 0.10],
        "reject_grid": [0.50, 0.75],
        "constraints": RobustPolicyConstraints(max_review_rate=0.8, min_approval_rate=0.1),
    }
    first = optimize_robust_policy(_probabilities(), **kwargs).to_dict()
    second = optimize_robust_policy(_probabilities(), **kwargs).to_dict()
    assert first == second
    assert json.loads(json.dumps(first))["selected"]["worst_case_scenario"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.1, float("nan")], "finite"),
        ([0.1, 1.1], r"\[0, 1\]"),
        ([], "non-empty"),
        ([[0.1], [0.2]], "one-dimensional"),
    ],
)
def test_invalid_probabilities_fail_closed(values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        optimize_robust_policy(values, scenarios=[PolicyScenario("base")])


def test_duplicate_scenarios_and_invalid_grids_fail_closed() -> None:
    with pytest.raises(ValueError, match="unique"):
        optimize_robust_policy(
            _probabilities(),
            scenarios=[PolicyScenario("same"), PolicyScenario("same")],
        )
    with pytest.raises(ValueError, match="duplicates"):
        optimize_robust_policy(
            _probabilities(),
            scenarios=[PolicyScenario("base")],
            approve_grid=[0.1, 0.1],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        optimize_robust_policy(
            _probabilities(),
            scenarios=[PolicyScenario("base")],
            reject_grid=[float("inf")],
        )


def test_invalid_scenario_economics_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PolicyScenario(
            "bad-cost",
            assumptions=EconomicsAssumptions(manual_review_cost=-1.0),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PolicyScenario(
            "bad-capture",
            assumptions=EconomicsAssumptions(review_default_capture_rate=1.2),
        )
    with pytest.raises(ValueError, match=r"\(0, 10\]"):
        PolicyScenario("bad-pd", pd_multiplier=0.0)
    with pytest.raises(ValueError, match=r"\[-20, 20\]"):
        PolicyScenario("bad-shift", pd_logit_shift=21.0)


def test_candidate_count_excludes_inverted_threshold_pairs() -> None:
    report = optimize_robust_policy(
        _probabilities(),
        scenarios=[PolicyScenario("base")],
        approve_grid=[0.2, 0.7],
        reject_grid=[0.5, 0.8],
        constraints=RobustPolicyConstraints(max_review_rate=1.0, min_approval_rate=0.0),
    )
    assert report.evaluated_candidates == 3
    assert report.feasible_candidates == 3
