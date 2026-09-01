from __future__ import annotations

import numpy as np

from src.decision_economics import (
    EconomicsAssumptions,
    evaluate_policy,
    optimize_thresholds,
)


def test_evaluate_policy_rates_sum_to_one() -> None:
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.02, 0.15, 0.25, 0.45, 0.72, 0.91])
    result = evaluate_policy(y, p, approve_below=0.20, reject_above=0.70)

    assert abs(result.approval_rate + result.review_rate + result.reject_rate - 1.0) < 1e-12
    assert result.bad_rate_approved == 0.0


def test_review_cost_reduces_policy_value() -> None:
    y = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.05, 0.12, 0.28, 0.34, 0.48, 0.52, 0.66, 0.82])

    cheap = evaluate_policy(
        y,
        p,
        approve_below=0.20,
        reject_above=0.75,
        assumptions=EconomicsAssumptions(manual_review_cost=1.0),
    )
    expensive = evaluate_policy(
        y,
        p,
        approve_below=0.20,
        reject_above=0.75,
        assumptions=EconomicsAssumptions(manual_review_cost=100.0),
    )
    assert cheap.expected_total_value > expensive.expected_total_value


def test_optimizer_respects_review_capacity() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.15, size=1000)
    scores = np.clip(0.08 + 0.65 * y + rng.normal(0.0, 0.13, size=len(y)), 0.0, 1.0)

    table = optimize_thresholds(y, scores, max_review_rate=0.25)

    assert not table.empty
    assert (table["review_rate"] <= 0.25 + 1e-12).all()
    assert table.iloc[0]["expected_value_per_application"] >= table.iloc[-1][
        "expected_value_per_application"
    ]


def test_invalid_probability_rejected() -> None:
    y = np.array([0, 1])
    scores = np.array([0.2, 1.2])
    try:
        evaluate_policy(y, scores, approve_below=0.2, reject_above=0.8)
    except ValueError as exc:
        assert "probabilities" in str(exc)
    else:
        raise AssertionError("invalid score should fail validation")
