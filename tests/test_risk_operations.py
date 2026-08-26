from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.cost_sensitive_policy import CostModel, expected_cost, expected_loss_table, search_policy
from src.drift_monitoring import numeric_drift, psi, score_stability
from src.temporal_validation import evaluate_temporally, rolling_time_splits, stability_summary


def test_cost_sensitive_policy_respects_review_capacity():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.04, 0.10, 0.55, 0.75, 0.92])

    result = expected_cost(
        y,
        scores,
        approve_below=0.12,
        decline_above=0.70,
        costs=CostModel(false_approve_cost=1000, false_decline_cost=100, manual_review_cost=10),
    )

    assert result.approval_rate == 0.5
    assert result.decline_rate == 2 / 6
    assert result.review_rate == 1 / 6
    assert result.bad_rate_approved == 0.0
    assert result.default_capture_rate == 1.0


def test_policy_search_returns_lowest_cost_first():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.01, 0.04, 0.08, 0.15, 0.35, 0.55, 0.75, 0.95])
    frame = search_policy(
        y,
        scores,
        approve_grid=[0.05, 0.10, 0.20],
        decline_grid=[0.50, 0.70, 0.90],
        max_review_rate=0.75,
        min_approval_rate=0.10,
        min_default_capture_rate=0.50,
    )

    assert not frame.empty
    assert frame["expected_cost_per_application"].is_monotonic_increasing


def test_expected_loss_formula():
    table = expected_loss_table(
        balances=np.array([1000.0, 2000.0]),
        pd_scores=np.array([0.10, 0.20]),
        lgd=0.50,
    )
    np.testing.assert_allclose(table["expected_loss"].to_numpy(), [50.0, 200.0])


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.2, 1.1, 5000)

    assert psi(reference, same) < 0.10
    assert psi(reference, shifted) > 0.25


def test_numeric_and_score_drift_reports():
    rng = np.random.default_rng(12)
    reference = pd.DataFrame({"income": rng.normal(100, 10, 1000), "utilization": rng.beta(2, 5, 1000)})
    current = pd.DataFrame({"income": rng.normal(115, 12, 1000), "utilization": rng.beta(3, 4, 1000)})

    report = numeric_drift(reference, current, ["income", "utilization"])
    assert set(report["feature"]) == {"income", "utilization"}
    assert set(report["severity"]).issubset({"stable", "warning", "alert"})

    score_report = score_stability(reference["utilization"].to_numpy(), current["utilization"].to_numpy())
    assert "score_psi" in score_report
    assert "severity" in score_report


def _temporal_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for day_index, timestamp in enumerate(pd.date_range("2024-01-01", periods=420, freq="D", tz="UTC")):
        for _ in range(8):
            x1 = rng.normal()
            x2 = rng.normal(0.001 * day_index, 1.0)
            probability = 1 / (1 + np.exp(-(-1.8 + 0.9 * x1 + 0.4 * x2)))
            target = rng.binomial(1, probability)
            rows.append((timestamp, x1, x2, target))
    return pd.DataFrame(rows, columns=["event_time", "x1", "x2", "default"])


def test_rolling_time_splits_never_train_on_future_data():
    frame = _temporal_frame()
    splits = list(
        rolling_time_splits(
            frame,
            time_column="event_time",
            min_train_days=180,
            test_days=30,
            step_days=30,
            gap_days=2,
        )
    )
    assert splits
    for _, train, test in splits:
        assert train["event_time"].max() < test["event_time"].min()


def test_temporal_evaluation_produces_stability_metrics():
    frame = _temporal_frame()
    folds = evaluate_temporally(
        LogisticRegression(max_iter=500),
        frame,
        features=["x1", "x2"],
        target="default",
        time_column="event_time",
        min_train_days=180,
        test_days=30,
        step_days=30,
        gap_days=1,
    )
    summary = stability_summary(folds)

    assert len(folds) >= 3
    assert "roc_auc_mean" in summary
    assert "brier_score_mean" in summary
