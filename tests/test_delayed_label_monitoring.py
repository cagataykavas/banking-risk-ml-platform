import pandas as pd

from src.delayed_label_monitoring import (
    LabelMaturityPolicy,
    cohort_performance,
    mature_predictions,
    maturity_report,
)


def _fixtures():
    predictions = pd.DataFrame(
        {
            "application_id": ["a", "b", "c", "d"],
            "decision_time": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-05-01", "2026-08-01"],
                utc=True,
            ),
            "default_probability": [0.05, 0.20, 0.80, 0.10],
        }
    )
    outcomes = pd.DataFrame(
        {
            "application_id": ["a", "b", "c"],
            "default_label": [0, 1, 1],
            "observed_at": pd.to_datetime(
                ["2026-04-01", "2026-05-02", "2026-07-30"],
                utc=True,
            ),
        }
    )
    return predictions, outcomes


def test_recent_predictions_are_not_scored_as_negatives():
    predictions, outcomes = _fixtures()
    policy = LabelMaturityPolicy(outcome_horizon_days=90, reporting_lag_days=7, min_rows=1)
    mature = mature_predictions(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=policy,
    )

    assert set(mature["application_id"]) == {"a", "b", "c"}
    assert "d" not in set(mature["application_id"])
    assert mature["label_missing"].sum() == 0


def test_missing_mature_label_is_data_quality_signal():
    predictions, outcomes = _fixtures()
    outcomes = outcomes.loc[outcomes["application_id"] != "b"]
    report = maturity_report(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(outcome_horizon_days=90, reporting_lag_days=7, min_rows=1),
    )
    assert report["missing_mature_labels"] == 1
    assert report["missing_label_rate"] > 0


def test_cohort_metrics_use_only_labeled_mature_rows():
    predictions, outcomes = _fixtures()
    mature = mature_predictions(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(outcome_horizon_days=90, reporting_lag_days=7, min_rows=1),
    )
    metrics = cohort_performance(mature, min_rows=1)
    assert len(metrics) == 3
    assert set(metrics["rows"]) == {1}
