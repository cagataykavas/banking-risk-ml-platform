import pandas as pd
import pytest

from src.delayed_label_monitoring import (
    LabelMaturityPolicy,
    cohort_performance,
    evaluate_mature_vintage_calibration,
    mature_predictions,
    maturity_report,
)
from src.vintage_calibration import VintageCalibrationPolicy


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
    policy = LabelMaturityPolicy(
        outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
    )
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
        policy=LabelMaturityPolicy(
            outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
        ),
    )
    assert report["missing_mature_labels"] == 1
    assert report["missing_label_rate"] > 0


def test_cohort_metrics_use_only_labeled_mature_rows():
    predictions, outcomes = _fixtures()
    mature = mature_predictions(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(
            outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
        ),
    )
    metrics = cohort_performance(mature, min_rows=1)
    assert len(metrics) == 3
    assert set(metrics["rows"]) == {1}


def test_future_observation_is_unavailable_at_point_in_time():
    predictions, outcomes = _fixtures()
    outcomes.loc[outcomes["application_id"] == "b", "observed_at"] = pd.Timestamp(
        "2026-09-01", tz="UTC"
    )

    mature = mature_predictions(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(
            outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
        ),
    )

    row = mature.loc[mature["application_id"] == "b"].iloc[0]
    assert bool(row["label_unavailable_as_of"])
    assert bool(row["label_missing"])
    assert int(cohort_performance(mature, min_rows=1)["rows"].sum()) == 2
    report = maturity_report(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(
            outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
        ),
    )
    assert report["labels_unavailable_as_of"] == 1


def test_outcome_before_decision_is_rejected():
    predictions, outcomes = _fixtures()
    outcomes.loc[outcomes["application_id"] == "b", "observed_at"] = pd.Timestamp(
        "2025-12-31", tz="UTC"
    )

    with pytest.raises(ValueError, match="cannot precede"):
        mature_predictions(
            predictions,
            outcomes,
            as_of=pd.Timestamp("2026-08-26", tz="UTC"),
            policy=LabelMaturityPolicy(
                outcome_horizon_days=90, reporting_lag_days=7, min_rows=1
            ),
        )


def test_mature_adapter_builds_sorted_vintages_without_manual_arrays():
    decision_time = pd.to_datetime(
        ["2026-01-01", "2026-01-10", "2026-04-01", "2026-04-10"],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "application_id": ["a", "b", "c", "d"],
            "decision_time": decision_time,
            "default_probability": [0.25, 0.25, 0.25, 0.25],
        }
    )
    outcomes = pd.DataFrame(
        {
            "application_id": ["d", "c", "b", "a"],
            "default_label": [0, 1, 0, 1],
            "observed_at": decision_time[::-1] + pd.Timedelta(days=90),
        }
    )
    calibration_policy = VintageCalibrationPolicy(
        min_rows=2,
        min_defaults=0,
        min_oe_ratio=0.1,
        max_oe_ratio=2.0,
        max_abs_calibration_gap=0.30,
    )

    report = evaluate_mature_vintage_calibration(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-01", tz="UTC"),
        maturity_policy=LabelMaturityPolicy(
            outcome_horizon_days=90,
            reporting_lag_days=7,
            min_rows=1,
        ),
        calibration_policy=calibration_policy,
    )

    assert report.passed
    assert [result.vintage for result in report.vintages] == ["2026Q1", "2026Q2"]
    assert [result.rows for result in report.vintages] == [2, 2]


def test_mature_adapter_fails_closed_on_incomplete_labels():
    predictions, outcomes = _fixtures()
    outcomes = outcomes.loc[outcomes["application_id"] != "b"]

    with pytest.raises(ValueError, match="unavailable labels"):
        evaluate_mature_vintage_calibration(
            predictions,
            outcomes,
            as_of=pd.Timestamp("2026-08-26", tz="UTC"),
            maturity_policy=LabelMaturityPolicy(
                outcome_horizon_days=90,
                reporting_lag_days=7,
                min_rows=1,
            ),
        )


def test_mature_adapter_rejects_empty_maturity_window():
    predictions, outcomes = _fixtures()

    with pytest.raises(ValueError, match="no predictions are mature"):
        evaluate_mature_vintage_calibration(
            predictions,
            outcomes,
            as_of=pd.Timestamp("2026-01-15", tz="UTC"),
            maturity_policy=LabelMaturityPolicy(
                outcome_horizon_days=90,
                reporting_lag_days=7,
                min_rows=1,
            ),
        )
