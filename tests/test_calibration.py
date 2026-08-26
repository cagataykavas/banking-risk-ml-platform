import numpy as np
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    compare_calibrators,
    expected_calibration_error,
    reliability_table,
)


def test_ece_is_zero_for_perfect_two_bin_forecast():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, bins=2) == 0.0


def test_reliability_table_has_expected_columns():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.60, 0.75, 0.90])
    table = reliability_table(y, p, bins=3)
    assert set(table.columns) == {
        "mean_predicted_probability",
        "observed_event_rate",
        "calibration_gap",
    }


def test_calibration_comparison_runs_on_separate_holdouts():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(900, 3))
    logit = -1.2 + 1.0 * x[:, 0] - 0.7 * x[:, 1] + 0.4 * x[:, 2]
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)

    report, models = compare_calibrators(
        LogisticRegression(max_iter=1000),
        x[:500],
        y[:500],
        x[500:700],
        y[500:700],
        x[700:],
        y[700:],
    )

    assert set(report["method"]) == {"uncalibrated", "sigmoid", "isotonic"}
    assert set(models) == {"uncalibrated", "sigmoid", "isotonic"}
    assert report["brier_score"].between(0, 1).all()
