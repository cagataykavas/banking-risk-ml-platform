from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    roc_auc: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right == 1.0:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not mask.any():
            continue
        confidence = float(probability[mask].mean())
        event_rate = float(y_true[mask].mean())
        ece += abs(confidence - event_rate) * (int(mask.sum()) / total)
    return float(ece)


def reliability_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probability, n_bins=bins, strategy="quantile")
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_event_rate": observed,
            "calibration_gap": observed - predicted,
        }
    )


def evaluate(method: str, y_true: np.ndarray, probability: np.ndarray) -> CalibrationResult:
    return CalibrationResult(
        method=method,
        roc_auc=float(roc_auc_score(y_true, probability)),
        brier_score=float(brier_score_loss(y_true, probability)),
        log_loss=float(log_loss(y_true, probability, labels=[0, 1])),
        expected_calibration_error=expected_calibration_error(y_true, probability),
    )


def compare_calibrators(
    estimator: BaseEstimator,
    x_train,
    y_train,
    x_calibration,
    y_calibration,
    x_test,
    y_test,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit the base model once, then calibrate only on a separate calibration period.

    Keeping tuning/training, calibration and final evaluation data separate avoids
    the common mistake of reporting probability quality on the data used to fit the
    calibrator itself.
    """
    base = clone(estimator)
    base.fit(x_train, y_train)

    models: dict[str, BaseEstimator] = {"uncalibrated": base}
    for method in ["sigmoid", "isotonic"]:
        calibrated = CalibratedClassifierCV(base, method=method, cv="prefit")
        calibrated.fit(x_calibration, y_calibration)
        models[method] = calibrated

    rows: list[CalibrationResult] = []
    target = np.asarray(y_test, dtype=int)
    for name, model in models.items():
        probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        rows.append(evaluate(name, target, probability))

    report = pd.DataFrame([row.__dict__ for row in rows]).sort_values(
        ["brier_score", "log_loss"], ascending=[True, True]
    )
    return report.reset_index(drop=True), models
