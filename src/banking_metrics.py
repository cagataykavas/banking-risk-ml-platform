from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class DecisionPoint:
    threshold: float
    approval_rate: float
    bad_rate_approved: float
    review_rate: float
    captured_bad_rate: float


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximum separation between empirical positive and negative score distributions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def lift_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = len(y_true)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(scores)[::-1]
    top_rate = float(np.mean(y_true[order[:k]]))
    base_rate = float(np.mean(y_true))
    return top_rate / max(base_rate, 1e-12)


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for left, right in pairwise(edges):
        if right == 1.0:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not np.any(mask):
            continue
        confidence = float(np.mean(scores[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += abs(confidence - observed) * (np.sum(mask) / n)
    return float(ece)


def decision_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    auto_approve_below: list[float] | np.ndarray | None = None,
    auto_reject_above: float = 0.65,
) -> pd.DataFrame:
    """Evaluate operational trade-offs for a two-threshold credit decision policy.

    Applications below the lower threshold are automatically approved, applications
    above the upper threshold are automatically declined, and the middle band is
    routed to human review.
    """
    thresholds = np.asarray(
        auto_approve_below if auto_approve_below is not None else np.linspace(0.02, 0.30, 15)
    )
    rows: list[DecisionPoint] = []

    for threshold in thresholds:
        approved = scores < threshold
        reviewed = (scores >= threshold) & (scores < auto_reject_above)
        captured_bad = scores >= threshold

        rows.append(
            DecisionPoint(
                threshold=float(threshold),
                approval_rate=float(np.mean(approved)),
                bad_rate_approved=float(np.mean(y_true[approved])) if np.any(approved) else 0.0,
                review_rate=float(np.mean(reviewed)),
                captured_bad_rate=float(np.sum(y_true[captured_bad]) / max(np.sum(y_true), 1)),
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def validation_report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Metrics commonly discussed for probability models in banking risk."""
    prob_true, prob_pred = calibration_curve(y_true, scores, n_bins=10, strategy="quantile")
    calibration_mae = float(np.mean(np.abs(prob_true - prob_pred)))
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "lift_at_10pct": lift_at_fraction(y_true, scores, 0.10),
        "ece": expected_calibration_error(y_true, scores),
        "calibration_mae": calibration_mae,
        "event_rate": float(np.mean(y_true)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.12, size=5000)
    latent = 1.8 * y + rng.normal(0, 1.2, size=len(y))
    p = 1 / (1 + np.exp(-(latent - 2.0)))

    print(validation_report(y, p))
    print(decision_curve(y, p).head())
