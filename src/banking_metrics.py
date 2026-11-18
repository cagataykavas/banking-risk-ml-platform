from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CapacityPoint:
    """Ranking quality at a fixed review capacity with explicit cutoff-tie uncertainty."""

    fraction: float
    reviewed_count: int
    cutoff_score: float
    fixed_count: int
    boundary_tie_count: int
    boundary_slots: int
    capture_rate: float
    capture_rate_lower: float
    capture_rate_upper: float
    precision: float
    precision_lower: float
    precision_upper: float
    lift: float
    lift_lower: float
    lift_upper: float


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximum separation between empirical positive and negative score distributions."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _validate_ranking_inputs(
    y_true: np.ndarray, scores: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError(
            "y_true and scores must be one-dimensional arrays of equal length"
        )
    if len(y) == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.all(np.isfinite(p)):
        raise ValueError("scores must contain only finite values")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0, 1]")
    if not np.any(y == 1):
        raise ValueError("at least one positive event is required")
    return y.astype(int, copy=False), p


def capacity_at_fraction(
    y_true: np.ndarray,
    scores: np.ndarray,
    fraction: float = 0.10,
) -> CapacityPoint:
    """Evaluate a top-score queue without resolving cutoff ties by row order.

    When a tied score group straddles the review-capacity cutoff, the central
    estimate allocates that group's events proportionally. Lower and upper bounds
    report the exact range across every possible ordering inside that tie group.
    """
    y, p = _validate_ranking_inputs(y_true, scores, fraction)
    reviewed_count = max(1, int(np.ceil(len(y) * fraction)))
    cutoff_score = float(np.sort(p)[::-1][reviewed_count - 1])

    fixed = p > cutoff_score
    boundary = p == cutoff_score
    fixed_count = int(np.sum(fixed))
    boundary_tie_count = int(np.sum(boundary))
    boundary_slots = reviewed_count - fixed_count

    fixed_events = int(np.sum(y[fixed]))
    boundary_events = int(np.sum(y[boundary]))
    event_count = int(np.sum(y))
    base_rate = event_count / len(y)

    minimum_boundary_events = max(
        0,
        boundary_events - (boundary_tie_count - boundary_slots),
    )
    maximum_boundary_events = min(boundary_slots, boundary_events)
    expected_boundary_events = boundary_events * boundary_slots / boundary_tie_count

    expected_events = fixed_events + expected_boundary_events
    lower_events = fixed_events + minimum_boundary_events
    upper_events = fixed_events + maximum_boundary_events

    capture_rate = expected_events / event_count
    capture_rate_lower = lower_events / event_count
    capture_rate_upper = upper_events / event_count
    precision = expected_events / reviewed_count
    precision_lower = lower_events / reviewed_count
    precision_upper = upper_events / reviewed_count

    return CapacityPoint(
        fraction=float(fraction),
        reviewed_count=reviewed_count,
        cutoff_score=cutoff_score,
        fixed_count=fixed_count,
        boundary_tie_count=boundary_tie_count,
        boundary_slots=boundary_slots,
        capture_rate=float(capture_rate),
        capture_rate_lower=float(capture_rate_lower),
        capture_rate_upper=float(capture_rate_upper),
        precision=float(precision),
        precision_lower=float(precision_lower),
        precision_upper=float(precision_upper),
        lift=float(precision / base_rate),
        lift_lower=float(precision_lower / base_rate),
        lift_upper=float(precision_upper / base_rate),
    )


def capacity_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    fractions: Iterable[float] = (0.01, 0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Return tie-aware ranking evidence for several operational capacities."""
    points = [capacity_at_fraction(y_true, scores, fraction) for fraction in fractions]
    if not points:
        raise ValueError("fractions must contain at least one capacity")
    return pd.DataFrame(asdict(point) for point in points)


def lift_at_fraction(
    y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.10
) -> float:
    """Tie-neutral expected lift at a fixed top-score fraction."""
    return capacity_at_fraction(y_true, scores, fraction).lift


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
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
        auto_approve_below
        if auto_approve_below is not None
        else np.linspace(0.02, 0.30, 15)
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
                bad_rate_approved=float(np.mean(y_true[approved]))
                if np.any(approved)
                else 0.0,
                review_rate=float(np.mean(reviewed)),
                captured_bad_rate=float(
                    np.sum(y_true[captured_bad]) / max(np.sum(y_true), 1)
                ),
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def validation_report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Metrics commonly discussed for probability models in banking risk."""
    prob_true, prob_pred = calibration_curve(
        y_true, scores, n_bins=10, strategy="quantile"
    )
    calibration_mae = float(np.mean(np.abs(prob_true - prob_pred)))
    capacity = capacity_at_fraction(y_true, scores, 0.10)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "lift_at_10pct": capacity.lift,
        "lift_at_10pct_lower": capacity.lift_lower,
        "lift_at_10pct_upper": capacity.lift_upper,
        "capture_at_10pct": capacity.capture_rate,
        "capture_at_10pct_lower": capacity.capture_rate_lower,
        "capture_at_10pct_upper": capacity.capture_rate_upper,
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
