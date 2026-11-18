from __future__ import annotations

import numpy as np
import pytest

from src.banking_metrics import (
    capacity_at_fraction,
    capacity_curve,
    lift_at_fraction,
    validation_report,
)


def test_cutoff_tie_reports_exact_ranking_bounds() -> None:
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.90, 0.80, 0.80, 0.10])

    point = capacity_at_fraction(labels, scores, fraction=0.50)

    assert point.reviewed_count == 2
    assert point.cutoff_score == 0.80
    assert point.fixed_count == 1
    assert point.boundary_tie_count == 2
    assert point.boundary_slots == 1
    assert point.capture_rate == 0.75
    assert point.capture_rate_lower == 0.50
    assert point.capture_rate_upper == 1.00
    assert point.precision == 0.75
    assert point.lift == 1.50
    assert point.lift_lower == 1.00
    assert point.lift_upper == 2.00


def test_tie_aware_metrics_are_permutation_invariant() -> None:
    labels = np.array([1, 0, 1, 0, 1, 0])
    scores = np.array([0.90, 0.80, 0.80, 0.80, 0.30, 0.10])
    permutation = np.array([5, 2, 0, 4, 1, 3])

    original = capacity_at_fraction(labels, scores, fraction=0.50)
    shuffled = capacity_at_fraction(
        labels[permutation], scores[permutation], fraction=0.50
    )

    assert shuffled == original
    assert (
        lift_at_fraction(labels[permutation], scores[permutation], 0.50)
        == original.lift
    )


def test_bounds_collapse_without_a_partial_cutoff_tie() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.90, 0.80, 0.70, 0.60])

    point = capacity_at_fraction(labels, scores, fraction=0.50)

    assert point.capture_rate_lower == point.capture_rate == point.capture_rate_upper
    assert point.precision_lower == point.precision == point.precision_upper
    assert point.lift_lower == point.lift == point.lift_upper


def test_full_capacity_has_unit_capture_and_lift() -> None:
    point = capacity_at_fraction(
        np.array([1, 0, 1, 0]),
        np.array([0.70, 0.70, 0.20, 0.10]),
        fraction=1.0,
    )

    assert point.capture_rate == 1.0
    assert point.precision == 0.5
    assert point.lift == 1.0


def test_capacity_curve_preserves_requested_operating_points() -> None:
    frame = capacity_curve(
        np.array([1, 0, 1, 0, 0]),
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        fractions=[0.2, 0.6],
    )

    assert frame["fraction"].tolist() == [0.2, 0.6]
    assert frame["reviewed_count"].tolist() == [1, 3]


def test_validation_report_exposes_tie_bounds() -> None:
    report = validation_report(
        np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
    )

    assert report["lift_at_10pct_lower"] == 0.0
    assert report["lift_at_10pct"] == 2.5
    assert report["lift_at_10pct_upper"] == 5.0
    assert report["capture_at_10pct_lower"] == 0.0
    assert report["capture_at_10pct"] == 0.25
    assert report["capture_at_10pct_upper"] == 0.5


@pytest.mark.parametrize(
    ("labels", "scores", "fraction", "message"),
    [
        (np.array([]), np.array([]), 0.1, "observation"),
        (np.array([0, 2]), np.array([0.1, 0.9]), 0.5, "0/1"),
        (np.array([0, 1]), np.array([0.1, np.nan]), 0.5, "finite"),
        (np.array([0, 1]), np.array([0.1, 0.9]), 0.0, "fraction"),
        (np.array([0, 0]), np.array([0.1, 0.9]), 0.5, "positive"),
    ],
)
def test_invalid_capacity_inputs_fail_closed(
    labels: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        capacity_at_fraction(labels, scores, fraction)


def test_empty_capacity_curve_is_rejected() -> None:
    with pytest.raises(ValueError, match="fractions"):
        capacity_curve(np.array([0, 1]), np.array([0.1, 0.9]), fractions=[])
