import math

import pytest

from src.vintage_calibration import (
    VintageCalibrationPolicy,
    evaluate_vintage,
    evaluate_vintages,
)

POLICY = VintageCalibrationPolicy(
    min_rows=4,
    min_defaults=1,
    min_oe_ratio=0.5,
    max_oe_ratio=1.5,
    max_abs_calibration_gap=0.15,
)


def test_well_calibrated_vintage_passes_with_interval_evidence():
    result = evaluate_vintage("2026-Q1", [0.2] * 10, [1, 1] + [0] * 8, policy=POLICY)

    assert result.passed
    assert result.oe_ratio == pytest.approx(1.0)
    assert (
        result.observed_rate_ci_low
        < result.observed_rate
        < result.observed_rate_ci_high
    )
    assert result.to_dict()["reasons"] == []


def test_hidden_vintage_underprediction_fails_both_gates():
    result = evaluate_vintage("2026-Q2", [0.05] * 20, [1] * 5 + [0] * 15, policy=POLICY)

    assert not result.passed
    assert result.reasons == (
        "oe_ratio_above_policy",
        "calibration_gap_exceeds_policy",
    )


def test_small_or_event_sparse_cohort_fails_closed():
    result = evaluate_vintage("2026-Q3", [0.1, 0.1], [0, 0], policy=POLICY)

    assert result.reasons == (
        "insufficient_rows",
        "insufficient_defaults",
        "oe_ratio_below_policy",
    )


def test_zero_expected_defaults_is_explicit_not_division_error():
    result = evaluate_vintage("2026-Q4", [0.0] * 4, [0, 0, 0, 1], policy=POLICY)

    assert math.isinf(result.oe_ratio)
    assert "zero_expected_defaults" in result.reasons


@pytest.mark.parametrize(
    ("probabilities", "outcomes"),
    [([0.1], []), ([float("nan")], [0]), ([1.1], [1]), ([0.1], [2])],
)
def test_invalid_artifacts_are_rejected(probabilities, outcomes):
    with pytest.raises(ValueError):
        evaluate_vintage("bad", probabilities, outcomes, policy=POLICY)


def test_multi_vintage_report_is_deterministic_and_blocks_any_failure():
    report = evaluate_vintages(
        [
            ("2026-Q2", [0.05] * 20, [1] * 5 + [0] * 15),
            ("2026-Q1", [0.2] * 10, [1, 1] + [0] * 8),
        ],
        policy=POLICY,
    )

    assert not report.passed
    assert report.reasons == ("vintage_failed:2026-Q2",)
    assert [item.vintage for item in report.vintages] == ["2026-Q1", "2026-Q2"]
    assert report.to_dict()["vintages"][1]["passed"] is False
