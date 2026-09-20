from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from statistics import NormalDist


@dataclass(frozen=True)
class VintageCalibrationPolicy:
    min_rows: int = 200
    min_defaults: int = 10
    min_oe_ratio: float = 0.75
    max_oe_ratio: float = 1.25
    max_abs_calibration_gap: float = 0.03
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.min_rows < 1 or self.min_defaults < 0:
            raise ValueError("sample thresholds must be non-negative")
        if not 0 < self.min_oe_ratio <= self.max_oe_ratio:
            raise ValueError("O/E bounds must be positive and ordered")
        if not 0 <= self.max_abs_calibration_gap <= 1:
            raise ValueError("calibration-gap tolerance must be in [0, 1]")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")


@dataclass(frozen=True)
class VintageCalibrationResult:
    vintage: str
    passed: bool
    reasons: tuple[str, ...]
    rows: int
    observed_defaults: int
    expected_defaults: float
    observed_rate: float
    expected_rate: float
    calibration_gap: float
    oe_ratio: float
    observed_rate_ci_low: float
    observed_rate_ci_high: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class VintageCalibrationReport:
    passed: bool
    reasons: tuple[str, ...]
    vintages: tuple[VintageCalibrationResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "vintages": [vintage.to_dict() for vintage in self.vintages],
        }


def _wilson_interval(
    events: int, rows: int, confidence_level: float
) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    rate = events / rows
    denominator = 1 + z * z / rows
    centre = (rate + z * z / (2 * rows)) / denominator
    radius = z * sqrt((rate * (1 - rate) + z * z / (4 * rows)) / rows) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def evaluate_vintage(
    vintage: str,
    predicted_probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    policy: VintageCalibrationPolicy | None = None,
) -> VintageCalibrationResult:
    active_policy = policy or VintageCalibrationPolicy()
    probabilities = [float(value) for value in predicted_probabilities]
    labels = list(outcomes)

    if not vintage.strip():
        raise ValueError("vintage must not be empty")
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError(
            "probabilities and outcomes must be non-empty and equal length"
        )
    if any(not isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if any(type(value) not in (bool, int) or value not in (0, 1) for value in labels):
        raise ValueError("outcomes must be binary integers")

    rows = len(labels)
    observed_defaults = int(sum(labels))
    expected_defaults = sum(probabilities)
    observed_rate = observed_defaults / rows
    expected_rate = expected_defaults / rows
    calibration_gap = observed_rate - expected_rate
    oe_ratio = (
        observed_defaults / expected_defaults if expected_defaults > 0 else float("inf")
    )
    ci_low, ci_high = _wilson_interval(
        observed_defaults, rows, active_policy.confidence_level
    )

    reasons: list[str] = []
    if rows < active_policy.min_rows:
        reasons.append("insufficient_rows")
    if observed_defaults < active_policy.min_defaults:
        reasons.append("insufficient_defaults")
    if expected_defaults <= 0:
        reasons.append("zero_expected_defaults")
    elif oe_ratio < active_policy.min_oe_ratio:
        reasons.append("oe_ratio_below_policy")
    elif oe_ratio > active_policy.max_oe_ratio:
        reasons.append("oe_ratio_above_policy")
    if abs(calibration_gap) > active_policy.max_abs_calibration_gap:
        reasons.append("calibration_gap_exceeds_policy")

    return VintageCalibrationResult(
        vintage=vintage,
        passed=not reasons,
        reasons=tuple(reasons),
        rows=rows,
        observed_defaults=observed_defaults,
        expected_defaults=expected_defaults,
        observed_rate=observed_rate,
        expected_rate=expected_rate,
        calibration_gap=calibration_gap,
        oe_ratio=oe_ratio,
        observed_rate_ci_low=ci_low,
        observed_rate_ci_high=ci_high,
    )


def evaluate_vintages(
    cohorts: Sequence[tuple[str, Sequence[float], Sequence[int]]],
    *,
    policy: VintageCalibrationPolicy | None = None,
) -> VintageCalibrationReport:
    if not cohorts:
        raise ValueError("at least one vintage is required")
    names = [name for name, _, _ in cohorts]
    if len(names) != len(set(names)):
        raise ValueError("vintage names must be unique")

    results = tuple(
        evaluate_vintage(name, probabilities, outcomes, policy=policy)
        for name, probabilities, outcomes in sorted(cohorts, key=lambda item: item[0])
    )
    failed = [result.vintage for result in results if not result.passed]
    reasons = tuple(f"vintage_failed:{name}" for name in failed)
    return VintageCalibrationReport(
        passed=not failed, reasons=reasons, vintages=results
    )
