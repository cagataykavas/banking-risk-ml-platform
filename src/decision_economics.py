from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicsAssumptions:
    approved_account_revenue: float = 220.0
    loss_given_default: float = 2800.0
    manual_review_cost: float = 18.0
    false_decline_opportunity_cost: float = 120.0
    review_default_capture_rate: float = 0.72


@dataclass(frozen=True)
class PolicyEconomics:
    approve_below: float
    reject_above: float
    approval_rate: float
    review_rate: float
    reject_rate: float
    bad_rate_approved: float
    expected_value_per_application: float
    expected_total_value: float


def _validate(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true and scores must be one-dimensional arrays of equal length")
    if len(y) == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must contain only 0/1 labels")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("scores must be probabilities in [0, 1]")
    return y, p


def evaluate_policy(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    approve_below: float,
    reject_above: float,
    assumptions: EconomicsAssumptions | None = None,
) -> PolicyEconomics:
    """Estimate policy value using realized labels for an offline decision simulation.

    This is deliberately a transparent business layer on top of model probabilities.
    It is not presented as a causal profit estimate: selection effects, credit limits,
    exposure at default, discounting and portfolio constraints would need richer data.
    """
    if not 0.0 <= approve_below <= reject_above <= 1.0:
        raise ValueError("expected 0 <= approve_below <= reject_above <= 1")

    assumptions = assumptions or EconomicsAssumptions()
    y, p = _validate(y_true, scores)

    approve = p < approve_below
    review = (p >= approve_below) & (p < reject_above)
    reject = p >= reject_above

    value = np.zeros(len(y), dtype=float)

    approved_good = approve & (y == 0)
    approved_bad = approve & (y == 1)
    value[approved_good] += assumptions.approved_account_revenue
    value[approved_bad] += assumptions.approved_account_revenue - assumptions.loss_given_default

    reviewed_good = review & (y == 0)
    reviewed_bad = review & (y == 1)
    value[review] -= assumptions.manual_review_cost
    value[reviewed_good] += assumptions.approved_account_revenue

    bad_capture = assumptions.review_default_capture_rate
    value[reviewed_bad] += (
        (1.0 - bad_capture)
        * (assumptions.approved_account_revenue - assumptions.loss_given_default)
    )

    rejected_good = reject & (y == 0)
    value[rejected_good] -= assumptions.false_decline_opportunity_cost

    approved_bad_count = int(np.sum(approved_bad))
    approved_count = int(np.sum(approve))

    return PolicyEconomics(
        approve_below=float(approve_below),
        reject_above=float(reject_above),
        approval_rate=float(np.mean(approve)),
        review_rate=float(np.mean(review)),
        reject_rate=float(np.mean(reject)),
        bad_rate_approved=approved_bad_count / approved_count if approved_count else 0.0,
        expected_value_per_application=float(np.mean(value)),
        expected_total_value=float(np.sum(value)),
    )


def optimize_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    approve_grid: np.ndarray | None = None,
    reject_grid: np.ndarray | None = None,
    assumptions: EconomicsAssumptions | None = None,
    max_review_rate: float | None = None,
) -> pd.DataFrame:
    """Grid-search two-threshold policies and rank them by offline expected value."""
    approve_values = np.asarray(
        approve_grid if approve_grid is not None else np.linspace(0.03, 0.30, 10),
        dtype=float,
    )
    reject_values = np.asarray(
        reject_grid if reject_grid is not None else np.linspace(0.35, 0.80, 10),
        dtype=float,
    )
    if max_review_rate is not None and not 0.0 <= max_review_rate <= 1.0:
        raise ValueError("max_review_rate must be in [0, 1]")

    rows: list[dict[str, float]] = []
    for approve_below in approve_values:
        for reject_above in reject_values:
            if approve_below > reject_above:
                continue
            result = evaluate_policy(
                y_true,
                scores,
                approve_below=float(approve_below),
                reject_above=float(reject_above),
                assumptions=assumptions,
            )
            if max_review_rate is not None and result.review_rate > max_review_rate:
                continue
            rows.append(asdict(result))

    if not rows:
        return pd.DataFrame(columns=list(PolicyEconomics.__dataclass_fields__))
    return pd.DataFrame(rows).sort_values(
        ["expected_value_per_application", "approval_rate"],
        ascending=[False, False],
        ignore_index=True,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    labels = rng.binomial(1, 0.11, size=8000)
    logits = -2.6 + 2.2 * labels + rng.normal(0.0, 1.0, size=len(labels))
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    print(optimize_thresholds(labels, probabilities, max_review_rate=0.35).head(10).to_string())
