from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    false_approve_cost: float = 8500.0
    false_decline_cost: float = 450.0
    manual_review_cost: float = 18.0
    true_approve_value: float = 120.0


@dataclass(frozen=True)
class PolicyResult:
    approve_below: float
    decline_above: float
    approval_rate: float
    decline_rate: float
    review_rate: float
    bad_rate_approved: float
    default_capture_rate: float
    expected_cost_per_application: float


def route(scores: np.ndarray, approve_below: float, decline_above: float) -> np.ndarray:
    if approve_below >= decline_above:
        raise ValueError("approve_below must be strictly below decline_above")
    decision = np.full(scores.shape, "review", dtype=object)
    decision[scores < approve_below] = "approve"
    decision[scores >= decline_above] = "decline"
    return decision


def expected_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    approve_below: float,
    decline_above: float,
    costs: CostModel,
) -> PolicyResult:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and scores must have the same shape")

    decisions = route(p, approve_below, decline_above)
    approved = decisions == "approve"
    declined = decisions == "decline"
    reviewed = decisions == "review"

    bad_approved = approved & (y == 1)
    good_declined = declined & (y == 0)
    good_approved = approved & (y == 0)

    total_cost = (
        bad_approved.sum() * costs.false_approve_cost
        + good_declined.sum() * costs.false_decline_cost
        + reviewed.sum() * costs.manual_review_cost
        - good_approved.sum() * costs.true_approve_value
    )

    defaults = max(int((y == 1).sum()), 1)
    return PolicyResult(
        approve_below=float(approve_below),
        decline_above=float(decline_above),
        approval_rate=float(approved.mean()),
        decline_rate=float(declined.mean()),
        review_rate=float(reviewed.mean()),
        bad_rate_approved=float(y[approved].mean()) if approved.any() else 0.0,
        default_capture_rate=float(((declined | reviewed) & (y == 1)).sum() / defaults),
        expected_cost_per_application=float(total_cost / len(y)),
    )


def search_policy(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    approve_grid: Iterable[float] = np.linspace(0.02, 0.25, 24),
    decline_grid: Iterable[float] = np.linspace(0.45, 0.85, 21),
    costs: CostModel = CostModel(),
    max_review_rate: float | None = 0.30,
    min_approval_rate: float | None = 0.20,
    min_default_capture_rate: float | None = 0.80,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for approve_below in approve_grid:
        for decline_above in decline_grid:
            if approve_below >= decline_above:
                continue
            result = expected_cost(
                y_true,
                scores,
                approve_below=float(approve_below),
                decline_above=float(decline_above),
                costs=costs,
            )
            if max_review_rate is not None and result.review_rate > max_review_rate:
                continue
            if min_approval_rate is not None and result.approval_rate < min_approval_rate:
                continue
            if min_default_capture_rate is not None and result.default_capture_rate < min_default_capture_rate:
                continue
            rows.append(result.__dict__)

    if not rows:
        return pd.DataFrame(columns=PolicyResult.__annotations__.keys())

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["expected_cost_per_application", "review_rate", "bad_rate_approved"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def expected_loss_table(
    balances: np.ndarray,
    pd_scores: np.ndarray,
    *,
    lgd: float | np.ndarray = 0.45,
    ead_multiplier: float | np.ndarray = 1.0,
) -> pd.DataFrame:
    """Illustrative expected-loss decomposition: EL = PD × LGD × EAD.

    This is intentionally a transparent portfolio example, not a regulatory IFRS 9
    implementation. It demonstrates how model probabilities connect to economic loss.
    """
    balance = np.asarray(balances, dtype=float)
    pd_score = np.asarray(pd_scores, dtype=float)
    lgd_arr = np.broadcast_to(np.asarray(lgd, dtype=float), balance.shape)
    ead = balance * np.broadcast_to(np.asarray(ead_multiplier, dtype=float), balance.shape)

    if np.any((pd_score < 0) | (pd_score > 1)):
        raise ValueError("PD scores must be in [0, 1]")
    if np.any((lgd_arr < 0) | (lgd_arr > 1)):
        raise ValueError("LGD must be in [0, 1]")

    loss = pd_score * lgd_arr * ead
    return pd.DataFrame(
        {
            "balance": balance,
            "pd": pd_score,
            "lgd": lgd_arr,
            "ead": ead,
            "expected_loss": loss,
        }
    )


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    y = rng.binomial(1, 0.11, 8000)
    latent = -2.4 + 2.1 * y + rng.normal(0, 1.1, len(y))
    scores = 1 / (1 + np.exp(-latent))

    policies = search_policy(y, scores)
    print(policies.head(10).to_string(index=False))

    balances = rng.lognormal(mean=9.5, sigma=0.6, size=5)
    print(expected_loss_table(balances, scores[:5]).to_string(index=False))
