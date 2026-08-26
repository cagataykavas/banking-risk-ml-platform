from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class DecisionPolicy:
    approve_below: float = 0.08
    decline_above: float = 0.65


@dataclass(frozen=True)
class ShadowSummary:
    rows: int
    score_mae: float
    score_p95_abs_delta: float
    route_disagreement_rate: float
    champion_review_rate: float
    challenger_review_rate: float
    review_rate_delta: float
    challenger_latency_p95_ms: float
    latency_p95_ratio: float


@dataclass(frozen=True)
class OutcomeComparison:
    rows: int
    champion_roc_auc: float
    challenger_roc_auc: float
    champion_average_precision: float
    challenger_average_precision: float
    champion_brier: float
    challenger_brier: float
    auc_delta: float
    ap_delta: float
    brier_delta: float


def route(scores: np.ndarray, policy: DecisionPolicy) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    routes = np.full(scores.shape, "review", dtype=object)
    routes[scores < policy.approve_below] = "approve"
    routes[scores >= policy.decline_above] = "decline"
    return routes


def shadow_summary(
    frame: pd.DataFrame,
    *,
    champion_score: str = "champion_score",
    challenger_score: str = "challenger_score",
    champion_latency_ms: str = "champion_latency_ms",
    challenger_latency_ms: str = "challenger_latency_ms",
    policy: DecisionPolicy | None = None,
) -> ShadowSummary:
    active_policy = policy or DecisionPolicy()
    champion = frame[champion_score].to_numpy(dtype=float)
    challenger = frame[challenger_score].to_numpy(dtype=float)
    champion_routes = route(champion, active_policy)
    challenger_routes = route(challenger, active_policy)

    score_delta = np.abs(challenger - champion)
    champion_p95 = float(np.quantile(frame[champion_latency_ms].to_numpy(dtype=float), 0.95))
    challenger_p95 = float(np.quantile(frame[challenger_latency_ms].to_numpy(dtype=float), 0.95))
    champion_review = float(np.mean(champion_routes == "review"))
    challenger_review = float(np.mean(challenger_routes == "review"))

    return ShadowSummary(
        rows=len(frame),
        score_mae=float(np.mean(score_delta)),
        score_p95_abs_delta=float(np.quantile(score_delta, 0.95)),
        route_disagreement_rate=float(np.mean(champion_routes != challenger_routes)),
        champion_review_rate=champion_review,
        challenger_review_rate=challenger_review,
        review_rate_delta=challenger_review - champion_review,
        challenger_latency_p95_ms=challenger_p95,
        latency_p95_ratio=challenger_p95 / max(champion_p95, 1e-9),
    )


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")


def outcome_comparison(
    mature_frame: pd.DataFrame,
    *,
    target: str = "default_label",
    champion_score: str = "champion_score",
    challenger_score: str = "challenger_score",
) -> OutcomeComparison:
    clean = mature_frame[[target, champion_score, challenger_score]].dropna()
    if clean.empty:
        raise ValueError("no mature labeled rows available")

    y = clean[target].to_numpy(dtype=int)
    champion = clean[champion_score].to_numpy(dtype=float)
    challenger = clean[challenger_score].to_numpy(dtype=float)

    champion_auc = _safe_auc(y, champion)
    challenger_auc = _safe_auc(y, challenger)
    champion_ap = _safe_ap(y, champion)
    challenger_ap = _safe_ap(y, challenger)
    champion_brier = float(brier_score_loss(y, champion))
    challenger_brier = float(brier_score_loss(y, challenger))

    return OutcomeComparison(
        rows=len(clean),
        champion_roc_auc=champion_auc,
        challenger_roc_auc=challenger_auc,
        champion_average_precision=champion_ap,
        challenger_average_precision=challenger_ap,
        champion_brier=champion_brier,
        challenger_brier=challenger_brier,
        auc_delta=challenger_auc - champion_auc,
        ap_delta=challenger_ap - champion_ap,
        brier_delta=challenger_brier - champion_brier,
    )


def disagreement_slice(
    frame: pd.DataFrame,
    *,
    policy: DecisionPolicy | None = None,
    champion_score: str = "champion_score",
    challenger_score: str = "challenger_score",
) -> pd.DataFrame:
    active_policy = policy or DecisionPolicy()
    out = frame.copy()
    out["champion_route"] = route(out[champion_score].to_numpy(dtype=float), active_policy)
    out["challenger_route"] = route(out[challenger_score].to_numpy(dtype=float), active_policy)
    out["absolute_score_delta"] = (out[challenger_score] - out[champion_score]).abs()
    return out.loc[out["champion_route"] != out["challenger_route"]].sort_values(
        "absolute_score_delta",
        ascending=False,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(62)
    rows = 5000
    champion = np.clip(rng.beta(1.4, 8.5, rows), 0.001, 0.98)
    challenger = np.clip(champion + rng.normal(0.0, 0.035, rows), 0.001, 0.999)
    data = pd.DataFrame(
        {
            "application_id": [f"app-{index:05d}" for index in range(rows)],
            "champion_score": champion,
            "challenger_score": challenger,
            "champion_latency_ms": rng.lognormal(3.0, 0.25, rows),
            "challenger_latency_ms": rng.lognormal(3.1, 0.28, rows),
            "default_label": rng.binomial(1, champion),
        }
    )

    print(shadow_summary(data))
    print(outcome_comparison(data))
    print(disagreement_slice(data).head(10).to_string(index=False))
