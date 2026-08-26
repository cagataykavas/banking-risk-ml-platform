from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")


def validate_segments(
    frame: pd.DataFrame,
    *,
    segment_column: str,
    target_column: str,
    score_column: str,
    approve_below: float = 0.08,
    decline_above: float = 0.65,
    min_rows: int = 200,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | bool]] = []

    for segment, group in frame.groupby(segment_column, dropna=False):
        y = group[target_column].to_numpy(dtype=int)
        score = group[score_column].to_numpy(dtype=float)

        approved = score < approve_below
        reviewed = (score >= approve_below) & (score < decline_above)
        declined = score >= decline_above

        rows.append(
            {
                "segment": str(segment),
                "rows": int(len(group)),
                "small_sample_warning": bool(len(group) < min_rows),
                "event_rate": float(np.mean(y)),
                "mean_score": float(np.mean(score)),
                "roc_auc": _safe_auc(y, score),
                "average_precision": _safe_ap(y, score),
                "brier_score": float(brier_score_loss(y, score)),
                "approval_rate": float(np.mean(approved)),
                "review_rate": float(np.mean(reviewed)),
                "decline_rate": float(np.mean(declined)),
                "bad_rate_approved": float(np.mean(y[approved])) if approved.any() else float("nan"),
                "default_capture_rate": float(np.sum(y[reviewed | declined]) / max(np.sum(y), 1)),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    overall_event_rate = float(frame[target_column].mean())
    overall_approval_rate = float((frame[score_column] < approve_below).mean())
    overall_review_rate = float(
        ((frame[score_column] >= approve_below) & (frame[score_column] < decline_above)).mean()
    )

    result["event_rate_delta_vs_overall"] = result["event_rate"] - overall_event_rate
    result["approval_rate_delta_vs_overall"] = result["approval_rate"] - overall_approval_rate
    result["review_rate_delta_vs_overall"] = result["review_rate"] - overall_review_rate

    return result.sort_values(["small_sample_warning", "rows"], ascending=[True, False]).reset_index(drop=True)


def stability_flags(
    report: pd.DataFrame,
    *,
    max_auc_gap: float = 0.10,
    max_approval_gap: float = 0.20,
    max_review_gap: float = 0.20,
) -> pd.DataFrame:
    """Flag large cross-segment differences for investigation, not automatic conclusions."""
    if report.empty:
        return report.copy()

    out = report.copy()
    auc_values = out["roc_auc"].dropna()
    auc_reference = float(auc_values.median()) if not auc_values.empty else float("nan")

    out["auc_gap_flag"] = (
        (out["roc_auc"] - auc_reference).abs() > max_auc_gap
        if np.isfinite(auc_reference)
        else False
    )
    out["approval_gap_flag"] = out["approval_rate_delta_vs_overall"].abs() > max_approval_gap
    out["review_gap_flag"] = out["review_rate_delta_vs_overall"].abs() > max_review_gap
    out["requires_investigation"] = (
        out["auc_gap_flag"]
        | out["approval_gap_flag"]
        | out["review_gap_flag"]
        | out["small_sample_warning"]
    )
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    n = 6000
    countries = rng.choice(["TR", "DE", "NL", "PL"], n, p=[0.35, 0.25, 0.20, 0.20])
    base = np.select(
        [countries == "TR", countries == "DE", countries == "NL", countries == "PL"],
        [-2.6, -2.9, -3.0, -2.7],
    )
    latent = base + rng.normal(0, 1, n)
    score = 1 / (1 + np.exp(-latent))
    outcome = rng.binomial(1, np.clip(score * 1.05, 0, 1))

    demo = pd.DataFrame({"country": countries, "score": score, "default": outcome})
    report = validate_segments(
        demo,
        segment_column="country",
        target_column="default",
        score_column="score",
    )
    print(stability_flags(report).to_string(index=False))
