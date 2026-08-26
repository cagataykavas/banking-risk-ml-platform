from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class LabelMaturityPolicy:
    outcome_horizon_days: int = 90
    reporting_lag_days: int = 7
    min_rows: int = 200

    @property
    def maturity_days(self) -> int:
        return self.outcome_horizon_days + self.reporting_lag_days


@dataclass(frozen=True)
class CohortMetric:
    cohort: str
    rows: int
    event_rate: float
    mean_score: float
    roc_auc: float
    average_precision: float
    brier_score: float
    calibration_gap: float


def mature_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    policy: LabelMaturityPolicy = LabelMaturityPolicy(),
    id_column: str = "application_id",
    decision_time_column: str = "decision_time",
    outcome_time_column: str = "observed_at",
    target_column: str = "default_label",
) -> pd.DataFrame:
    """Join predictions to labels only after the outcome window is old enough.

    A recently booked account without a 90-day adverse event cannot safely be treated
    as a negative before its label horizon matures. The maturity cutoff prevents that
    common monitoring bias.
    """
    pred = predictions.copy()
    out = outcomes.copy()
    pred[decision_time_column] = pd.to_datetime(pred[decision_time_column], utc=True)
    out[outcome_time_column] = pd.to_datetime(out[outcome_time_column], utc=True)
    current = pd.Timestamp(as_of)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")

    cutoff = current - pd.Timedelta(days=policy.maturity_days)
    eligible = pred.loc[pred[decision_time_column] <= cutoff].copy()
    joined = eligible.merge(
        out[[id_column, target_column, outcome_time_column]],
        on=id_column,
        how="left",
        validate="one_to_one",
    )

    # A matured cohort with a genuinely missing outcome is a data-quality problem,
    # not an implied negative label. Keep it visible and exclude it from metrics.
    joined["label_missing"] = joined[target_column].isna()
    return joined


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan")


def cohort_performance(
    mature: pd.DataFrame,
    *,
    score_column: str = "default_probability",
    target_column: str = "default_label",
    decision_time_column: str = "decision_time",
    frequency: str = "M",
    min_rows: int = 200,
) -> pd.DataFrame:
    labeled = mature.loc[~mature["label_missing"]].copy()
    if labeled.empty:
        return pd.DataFrame(columns=CohortMetric.__annotations__.keys())

    labeled[decision_time_column] = pd.to_datetime(labeled[decision_time_column], utc=True)
    labeled["cohort"] = labeled[decision_time_column].dt.to_period(frequency).astype(str)

    rows: list[CohortMetric] = []
    for cohort, group in labeled.groupby("cohort", sort=True):
        if len(group) < min_rows:
            continue
        y = group[target_column].to_numpy(dtype=int)
        score = group[score_column].to_numpy(dtype=float)
        event_rate = float(np.mean(y))
        mean_score = float(np.mean(score))
        rows.append(
            CohortMetric(
                cohort=str(cohort),
                rows=len(group),
                event_rate=event_rate,
                mean_score=mean_score,
                roc_auc=_safe_auc(y, score),
                average_precision=_safe_ap(y, score),
                brier_score=float(brier_score_loss(y, score)),
                calibration_gap=event_rate - mean_score,
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def maturity_report(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    policy: LabelMaturityPolicy = LabelMaturityPolicy(),
) -> dict[str, object]:
    mature = mature_predictions(predictions, outcomes, as_of=as_of, policy=policy)
    missing = int(mature["label_missing"].sum()) if not mature.empty else 0
    labeled = int((~mature["label_missing"]).sum()) if not mature.empty else 0
    metrics = cohort_performance(mature, min_rows=policy.min_rows)
    return {
        "as_of": str(as_of),
        "maturity_days": policy.maturity_days,
        "eligible_predictions": len(mature),
        "mature_labels": labeled,
        "missing_mature_labels": missing,
        "missing_label_rate": float(missing / max(len(mature), 1)),
        "cohorts": metrics.to_dict(orient="records"),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(88)
    decision_time = pd.date_range("2025-01-01", periods=1500, freq="12h", tz="UTC")
    score = np.clip(rng.beta(1.4, 8.0, len(decision_time)), 0.001, 0.95)
    predictions = pd.DataFrame(
        {
            "application_id": [f"app-{i:05d}" for i in range(len(decision_time))],
            "decision_time": decision_time,
            "default_probability": score,
        }
    )
    target = rng.binomial(1, score)
    outcomes = pd.DataFrame(
        {
            "application_id": predictions["application_id"],
            "default_label": target,
            "observed_at": decision_time + pd.Timedelta(days=90),
        }
    )
    report = maturity_report(
        predictions,
        outcomes,
        as_of=pd.Timestamp("2026-08-26", tz="UTC"),
        policy=LabelMaturityPolicy(min_rows=30),
    )
    print(report)
