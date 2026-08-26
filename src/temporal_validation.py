from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_rows: int
    test_rows: int
    event_rate: float
    roc_auc: float
    average_precision: float
    brier_score: float


def rolling_time_splits(
    df: pd.DataFrame,
    *,
    time_column: str,
    min_train_days: int = 180,
    test_days: int = 30,
    step_days: int = 30,
    gap_days: int = 1,
):
    ordered = df.sort_values(time_column).copy()
    ordered[time_column] = pd.to_datetime(ordered[time_column], utc=True)
    start = ordered[time_column].min()
    end = ordered[time_column].max()

    fold = 0
    train_end = start + pd.Timedelta(days=min_train_days)
    while train_end + pd.Timedelta(days=gap_days + test_days) <= end:
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = test_start + pd.Timedelta(days=test_days)

        train_mask = ordered[time_column] < train_end
        test_mask = (ordered[time_column] >= test_start) & (ordered[time_column] < test_end)

        train = ordered.loc[train_mask]
        test = ordered.loc[test_mask]
        if not train.empty and not test.empty:
            yield fold, train, test
            fold += 1

        train_end += pd.Timedelta(days=step_days)


def evaluate_temporally(
    estimator,
    df: pd.DataFrame,
    *,
    features: list[str],
    target: str,
    time_column: str,
    min_train_days: int = 180,
    test_days: int = 30,
    step_days: int = 30,
    gap_days: int = 1,
) -> pd.DataFrame:
    rows: list[FoldResult] = []

    for fold, train, test in rolling_time_splits(
        df,
        time_column=time_column,
        min_train_days=min_train_days,
        test_days=test_days,
        step_days=step_days,
        gap_days=gap_days,
    ):
        model = clone(estimator)
        model.fit(train[features], train[target])
        probability = model.predict_proba(test[features])[:, 1]

        y = test[target].to_numpy()
        roc = float(roc_auc_score(y, probability)) if np.unique(y).size > 1 else float("nan")
        ap = float(average_precision_score(y, probability)) if np.unique(y).size > 1 else float("nan")

        rows.append(
            FoldResult(
                fold=fold,
                train_start=str(train[time_column].min()),
                train_end=str(train[time_column].max()),
                test_start=str(test[time_column].min()),
                test_end=str(test[time_column].max()),
                train_rows=len(train),
                test_rows=len(test),
                event_rate=float(np.mean(y)),
                roc_auc=roc,
                average_precision=ap,
                brier_score=float(brier_score_loss(y, probability)),
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def stability_summary(folds: pd.DataFrame) -> dict[str, float]:
    if folds.empty:
        return {}

    result: dict[str, float] = {}
    for metric in ["roc_auc", "average_precision", "brier_score", "event_rate"]:
        values = folds[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values))
        result[f"{metric}_worst"] = float(np.min(values) if metric != "brier_score" else np.max(values))
    return result


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(9)
    dates = pd.date_range("2024-01-01", periods=600, freq="D", tz="UTC")
    rows = []
    for day_index, day in enumerate(dates):
        for _ in range(20):
            income = rng.lognormal(10.8, 0.45)
            utilization = np.clip(rng.beta(2, 5) + 0.00025 * day_index, 0, 1)
            late = rng.poisson(0.5 + 0.0008 * day_index)
            logit = -3.2 + 3.0 * utilization + 0.35 * late - 0.000008 * income
            default = rng.binomial(1, 1 / (1 + np.exp(-logit)))
            rows.append((day, income, utilization, late, default))

    frame = pd.DataFrame(rows, columns=["event_time", "income", "utilization", "late_payments", "default"])
    report = evaluate_temporally(
        LogisticRegression(max_iter=1000),
        frame,
        features=["income", "utilization", "late_payments"],
        target="default",
        time_column="event_time",
    )
    print(report.to_string(index=False))
    print(stability_summary(report))
