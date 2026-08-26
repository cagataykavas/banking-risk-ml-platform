from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class FeatureStability:
    feature: str
    mean_importance: float
    std_importance: float
    coefficient_of_variation: float
    sign_consistency: float
    top_k_frequency: float


def summarize_importance_runs(
    runs: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    run_column: str = "run",
    top_k: int = 10,
) -> pd.DataFrame:
    required = {feature_column, importance_column, run_column}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    run_ids = runs[run_column].drop_duplicates().tolist()
    if not run_ids:
        return pd.DataFrame(columns=FeatureStability.__annotations__.keys())

    top_features: dict[object, set[str]] = {}
    for run_id, group in runs.groupby(run_column):
        ordered = group.assign(abs_importance=group[importance_column].abs()).sort_values(
            "abs_importance", ascending=False
        )
        top_features[run_id] = set(ordered.head(top_k)[feature_column].astype(str))

    rows: list[FeatureStability] = []
    for feature, group in runs.groupby(feature_column):
        values = group[importance_column].to_numpy(dtype=float)
        abs_values = np.abs(values)
        mean_abs = float(np.mean(abs_values))
        std_abs = float(np.std(abs_values))
        cv = std_abs / max(mean_abs, 1e-12)
        nonzero = values[np.abs(values) > 1e-12]
        if len(nonzero):
            dominant_sign = np.sign(np.median(nonzero))
            sign_consistency = float(np.mean(np.sign(nonzero) == dominant_sign))
        else:
            sign_consistency = 1.0

        frequency = float(
            np.mean([str(feature) in top_features.get(run_id, set()) for run_id in run_ids])
        )
        rows.append(
            FeatureStability(
                feature=str(feature),
                mean_importance=mean_abs,
                std_importance=std_abs,
                coefficient_of_variation=cv,
                sign_consistency=sign_consistency,
                top_k_frequency=frequency,
            )
        )

    frame = pd.DataFrame([row.__dict__ for row in rows])
    return frame.sort_values(
        ["top_k_frequency", "mean_importance", "sign_consistency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def rank_stability_matrix(
    runs: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    run_column: str = "run",
) -> pd.DataFrame:
    pivot = runs.pivot_table(
        index=feature_column,
        columns=run_column,
        values=importance_column,
        aggfunc="mean",
        fill_value=0.0,
    ).abs()

    ranks = pivot.rank(axis=0, ascending=False, method="average")
    columns = list(ranks.columns)
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            correlation = spearmanr(ranks[left], ranks[right]).statistic
            value = float(correlation) if np.isfinite(correlation) else 0.0
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def flag_unstable_features(
    summary: pd.DataFrame,
    *,
    max_cv: float = 1.0,
    min_sign_consistency: float = 0.75,
    min_top_k_frequency: float = 0.50,
) -> pd.DataFrame:
    mask = (
        (summary["coefficient_of_variation"] > max_cv)
        | (summary["sign_consistency"] < min_sign_consistency)
        | (summary["top_k_frequency"] < min_top_k_frequency)
    )
    return summary.loc[mask].copy()


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    records = []
    features = ["utilization", "late_payments", "income", "debt_ratio", "account_age"]
    base = np.array([0.42, 0.31, -0.12, 0.22, -0.08])
    for run in range(12):
        values = base + rng.normal(0, [0.03, 0.04, 0.09, 0.04, 0.05])
        for feature, importance in zip(features, values, strict=True):
            records.append({"run": run, "feature": feature, "importance": importance})

    frame = pd.DataFrame(records)
    summary = summarize_importance_runs(frame, top_k=3)
    print(summary.to_string(index=False))
    print("\nRank stability:\n", rank_stability_matrix(frame))
