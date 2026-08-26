from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftSignal:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    severity: str


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("reference and current must contain finite values")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(ref), np.inf], dtype=float)
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_dist = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_dist = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def severity_from_psi(value: float) -> str:
    if value >= 0.25:
        return "alert"
    if value >= 0.10:
        return "warning"
    return "stable"


def numeric_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[DriftSignal] = []
    for feature in features:
        ref = reference[feature].dropna().to_numpy(dtype=float)
        cur = current[feature].dropna().to_numpy(dtype=float)
        ks = ks_2samp(ref, cur)
        psi_value = psi(ref, cur)
        rows.append(
            DriftSignal(
                feature=feature,
                psi=psi_value,
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                reference_mean=float(np.mean(ref)),
                current_mean=float(np.mean(cur)),
                severity=severity_from_psi(psi_value),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows]).sort_values("psi", ascending=False)


def score_stability(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict[str, float | str]:
    psi_value = psi(reference_scores, current_scores)
    ks = ks_2samp(reference_scores, current_scores)
    return {
        "score_psi": psi_value,
        "score_ks": float(ks.statistic),
        "score_ks_pvalue": float(ks.pvalue),
        "reference_mean_score": float(np.mean(reference_scores)),
        "current_mean_score": float(np.mean(current_scores)),
        "severity": severity_from_psi(psi_value),
    }


def monthly_drift_report(
    frame: pd.DataFrame,
    *,
    time_column: str,
    features: list[str],
    reference_month: str,
) -> pd.DataFrame:
    data = frame.copy()
    data[time_column] = pd.to_datetime(data[time_column], utc=True)
    data["month"] = data[time_column].dt.to_period("M").astype(str)
    reference = data[data["month"] == reference_month]
    if reference.empty:
        raise ValueError(f"reference month {reference_month!r} is empty")

    rows: list[dict] = []
    for month, current in data.groupby("month"):
        if month == reference_month:
            continue
        report = numeric_drift(reference, current, features)
        for record in report.to_dict(orient="records"):
            rows.append({"month": month, **record})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(21)
    reference = pd.DataFrame({
        "income": rng.lognormal(10.8, 0.4, 5000),
        "utilization": rng.beta(2, 5, 5000),
        "late_payments": rng.poisson(0.6, 5000),
    })
    current = pd.DataFrame({
        "income": rng.lognormal(10.7, 0.46, 4000),
        "utilization": np.clip(rng.beta(2.4, 4.4, 4000), 0, 1),
        "late_payments": rng.poisson(0.95, 4000),
    })
    print(numeric_drift(reference, current, list(reference.columns)).to_string(index=False))
