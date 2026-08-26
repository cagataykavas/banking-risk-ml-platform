from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinStat:
    variable: str
    bin_label: str
    count: int
    event_count: int
    non_event_count: int
    event_rate: float
    event_distribution: float
    non_event_distribution: float
    woe: float
    iv_component: float


def _safe_distribution(value: float, total: float, epsilon: float = 0.5) -> float:
    return (value + epsilon) / (total + epsilon)


def woe_iv_from_bins(
    frame: pd.DataFrame,
    *,
    variable: str,
    target: str,
    bin_column: pd.Series,
) -> tuple[pd.DataFrame, float]:
    """Calculate Weight of Evidence and Information Value for pre-defined bins.

    Convention used here:
        WoE = ln(non-event distribution / event distribution)

    A positive WoE therefore indicates a bin that is relatively more concentrated
    among non-events. Conventions differ across organizations, so the sign convention
    should always be documented rather than assumed.
    """
    data = pd.DataFrame(
        {
            "value": frame[variable],
            "target": frame[target].astype(int),
            "bin": bin_column.astype(str),
        }
    )

    total_events = float(data["target"].sum())
    total_non_events = float(len(data) - data["target"].sum())
    if total_events == 0 or total_non_events == 0:
        raise ValueError("target must contain both events and non-events")

    rows: list[BinStat] = []
    for label, group in data.groupby("bin", observed=True, sort=False):
        count = len(group)
        events = int(group["target"].sum())
        non_events = count - events
        event_dist = _safe_distribution(events, total_events)
        non_event_dist = _safe_distribution(non_events, total_non_events)
        woe = float(np.log(non_event_dist / event_dist))
        iv_component = float((non_event_dist - event_dist) * woe)
        rows.append(
            BinStat(
                variable=variable,
                bin_label=str(label),
                count=count,
                event_count=events,
                non_event_count=non_events,
                event_rate=float(events / count),
                event_distribution=event_dist,
                non_event_distribution=non_event_dist,
                woe=woe,
                iv_component=iv_component,
            )
        )

    result = pd.DataFrame([row.__dict__ for row in rows])
    return result, float(result["iv_component"].sum())


def quantile_woe(
    frame: pd.DataFrame,
    *,
    variable: str,
    target: str,
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    numeric = pd.to_numeric(frame[variable], errors="coerce")
    valid = numeric.notna() & frame[target].notna()
    subset = frame.loc[valid].copy()
    if subset.empty:
        raise ValueError(f"no valid rows for {variable}")

    quantiles = pd.qcut(subset[variable], q=bins, duplicates="drop")
    return woe_iv_from_bins(subset, variable=variable, target=target, bin_column=quantiles)


def categorical_woe(
    frame: pd.DataFrame,
    *,
    variable: str,
    target: str,
) -> tuple[pd.DataFrame, float]:
    labels = frame[variable].fillna("__MISSING__").astype(str)
    return woe_iv_from_bins(frame, variable=variable, target=target, bin_column=labels)


def iv_summary(
    frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    target: str,
    bins: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for feature in numeric_features:
        table, iv = quantile_woe(frame, variable=feature, target=target, bins=bins)
        rows.append(
            {
                "feature": feature,
                "kind": "numeric",
                "bins": len(table),
                "information_value": iv,
            }
        )

    for feature in categorical_features:
        table, iv = categorical_woe(frame, variable=feature, target=target)
        rows.append(
            {
                "feature": feature,
                "kind": "categorical",
                "bins": len(table),
                "information_value": iv,
            }
        )

    return pd.DataFrame(rows).sort_values("information_value", ascending=False).reset_index(drop=True)


def iv_strength(iv: float) -> str:
    """Conventional rough interpretation used only as an interview aid.

    IV thresholds are heuristics, not universal scientific laws. Stability, leakage,
    sample size, regulation and business meaning matter more than blindly selecting
    features by a single threshold.
    """
    if iv < 0.02:
        return "very weak"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "suspiciously strong - investigate leakage/stability"


if __name__ == "__main__":
    rng = np.random.default_rng(101)
    rows = 10000
    frame = pd.DataFrame(
        {
            "utilization": rng.beta(2.0, 5.0, rows),
            "late_payments": rng.poisson(0.5, rows),
            "income": rng.lognormal(10.8, 0.5, rows),
            "channel": rng.choice(["branch", "web", "mobile"], rows, p=[0.15, 0.30, 0.55]),
        }
    )
    logit = -4.4 + 4.2 * frame["utilization"] + 0.5 * frame["late_payments"]
    probability = 1 / (1 + np.exp(-logit))
    frame["default"] = rng.binomial(1, probability)

    summary = iv_summary(
        frame,
        numeric_features=["utilization", "late_payments", "income"],
        categorical_features=["channel"],
        target="default",
    )
    summary["interpretation"] = summary["information_value"].map(iv_strength)
    print(summary.to_string(index=False))
