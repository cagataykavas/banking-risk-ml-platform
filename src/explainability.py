from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.risk_pipeline import CATEGORICAL, NUMERIC


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    observed: Any
    counterfactual: Any
    probability_delta: float
    direction: str


def _counterfactual_value(feature: str, value: Any) -> Any:
    """Return a conservative reference value for a local perturbation.

    This is intentionally model-agnostic and easy to explain in interviews. It
    does not pretend to be SHAP: each feature is perturbed independently and the
    resulting probability delta is reported as a local sensitivity measure.
    """
    references: dict[str, Any] = {
        "age": 35,
        "income": 60_000.0,
        "debt": 8_000.0,
        "utilization": 0.30,
        "late_payments": 0,
        "account_age_months": 72,
        "employment": "salaried",
        "housing": "rent",
    }
    return references.get(feature, value)


def explain_prediction(
    model, payload: dict[str, Any], top_k: int = 5
) -> dict[str, Any]:
    """Score the baseline and independent reference perturbations in one batch.

    Requires a row-independent binary classifier whose classes include label 1
    (default). Batch-dependent or stateful predictors are not supported.
    """
    features = NUMERIC + CATEGORICAL
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not 1 <= top_k <= len(features):
        raise ValueError(f"top_k must be between 1 and {len(features)}")

    classes = np.asarray(getattr(model, "classes_", None))
    if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
        raise ValueError("explanations require binary classes 0 and 1")
    default_column = int(np.flatnonzero(classes == 1)[0])

    baseline_row = {name: payload[name] for name in features}
    references = {
        feature: _counterfactual_value(feature, payload[feature])
        for feature in features
    }
    rows = [baseline_row] + [
        {**baseline_row, feature: references[feature]} for feature in features
    ]
    probabilities = np.asarray(model.predict_proba(pd.DataFrame(rows)), dtype=float)
    if probabilities.shape != (len(rows), 2):
        raise ValueError(
            "predict_proba must return one binary probability row per input"
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("predict_proba returned invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("binary probabilities must sum to one")

    baseline = float(probabilities[0, default_column])
    contributions: list[FeatureContribution] = []

    for index, feature in enumerate(features, start=1):
        reference = references[feature]
        counterfactual_probability = float(probabilities[index, default_column])
        delta = baseline - counterfactual_probability
        contributions.append(
            FeatureContribution(
                feature=feature,
                observed=payload[feature],
                counterfactual=reference,
                probability_delta=delta,
                direction=(
                    "no_change"
                    if abs(delta) <= 1e-12
                    else "increases_risk"
                    if delta > 0
                    else "decreases_risk"
                ),
            )
        )

    ranked = sorted(
        contributions, key=lambda item: abs(item.probability_delta), reverse=True
    )
    return {
        "probability": baseline,
        "method": "one_feature_counterfactual_sensitivity",
        "caveat": "Local perturbation explanation; feature interactions are not allocated additively.",
        "top_contributors": [item.__dict__ for item in ranked[:top_k]],
    }
