from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def explain_prediction(model, payload: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    features = NUMERIC + CATEGORICAL
    frame = pd.DataFrame([{name: payload[name] for name in features}])
    baseline = float(model.predict_proba(frame)[0, 1])
    contributions: list[FeatureContribution] = []

    for feature in features:
        changed = frame.copy()
        reference = _counterfactual_value(feature, payload[feature])
        changed.loc[0, feature] = reference
        counterfactual_probability = float(model.predict_proba(changed)[0, 1])
        delta = baseline - counterfactual_probability
        contributions.append(
            FeatureContribution(
                feature=feature,
                observed=payload[feature],
                counterfactual=reference,
                probability_delta=delta,
                direction="increases_risk" if delta > 0 else "decreases_risk",
            )
        )

    ranked = sorted(contributions, key=lambda item: abs(item.probability_delta), reverse=True)
    return {
        "probability": baseline,
        "method": "one_feature_counterfactual_sensitivity",
        "caveat": "Local perturbation explanation; feature interactions are not allocated additively.",
        "top_contributors": [item.__dict__ for item in ranked[:top_k]],
    }
