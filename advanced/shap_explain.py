from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass(frozen=True)
class LocalExplanation:
    base_value: float
    prediction: float
    contributions: list[dict[str, float]]


def explain_tree_model(
    model,
    x_background: pd.DataFrame,
    x_rows: pd.DataFrame,
    *,
    top_n: int = 8,
) -> list[LocalExplanation]:
    """Return local SHAP contributions for a tree model.

    This module intentionally uses the SHAP library and labels the output as SHAP.
    The simpler perturbation explainer elsewhere in this repository is deliberately
    described as local sensitivity analysis instead of being mislabeled.
    """
    explainer = shap.TreeExplainer(
        model,
        data=x_background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(x_rows)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    predictions = np.asarray(model.predict_proba(x_rows)[:, 1])

    results: list[LocalExplanation] = []
    for row_idx in range(len(x_rows)):
        row_values = values[row_idx]
        order = np.argsort(np.abs(row_values))[::-1][:top_n]
        contributions = [
            {
                "feature": str(x_rows.columns[index]),
                "value": float(x_rows.iloc[row_idx, index]),
                "shap_value": float(row_values[index]),
            }
            for index in order
        ]
        results.append(
            LocalExplanation(
                base_value=float(np.ravel(base_values[row_idx])[0]),
                prediction=float(predictions[row_idx]),
                contributions=contributions,
            )
        )
    return results


def global_importance(
    model,
    x_background: pd.DataFrame,
    x_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(
        model,
        data=x_background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    values = np.asarray(explainer(x_evaluation).values)
    importance = np.mean(np.abs(values), axis=0)
    frame = pd.DataFrame(
        {
            "feature": x_evaluation.columns,
            "mean_abs_shap": importance,
        }
    )
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def sanity_check_additivity(explanation: LocalExplanation, *, tolerance: float = 0.03) -> bool:
    reconstructed = explanation.base_value + sum(item["shap_value"] for item in explanation.contributions)
    # With top-N truncation, exact additivity is not expected. This check is useful only
    # when top_n includes every feature or when omitted contributions are negligible.
    return abs(reconstructed - explanation.prediction) <= tolerance
