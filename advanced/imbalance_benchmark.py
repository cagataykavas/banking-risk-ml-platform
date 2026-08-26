from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Result:
    strategy: str
    roc_auc: float
    average_precision: float
    brier_score: float
    recall_at_target_precision: float
    threshold_at_target_precision: float


def recall_at_precision(y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.40) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    valid = np.where(precision[:-1] >= target_precision)[0]
    if len(valid) == 0:
        return 0.0, 1.0
    best = valid[np.argmax(recall[valid])]
    return float(recall[best]), float(thresholds[best])


def evaluate(name: str, estimator, x_train, x_test, y_train, y_test) -> Result:
    estimator.fit(x_train, y_train)
    scores = estimator.predict_proba(x_test)[:, 1]
    recall, threshold = recall_at_precision(y_test.to_numpy(), scores)
    return Result(
        strategy=name,
        roc_auc=float(roc_auc_score(y_test, scores)),
        average_precision=float(average_precision_score(y_test, scores)),
        brier_score=float(brier_score_loss(y_test, scores)),
        recall_at_target_precision=recall,
        threshold_at_target_precision=threshold,
    )


def benchmark(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    weighted = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]
    )
    oversampled = ImbPipeline(
        [
            ("scale", StandardScaler()),
            ("resample", RandomOverSampler(random_state=42)),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    smote = ImbPipeline(
        [
            ("scale", StandardScaler()),
            ("resample", SMOTE(random_state=42, k_neighbors=5)),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    undersampled = ImbPipeline(
        [
            ("scale", StandardScaler()),
            ("resample", RandomUnderSampler(random_state=42)),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    candidates = [
        ("class_weight", weighted),
        ("random_oversampling", oversampled),
        ("smote", smote),
        ("random_undersampling", undersampled),
    ]
    results = [
        evaluate(name, estimator, x_train, x_test, y_train, y_test)
        for name, estimator in candidates
    ]
    return pd.DataFrame([result.__dict__ for result in results]).sort_values(
        ["average_precision", "brier_score"],
        ascending=[False, True],
    )


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    rows = 12000
    frame = pd.DataFrame(
        {
            "utilization": rng.beta(2, 5, rows),
            "late_payments": rng.poisson(0.4, rows),
            "income": rng.lognormal(10.9, 0.45, rows),
            "debt_ratio": rng.beta(2, 7, rows),
        }
    )
    logit = -5.0 + 4.0 * frame["utilization"] + 0.55 * frame["late_payments"] + 2.2 * frame["debt_ratio"]
    probability = 1 / (1 + np.exp(-logit))
    target = pd.Series(rng.binomial(1, probability), name="default")
    print("event rate:", float(target.mean()))
    print(benchmark(frame, target).to_string(index=False))
