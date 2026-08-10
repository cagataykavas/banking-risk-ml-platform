"""Reproducible credit-risk training pipeline using synthetic data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "income", "debt", "utilization", "late_payments", "account_age_months"]
CATEGORICAL = ["employment", "housing"]
TARGET = "defaulted"


def make_dataset(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, n)
    income = rng.lognormal(10.7, 0.55, n)
    debt = rng.gamma(2.2, 8500, n)
    utilization = np.clip(rng.beta(2.2, 2.8, n), 0, 1)
    late = rng.poisson(0.8, n)
    account_age = rng.integers(1, 360, n)
    employment = rng.choice(["salaried", "self_employed", "student", "unemployed"], n, p=[.62, .2, .08, .1])
    housing = rng.choice(["rent", "mortgage", "owned"], n, p=[.42, .38, .2])
    logit = -3.0 + 2.8 * utilization + .35 * late + 1.1 * (debt / (income + 1)) + .7 * (employment == "unemployed") - .002 * account_age
    probability = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, np.clip(probability, .01, .95))
    return pd.DataFrame({"age": age, "income": income, "debt": debt, "utilization": utilization, "late_payments": late, "account_age_months": account_age, "employment": employment, "housing": housing, TARGET: defaulted})


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    features = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=.06, max_leaf_nodes=24, l2_regularization=.5, random_state=42)
    return Pipeline([("features", features), ("model", model)])


def temporal_split(df: pd.DataFrame, test_fraction: float = .2):
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def train(output: Path, rows: int = 12000) -> dict[str, float]:
    df = make_dataset(rows)
    train_df, test_df = temporal_split(df)
    pipe = build_pipeline()
    pipe.fit(train_df[NUMERIC + CATEGORICAL], train_df[TARGET])
    probability = pipe.predict_proba(test_df[NUMERIC + CATEGORICAL])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test_df[TARGET], probability)),
        "average_precision": float(average_precision_score(test_df[TARGET], probability)),
        "brier_score": float(brier_score_loss(test_df[TARGET], probability)),
        "test_default_rate": float(test_df[TARGET].mean()),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, output / "credit_risk.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.rows), indent=2))
