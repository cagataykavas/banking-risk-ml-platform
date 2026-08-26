from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    roc_auc: float
    average_precision: float
    brier_score: float
    score: float
    params: dict[str, object]


def temporal_cv_score(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    splits: int = 4,
) -> tuple[float, float, float]:
    """Evaluate a probability model with order-preserving validation folds."""
    aucs: list[float] = []
    aps: list[float] = []
    briers: list[float] = []

    for train_idx, valid_idx in TimeSeriesSplit(n_splits=splits).split(x):
        model = clone(estimator)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(x.iloc[valid_idx])[:, 1]
        target = y.iloc[valid_idx]

        if target.nunique() < 2:
            continue

        aucs.append(float(roc_auc_score(target, probability)))
        aps.append(float(average_precision_score(target, probability)))
        briers.append(float(brier_score_loss(target, probability)))

    if not aucs:
        raise RuntimeError("temporal validation produced no usable folds")
    return float(np.mean(aucs)), float(np.mean(aps)), float(np.mean(briers))


def business_aware_score(roc_auc: float, average_precision: float, brier_score: float) -> float:
    """A transparent composite used only to rank portfolio benchmark candidates.

    Higher discrimination and ranking quality help; poor calibration is penalized.
    Production promotion should still use explicit business and governance gates.
    """
    return 0.45 * roc_auc + 0.40 * average_precision - 0.15 * brier_score


def make_xgb(params: dict[str, object] | None = None) -> XGBClassifier:
    values = {
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 4,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    values.update(params or {})
    return XGBClassifier(**values)


def make_lgbm(params: dict[str, object] | None = None) -> LGBMClassifier:
    values = {
        "n_estimators": 350,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_samples": 30,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    values.update(params or {})
    return LGBMClassifier(**values)


def tune(
    factory: Callable[[dict[str, object]], BaseEstimator],
    search_space: Callable[[optuna.Trial], dict[str, object]],
    x: pd.DataFrame,
    y: pd.Series,
    *,
    trials: int = 30,
) -> tuple[dict[str, object], float]:
    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        auc, ap, brier = temporal_cv_score(factory(params), x, y)
        trial.set_user_attr("roc_auc", auc)
        trial.set_user_attr("average_precision", ap)
        trial.set_user_attr("brier_score", brier)
        return business_aware_score(auc, ap, brier)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    return dict(study.best_params), float(study.best_value)


def xgb_space(trial: optuna.Trial) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 180, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 5.0, log=True),
    }


def lgbm_space(trial: optuna.Trial) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 180, 600),
        "num_leaves": trial.suggest_int("num_leaves", 15, 96),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 5.0, log=True),
    }


def evaluate_candidate(name: str, estimator: BaseEstimator, x: pd.DataFrame, y: pd.Series) -> BenchmarkResult:
    auc, ap, brier = temporal_cv_score(estimator, x, y)
    params = estimator.get_params(deep=False)
    compact = {key: value for key, value in params.items() if isinstance(value, (str, int, float, bool, type(None)))}
    return BenchmarkResult(
        model_name=name,
        roc_auc=auc,
        average_precision=ap,
        brier_score=brier,
        score=business_aware_score(auc, ap, brier),
        params=compact,
    )


def benchmark(x: pd.DataFrame, y: pd.Series, *, tune_models: bool = False, trials: int = 30) -> pd.DataFrame:
    candidates: list[tuple[str, BaseEstimator]] = [
        ("xgboost", make_xgb()),
        ("lightgbm", make_lgbm()),
    ]

    if tune_models:
        xgb_params, _ = tune(make_xgb, xgb_space, x, y, trials=trials)
        lgbm_params, _ = tune(make_lgbm, lgbm_space, x, y, trials=trials)
        candidates.extend(
            [
                ("xgboost_tuned", make_xgb(xgb_params)),
                ("lightgbm_tuned", make_lgbm(lgbm_params)),
            ]
        )

    results = [evaluate_candidate(name, estimator, x, y) for name, estimator in candidates]
    frame = pd.DataFrame([result.__dict__ for result in results])
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rows = 8000
    x = pd.DataFrame(
        {
            "income": rng.lognormal(10.8, 0.5, rows),
            "utilization": rng.beta(2.2, 4.8, rows),
            "late_payments": rng.poisson(0.55, rows),
            "account_age_months": rng.integers(1, 180, rows),
            "debt_ratio": rng.beta(2.0, 5.5, rows),
        }
    )
    logit = -3.7 + 3.4 * x["utilization"] + 0.48 * x["late_payments"] + 2.1 * x["debt_ratio"]
    probability = 1 / (1 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, probability), name="default")
    print(benchmark(x, y, tune_models=False).to_string(index=False))
