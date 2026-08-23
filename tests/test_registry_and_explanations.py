from pathlib import Path

import joblib

from src.explainability import explain_prediction
from src.model_registry import ModelRegistry
from src.risk_pipeline import CATEGORICAL, NUMERIC, build_pipeline, make_dataset


def test_registry_promotes_and_verifies(tmp_path: Path):
    df = make_dataset(500, seed=7)
    model = build_pipeline().fit(df[NUMERIC + CATEGORICAL], df["defaulted"])
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)

    registry = ModelRegistry(tmp_path / "registry.db")
    registry.register("v1", artifact, {"roc_auc": 0.75})
    promoted = registry.promote("v1", "production", "test promotion")

    assert promoted.stage == "production"
    assert registry.verify_artifact("v1")
    assert registry.latest("production").version == "v1"


def test_explanation_returns_ranked_local_sensitivities():
    df = make_dataset(1000, seed=11)
    model = build_pipeline().fit(df[NUMERIC + CATEGORICAL], df["defaulted"])
    row = df.iloc[0].to_dict()
    explanation = explain_prediction(model, row, top_k=4)

    assert 0 <= explanation["probability"] <= 1
    assert len(explanation["top_contributors"]) == 4
    magnitudes = [abs(item["probability_delta"]) for item in explanation["top_contributors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
