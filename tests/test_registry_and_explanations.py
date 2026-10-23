import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pytest

from src.explainability import explain_prediction
from src.model_registry import ModelRegistry, ModelRegistryError
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


def test_registration_is_immutable_and_preserves_original_evidence(tmp_path: Path):
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    first.write_bytes(b"first-model")
    second.write_bytes(b"second-model")
    registry = ModelRegistry(tmp_path / "registry.db")
    original = registry.register("risk-2026-10-23-a", first, {"roc_auc": 0.75})

    with pytest.raises(ModelRegistryError, match="MODEL_VERSION_ALREADY_EXISTS"):
        registry.register("risk-2026-10-23-a", second, {"roc_auc": 0.99})

    stored = registry.get(original.version)
    assert stored.sha256 == original.sha256
    assert stored.metrics == {"roc_auc": 0.75}


def test_tampered_artifact_cannot_be_promoted(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"registered-model")
    registry = ModelRegistry(tmp_path / "registry.db")
    registry.register("risk-2026-10-23-a", artifact, {"roc_auc": 0.75})
    artifact.write_bytes(b"tampered-model")

    with pytest.raises(ModelRegistryError, match="MODEL_ARTIFACT_INTEGRITY_FAILED"):
        registry.promote("risk-2026-10-23-a", "production", "quality gate passed")

    assert registry.get("risk-2026-10-23-a").stage == "candidate"
    assert registry.promotion_history() == []


def test_production_replacement_is_atomic_when_audit_write_fails(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.db")
    for version in ("risk-a", "risk-b"):
        artifact = tmp_path / f"{version}.joblib"
        artifact.write_bytes(version.encode())
        registry.register(version, artifact, {"roc_auc": 0.75})
    registry.promote("risk-a", "production", "initial release")

    with sqlite3.connect(registry.db_path) as con:
        con.executescript(
            """
            CREATE TRIGGER reject_risk_b_promotion
            BEFORE INSERT ON promotion_log
            WHEN NEW.version = 'risk-b'
            BEGIN
                SELECT RAISE(ABORT, 'simulated audit failure');
            END;
            """
        )

    with pytest.raises(ModelRegistryError, match="PROMOTION_TRANSACTION_FAILED"):
        registry.promote("risk-b", "production", "replacement release")

    assert registry.get("risk-a").stage == "production"
    assert registry.get("risk-b").stage == "candidate"
    assert [item["version"] for item in registry.promotion_history()] == ["risk-a"]


def test_concurrent_promotions_leave_exactly_one_production_model(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.db")
    versions = [f"risk-{index}" for index in range(8)]
    for version in versions:
        artifact = tmp_path / f"{version}.joblib"
        artifact.write_bytes(version.encode())
        registry.register(version, artifact, {"roc_auc": 0.75})

    with ThreadPoolExecutor(max_workers=len(versions)) as pool:
        promoted = list(
            pool.map(
                lambda version: registry.promote(
                    version, "production", "concurrent release test"
                ),
                versions,
            )
        )

    assert len(promoted) == len(versions)
    stored = registry.list_versions()
    assert sum(item["stage"] == "production" for item in stored) == 1
    assert sum(item["stage"] == "archived" for item in stored) == len(versions) - 1
    assert len(registry.promotion_history()) == len(versions)


@pytest.mark.parametrize(
    ("version", "metrics", "stage", "code"),
    [
        ("latest", {"roc_auc": 0.75}, "candidate", "INVALID_MODEL_VERSION"),
        ("risk-v1", {}, "candidate", "INVALID_MODEL_METRICS"),
        ("risk-v1", {"roc_auc": float("nan")}, "candidate", "INVALID_MODEL_METRICS"),
        ("risk-v1", {"roc_auc": 0.75}, "unknown", "INVALID_MODEL_STAGE"),
        ("risk-v1", {"roc_auc": 0.75}, "production", "INVALID_INITIAL_STAGE"),
    ],
)
def test_registration_metadata_fails_closed(
    tmp_path: Path,
    version: str,
    metrics: dict[str, float],
    stage: str,
    code: str,
):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry.db")

    with pytest.raises(ModelRegistryError, match=code):
        registry.register(version, artifact, metrics, stage=stage)


def test_missing_artifact_and_invalid_transition_fail_closed(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.db")
    with pytest.raises(ModelRegistryError, match="MODEL_ARTIFACT_NOT_FOUND"):
        registry.register("risk-v1", tmp_path / "missing.joblib", {"roc_auc": 0.75})

    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    registry.register("risk-v1", artifact, {"roc_auc": 0.75})
    registry.promote("risk-v1", "production", "quality gate passed")
    with pytest.raises(ModelRegistryError, match="INVALID_STAGE_TRANSITION"):
        registry.promote("risk-v1", "staging", "attempted rollback")


def test_explanation_returns_ranked_local_sensitivities():
    df = make_dataset(1000, seed=11)
    model = build_pipeline().fit(df[NUMERIC + CATEGORICAL], df["defaulted"])
    row = df.iloc[0].to_dict()
    explanation = explain_prediction(model, row, top_k=4)

    assert 0 <= explanation["probability"] <= 1
    assert len(explanation["top_contributors"]) == 4
    magnitudes = [
        abs(item["probability_delta"]) for item in explanation["top_contributors"]
    ]
    assert magnitudes == sorted(magnitudes, reverse=True)
