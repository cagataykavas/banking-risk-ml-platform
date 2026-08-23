from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability import explain_prediction
from src.model_registry import ModelRegistry
from src.risk_pipeline import CATEGORICAL, NUMERIC

REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", "artifacts/registry.db"))
REGISTRY = ModelRegistry(REGISTRY_PATH)
_MODEL_CACHE: dict[str, object] = {}

app = FastAPI(
    title="Banking Risk ML Platform",
    version="1.0.0",
    description="Synthetic credit-risk serving platform with model registry, explanations and human-review routing.",
)


class CreditApplication(BaseModel):
    application_id: str = Field(min_length=3)
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0)
    debt: float = Field(ge=0)
    utilization: float = Field(ge=0, le=1)
    late_payments: int = Field(ge=0)
    account_age_months: int = Field(ge=0)
    employment: Literal["salaried", "self_employed", "student", "unemployed"]
    housing: Literal["rent", "mortgage", "owned"]


class HumanDecision(BaseModel):
    application_id: str
    reviewer_id: str
    decision: Literal["approve", "decline", "request_information"]
    reason: str = Field(min_length=3)


REVIEW_LOG: list[dict] = []


def _load_production_model():
    try:
        record = REGISTRY.latest("production")
    except KeyError as exc:
        raise HTTPException(503, "No production model is registered") from exc
    if not REGISTRY.verify_artifact(record.version):
        raise HTTPException(503, "Production model artifact checksum verification failed")
    if record.version not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[record.version] = joblib.load(record.artifact_path)
    return record, _MODEL_CACHE[record.version]


def route_decision(probability: float) -> tuple[str, str]:
    if probability < 0.18:
        return "auto_approve", "risk probability below automatic approval boundary"
    if probability > 0.62:
        return "human_review", "high-risk applications require explicit human review"
    return "human_review", "uncertain probability region is deliberately deferred"


@app.get("/health")
def health() -> dict:
    try:
        record = REGISTRY.latest("production")
        verified = REGISTRY.verify_artifact(record.version)
        return {"status": "ok" if verified else "degraded", "model_version": record.version, "artifact_verified": verified}
    except KeyError:
        return {"status": "degraded", "reason": "no production model"}


@app.get("/models")
def list_models() -> list[dict]:
    return REGISTRY.list_versions()


@app.post("/score")
def score(application: CreditApplication) -> dict:
    record, model = _load_production_model()
    payload = application.model_dump()
    feature_payload = {name: payload[name] for name in NUMERIC + CATEGORICAL}
    frame = pd.DataFrame([feature_payload])
    probability = float(model.predict_proba(frame)[0, 1])
    route, reason = route_decision(probability)
    explanation = explain_prediction(model, feature_payload)
    return {
        "application_id": application.application_id,
        "model_version": record.version,
        "default_probability": probability,
        "route": route,
        "route_reason": reason,
        "explanation": explanation,
    }


@app.post("/human-decisions")
def record_human_decision(decision: HumanDecision) -> dict:
    event = decision.model_dump()
    REVIEW_LOG.append(event)
    return {"status": "recorded", "sequence": len(REVIEW_LOG), **event}


@app.get("/human-decisions")
def list_human_decisions() -> list[dict]:
    return REVIEW_LOG
