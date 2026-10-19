from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from app import api
from src.explainability import _counterfactual_value, explain_prediction
from src.risk_pipeline import CATEGORICAL, NUMERIC, build_pipeline, make_dataset

PAYLOAD = {
    "age": 42,
    "income": 80000.0,
    "debt": 14000.0,
    "utilization": 0.65,
    "late_payments": 2,
    "account_age_months": 100,
    "employment": "self_employed",
    "housing": "owned",
}


class CountingModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append(frame.copy(deep=True))
        probability = (
            0.05
            + frame["utilization"].to_numpy() * 0.4
            + frame["late_payments"].to_numpy() * 0.05
        )
        return np.column_stack([1 - probability, probability])


def test_batch_has_baseline_and_exactly_one_changed_feature_per_row():
    model = CountingModel()
    before = dict(PAYLOAD)
    result = explain_prediction(model, PAYLOAD, top_k=8)

    assert len(model.calls) == 1
    batch = model.calls[0]
    assert len(batch) == 9
    assert list(batch.columns) == NUMERIC + CATEGORICAL
    assert batch.iloc[0].to_dict() == PAYLOAD
    for index, feature in enumerate(NUMERIC + CATEGORICAL, start=1):
        expected = {
            **PAYLOAD,
            feature: _counterfactual_value(feature, PAYLOAD[feature]),
        }
        assert batch.iloc[index].to_dict() == expected
    assert PAYLOAD == before
    assert result["probability"] == pytest.approx(0.41)


def test_batch_matches_independent_row_reference_and_stable_ranking():
    model = CountingModel()
    baseline = float(model.predict_proba(pd.DataFrame([PAYLOAD]))[0, 1])
    expected = {}
    for feature in NUMERIC + CATEGORICAL:
        row = {**PAYLOAD, feature: _counterfactual_value(feature, PAYLOAD[feature])}
        expected[feature] = baseline - float(
            model.predict_proba(pd.DataFrame([row]))[0, 1]
        )

    result = explain_prediction(CountingModel(), PAYLOAD, top_k=8)
    rows = result["top_contributors"]
    assert {
        item["feature"]: item["probability_delta"] for item in rows
    } == pytest.approx(expected)
    assert [item["feature"] for item in rows] == sorted(
        expected, key=lambda k: -abs(expected[k])
    )
    assert all(
        item["direction"] == "no_change"
        for item in rows
        if item["probability_delta"] == 0
    )


def test_default_class_column_is_resolved_by_label_not_position():
    class ReversedModel(CountingModel):
        classes_ = np.array([1, 0])

        def predict_proba(self, frame):
            return super().predict_proba(frame)[:, ::-1]

    assert explain_prediction(ReversedModel(), PAYLOAD) == explain_prediction(
        CountingModel(), PAYLOAD
    )


def test_real_sklearn_pipeline_batch_matches_sequential_predictions():
    values = make_dataset(300, seed=37)
    model = build_pipeline().fit(values[NUMERIC + CATEGORICAL], values["defaulted"])
    payload = values.iloc[0].to_dict()
    row = {name: payload[name] for name in NUMERIC + CATEGORICAL}
    baseline = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    expected = {}
    for feature in NUMERIC + CATEGORICAL:
        changed = {**row, feature: _counterfactual_value(feature, payload[feature])}
        expected[feature] = baseline - float(
            model.predict_proba(pd.DataFrame([changed]))[0, 1]
        )

    result = explain_prediction(model, payload, top_k=8)
    assert result["probability"] == pytest.approx(baseline, abs=1e-12)
    assert {
        item["feature"]: item["probability_delta"]
        for item in result["top_contributors"]
    } == pytest.approx(expected, abs=1e-12)


def test_missing_class_metadata_fails_closed():
    with pytest.raises(ValueError, match="binary classes"):
        explain_prediction(object(), PAYLOAD)


def test_bad_perturbation_row_is_rejected_even_with_valid_baseline():
    class InvalidPerturbationModel(CountingModel):
        def predict_proba(self, frame):
            values = super().predict_proba(frame)
            values[-1] = np.nan
            return values

    with pytest.raises(ValueError, match="invalid probabilities"):
        explain_prediction(InvalidPerturbationModel(), PAYLOAD)


@pytest.mark.parametrize("top_k", [0, -1, 9, 1.5, True])
def test_invalid_top_k_is_rejected_before_model_call(top_k):
    model = CountingModel()
    with pytest.raises((TypeError, ValueError), match="top_k"):
        explain_prediction(model, PAYLOAD, top_k=top_k)
    assert model.calls == []


@pytest.mark.parametrize(
    "values",
    [
        np.full((9, 2), np.nan),
        np.full((9, 2), np.inf),
        np.tile([-0.1, 1.1], (9, 1)),
        np.full((9, 2), 0.7),
        np.full((8, 2), 0.5),
        np.full((9,), 0.5),
    ],
)
def test_malformed_probabilities_fail_closed(values):
    class InvalidModel(CountingModel):
        def predict_proba(self, frame):
            return values

    with pytest.raises(ValueError):
        explain_prediction(InvalidModel(), PAYLOAD)


def test_non_binary_classes_are_rejected_before_model_call():
    model = CountingModel()
    model.classes_ = np.array([0, 2])
    with pytest.raises(ValueError, match="binary classes"):
        explain_prediction(model, PAYLOAD)
    assert model.calls == []


@dataclass
class Record:
    version: str = "test-v1"


def test_score_reuses_batched_baseline_and_only_calls_model_once(monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(api, "_load_production_model", lambda: (Record(), model))
    result = api.score(api.CreditApplication(application_id="test-123", **PAYLOAD))

    assert len(model.calls) == 1
    assert result["default_probability"] == result["explanation"]["probability"]
    assert result["route"] == "human_review"


def test_score_rejects_invalid_probabilities_with_service_error(monkeypatch):
    class InvalidModel(CountingModel):
        def predict_proba(self, frame):
            return np.full((len(frame), 2), np.nan)

    monkeypatch.setattr(
        api, "_load_production_model", lambda: (Record(), InvalidModel())
    )
    with pytest.raises(HTTPException) as error:
        api.score(api.CreditApplication(application_id="test-123", **PAYLOAD))
    assert error.value.status_code == 503
    assert error.value.detail == "Production model probability contract failed"
