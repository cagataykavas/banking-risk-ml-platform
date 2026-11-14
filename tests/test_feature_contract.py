from __future__ import annotations

import json

import numpy as np

from src.feature_contract import (
    contract_sha256,
    credit_application_contract,
    validate_frame,
)
from src.risk_pipeline import CATEGORICAL, NUMERIC, make_dataset, train


def _features(rows: int = 200):
    return make_dataset(rows)[NUMERIC + CATEGORICAL]


def test_clean_credit_batch_matches_versioned_contract() -> None:
    contract = credit_application_contract()
    report = validate_frame(_features(), contract)

    assert report.valid is True
    assert report.issues == ()
    assert report.contract_sha256 == contract_sha256(contract)
    assert report.feature_stats["utilization"]["out_of_range_count"] == 0
    assert report.feature_stats["employment"]["unknown_count"] == 0


def test_schema_and_dtype_breaks_fail_closed_without_exposing_values() -> None:
    frame = _features().drop(columns=["income"])
    frame["age"] = frame["age"].astype(str)
    frame["undeclared_feature"] = "not persisted in findings"

    report = validate_frame(frame, credit_application_contract())
    issue_payload = [issue.__dict__ for issue in report.issues]
    codes = {issue.code for issue in report.issues}

    assert report.valid is False
    assert codes >= {"missing_feature", "unexpected_feature", "invalid_numeric_dtype"}
    assert "not persisted in findings" not in json.dumps(issue_payload)


def test_nonfinite_bounds_missingness_and_unknown_categories_are_distinct() -> None:
    frame = _features()
    frame.loc[0, "income"] = np.inf
    frame.loc[1, "utilization"] = 1.5
    frame.loc[2, "debt"] = np.nan
    frame.loc[3:8, "housing"] = "company_dormitory"

    report = validate_frame(frame, credit_application_contract())
    codes = {issue.code for issue in report.issues}

    assert codes >= {
        "nonfinite_numeric",
        "numeric_out_of_range",
        "missing_rate_exceeded",
        "unknown_category_rate_exceeded",
    }


def test_training_persists_contract_and_validation_receipt(tmp_path) -> None:
    metrics = train(tmp_path, rows=600)
    contract = json.loads((tmp_path / "feature_contract.json").read_text())
    receipt = json.loads((tmp_path / "feature_contract_validation.json").read_text())

    assert metrics["roc_auc"] > 0.5
    assert contract["version"] == "credit-application-v1"
    assert contract["contract_sha256"] == receipt["contract_sha256"]
    assert receipt["valid"] is True
    assert receipt["row_count"] == 600
