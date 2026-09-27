from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.explanation_direction_audit import (
    DirectionArtifactError,
    DirectionAuditPolicy,
    FeatureDirectionContract,
    audit_explanation_directions,
    load_json,
    policy_from_dict,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)


def policy(**overrides: object) -> DirectionAuditPolicy:
    values: dict[str, object] = {
        "required_explanation_method": "one_feature_counterfactual_sensitivity",
        "features": (
            FeatureDirectionContract("late_payments", "nondecreasing_risk", 1.0),
            FeatureDirectionContract("account_age_months", "nonincreasing_risk", 12.0),
        ),
        "min_cases_per_feature": 2,
        "max_cases": 100,
        "minimum_material_effect": 0.001,
        "max_flat_effect_fraction": 0.5,
    }
    values.update(overrides)
    return DirectionAuditPolicy(**values)  # type: ignore[arg-type]


def case(
    case_id: str,
    feature: str,
    baseline: float,
    perturbed: float,
    baseline_probability: float,
    perturbed_probability: float,
    attribution_delta: float | None = None,
) -> dict:
    delta = perturbed_probability - baseline_probability
    return {
        "case_id": case_id,
        "feature": feature,
        "baseline_value": baseline,
        "perturbed_value": perturbed,
        "baseline_probability": baseline_probability,
        "perturbed_probability": perturbed_probability,
        "attribution_delta": delta if attribution_delta is None else attribution_delta,
    }


def artifact() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-09-27T11:59:00Z",
        "model_sha256": "a" * 64,
        "cohort_sha256": "b" * 64,
        "explanation_method": "one_feature_counterfactual_sensitivity",
        "cases": [
            case("late-1", "late_payments", 0, 1, 0.10, 0.13),
            case("late-2", "late_payments", 2, 4, 0.20, 0.28),
            case("age-1", "account_age_months", 72, 60, 0.20, 0.24),
            case("age-2", "account_age_months", 120, 96, 0.15, 0.19),
        ],
    }


def policy_dict() -> dict:
    return policy().as_dict()


def test_accepts_model_and_attribution_direction_contracts() -> None:
    report = audit_explanation_directions(artifact(), policy(), now=NOW)
    assert report.accepted is True
    assert report.reason_codes == ()
    assert report.case_count == 4
    assert len(report.evidence_sha256) == 64


def test_report_is_private_and_deterministic() -> None:
    first = audit_explanation_directions(artifact(), policy(), now=NOW).as_dict()
    second = audit_explanation_directions(artifact(), policy(), now=NOW).as_dict()
    assert first == second
    encoded = json.dumps(first)
    assert "late-1" not in encoded
    assert "baseline_value" not in encoded


def test_artifact_digest_is_independent_of_case_order() -> None:
    left = artifact()
    right = artifact()
    right["cases"].reverse()
    assert (
        audit_explanation_directions(left, policy(), now=NOW).artifact_sha256
        == audit_explanation_directions(right, policy(), now=NOW).artifact_sha256
    )


def test_rejects_model_direction_breach() -> None:
    payload = artifact()
    payload["cases"][0]["perturbed_probability"] = 0.08
    payload["cases"][0]["attribution_delta"] = -0.02
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert "MODEL_DIRECTION_BREACH_RATE_EXCEEDED" in report.reason_codes


def test_rejects_attribution_direction_breach() -> None:
    payload = artifact()
    payload["cases"][0]["attribution_delta"] = -0.03
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert "ATTRIBUTION_DIRECTION_BREACH_RATE_EXCEEDED" in report.reason_codes


def test_rejects_attribution_probability_delta_mismatch() -> None:
    payload = artifact()
    payload["cases"][0]["attribution_delta"] = 0.02
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert "ATTRIBUTION_MISMATCH_RATE_EXCEEDED" in report.reason_codes


def test_rejects_excessively_flat_effects() -> None:
    payload = artifact()
    for row in payload["cases"][:2]:
        row["perturbed_probability"] = row["baseline_probability"]
        row["attribution_delta"] = 0.0
    report = audit_explanation_directions(payload, policy(max_flat_effect_fraction=0.25), now=NOW)
    assert "FLAT_EFFECT_RATE_EXCEEDED" in report.reason_codes


def test_rejects_wrong_perturbation_direction_or_step() -> None:
    payload = artifact()
    payload["cases"][0]["perturbed_value"] = -1
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert "PERTURBATION_DIRECTION_CONTRACT_BREACH" in report.reason_codes


def test_rejects_missing_feature_sample_coverage() -> None:
    payload = artifact()
    payload["cases"] = payload["cases"][:2]
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert "FEATURE_SAMPLE_INSUFFICIENT" in report.reason_codes


@pytest.mark.parametrize(
    ("generated_at", "code"),
    [
        ("2026-09-25T11:59:00Z", "ARTIFACT_STALE"),
        ("2026-09-27T12:06:00Z", "ARTIFACT_FROM_FUTURE"),
    ],
)
def test_rejects_stale_or_future_evidence(generated_at: str, code: str) -> None:
    payload = artifact()
    payload["generated_at"] = generated_at
    assert code in audit_explanation_directions(payload, policy(), now=NOW).reason_codes


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(extra=True),
        lambda value: value["cases"][0].update(extra=True),
        lambda value: value["cases"][1].update(case_id="late-1"),
        lambda value: value["cases"][0].update(feature="undeclared"),
        lambda value: value["cases"][0].update(baseline_probability=1.1),
        lambda value: value.update(generated_at="2026-09-27T11:59:00"),
        lambda value: value.update(explanation_method="other_method"),
    ],
)
def test_rejects_malformed_artifact(mutator) -> None:
    payload = artifact()
    mutator(payload)
    with pytest.raises(DirectionArtifactError):
        audit_explanation_directions(payload, policy(), now=NOW)


def test_policy_round_trip_and_digest_order_are_stable() -> None:
    first = policy_from_dict(policy_dict())
    raw = policy_dict()
    raw["features"].reverse()
    second = policy_from_dict(raw)
    assert first.as_dict() == second.as_dict()


def test_policy_preserves_specific_schema_failure_code() -> None:
    payload = policy_dict()
    payload["features"][0]["unexpected"] = True
    with pytest.raises(DirectionArtifactError, match="FEATURE_CONTRACT_SCHEMA_MISMATCH"):
        policy_from_dict(payload)


def test_policy_rejects_duplicate_features_and_nonfinite_values() -> None:
    duplicate = FeatureDirectionContract("late_payments", "nondecreasing_risk", 1.0)
    with pytest.raises(ValueError):
        DirectionAuditPolicy("method", (duplicate, duplicate))
    with pytest.raises(ValueError):
        policy(direction_tolerance=float("nan"))
    with pytest.raises(ValueError):
        policy(max_cases=True)
    with pytest.raises(ValueError):
        policy(max_features="many")
    with pytest.raises(ValueError):
        policy(features=("late_payments",))
    with pytest.raises(ValueError):
        FeatureDirectionContract("late_payments", [], 1.0)  # type: ignore[arg-type]


def test_strict_loader_rejects_duplicate_keys_nonfinite_and_budget(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(DirectionArtifactError, match="DUPLICATE_JSON_KEY"):
        load_json(duplicate, max_bytes=1_024)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(DirectionArtifactError, match="NON_FINITE_JSON_NUMBER"):
        load_json(nonfinite, max_bytes=1_024)
    oversized = tmp_path / "oversized.json"
    oversized.write_text("x" * 1_025, encoding="utf-8")
    with pytest.raises(DirectionArtifactError, match="JSON_BYTE_BUDGET_EXCEEDED"):
        load_json(oversized, max_bytes=1_024)


def run_cli(tmp_path: Path, payload: dict | str) -> subprocess.CompletedProcess[str]:
    artifact_path = tmp_path / "artifact.json"
    policy_path = tmp_path / "policy.json"
    if isinstance(payload, str):
        artifact_path.write_text(payload, encoding="utf-8")
    else:
        artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    policy_path.write_text(json.dumps(policy_dict()), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.explanation_direction_audit",
            str(artifact_path),
            "--policy",
            str(policy_path),
            "--now",
            "2026-09-27T12:00:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_distinguishes_accept_reject_and_malformed(tmp_path: Path) -> None:
    accepted = run_cli(tmp_path, artifact())
    rejected_payload = deepcopy(artifact())
    rejected_payload["cases"][0]["attribution_delta"] = -0.03
    rejected = run_cli(tmp_path, rejected_payload)
    malformed = run_cli(tmp_path, "{broken")
    assert accepted.returncode == 0
    assert rejected.returncode == 2
    assert malformed.returncode == 3
    assert json.loads(accepted.stdout)["accepted"] is True
    assert json.loads(rejected.stdout)["accepted"] is False
    assert json.loads(malformed.stdout)["status"] == "malformed"


def test_reason_codes_are_stably_sorted() -> None:
    payload = artifact()
    payload["cases"][0]["perturbed_probability"] = 0.08
    payload["cases"][0]["attribution_delta"] = -0.03
    report = audit_explanation_directions(payload, policy(), now=NOW)
    assert report.reason_codes == tuple(sorted(report.reason_codes))
