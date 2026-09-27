from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_KEYS = {
    "schema_version",
    "generated_at",
    "model_sha256",
    "cohort_sha256",
    "explanation_method",
    "cases",
}
CASE_KEYS = {
    "case_id",
    "feature",
    "baseline_value",
    "perturbed_value",
    "baseline_probability",
    "perturbed_probability",
    "attribution_delta",
}
POLICY_KEYS = {
    "schema_version",
    "required_explanation_method",
    "min_cases_per_feature",
    "max_cases",
    "max_features",
    "max_artifact_bytes",
    "max_age_seconds",
    "max_future_skew_seconds",
    "direction_tolerance",
    "max_attribution_error",
    "minimum_material_effect",
    "max_model_direction_breach_fraction",
    "max_attribution_direction_breach_fraction",
    "max_attribution_mismatch_fraction",
    "max_flat_effect_fraction",
    "features",
}
FEATURE_KEYS = {"feature", "direction", "minimum_step"}


class DirectionArtifactError(ValueError):
    """Malformed evidence error with a stable, non-sensitive reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class FeatureDirectionContract:
    feature: str
    direction: str
    minimum_step: float

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str) or not SAFE_NAME.fullmatch(self.feature):
            raise ValueError("feature must be a safe identifier")
        if not isinstance(self.direction, str) or self.direction not in {
            "nondecreasing_risk",
            "nonincreasing_risk",
        }:
            raise ValueError("direction is invalid")
        if (
            isinstance(self.minimum_step, bool)
            or not isinstance(self.minimum_step, (int, float))
            or not math.isfinite(self.minimum_step)
            or self.minimum_step <= 0
        ):
            raise ValueError("minimum_step must be finite and positive")


@dataclass(frozen=True)
class DirectionAuditPolicy:
    required_explanation_method: str
    features: tuple[FeatureDirectionContract, ...]
    min_cases_per_feature: int = 20
    max_cases: int = 20_000
    max_features: int = 128
    max_artifact_bytes: int = 4_000_000
    max_age_seconds: float = 86_400.0
    max_future_skew_seconds: float = 300.0
    direction_tolerance: float = 1e-9
    max_attribution_error: float = 1e-6
    minimum_material_effect: float = 1e-4
    max_model_direction_breach_fraction: float = 0.05
    max_attribution_direction_breach_fraction: float = 0.05
    max_attribution_mismatch_fraction: float = 0.0
    max_flat_effect_fraction: float = 0.80

    def __post_init__(self) -> None:
        if not isinstance(self.required_explanation_method, str) or not SAFE_NAME.fullmatch(
            self.required_explanation_method
        ):
            raise ValueError("required_explanation_method must be a safe identifier")
        if not isinstance(self.features, tuple) or not self.features:
            raise ValueError("at least one feature contract is required")
        for name, value, lower, upper in (
            ("min_cases_per_feature", self.min_cases_per_feature, 1, 100_000),
            ("max_cases", self.max_cases, 1, 1_000_000),
            ("max_features", self.max_features, 1, 10_000),
            ("max_artifact_bytes", self.max_artifact_bytes, 1_024, 64_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        if not all(isinstance(item, FeatureDirectionContract) for item in self.features):
            raise ValueError("features must contain direction contracts")
        if len(self.features) > self.max_features:
            raise ValueError("feature contract budget exceeded")
        names = [item.feature for item in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature contracts must be unique")
        if self.min_cases_per_feature * len(self.features) > self.max_cases:
            raise ValueError("minimum feature coverage exceeds max_cases")
        for name, value, lower, upper in (
            ("max_age_seconds", self.max_age_seconds, 1.0, 31_536_000.0),
            ("max_future_skew_seconds", self.max_future_skew_seconds, 0.0, 86_400.0),
            ("direction_tolerance", self.direction_tolerance, 0.0, 1.0),
            ("max_attribution_error", self.max_attribution_error, 0.0, 1.0),
            ("minimum_material_effect", self.minimum_material_effect, 0.0, 1.0),
            (
                "max_model_direction_breach_fraction",
                self.max_model_direction_breach_fraction,
                0.0,
                1.0,
            ),
            (
                "max_attribution_direction_breach_fraction",
                self.max_attribution_direction_breach_fraction,
                0.0,
                1.0,
            ),
            (
                "max_attribution_mismatch_fraction",
                self.max_attribution_mismatch_fraction,
                0.0,
                1.0,
            ),
            ("max_flat_effect_fraction", self.max_flat_effect_fraction, 0.0, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be between {lower} and {upper}")

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["schema_version"] = 1
        result["features"] = [
            asdict(item) for item in sorted(self.features, key=lambda x: x.feature)
        ]
        return result


@dataclass(frozen=True)
class FeatureDirectionSummary:
    feature: str
    direction: str
    case_count: int
    invalid_step_count: int
    model_direction_breach_count: int
    attribution_direction_breach_count: int
    attribution_mismatch_count: int
    flat_effect_count: int
    model_direction_breach_fraction: float
    attribution_direction_breach_fraction: float
    attribution_mismatch_fraction: float
    flat_effect_fraction: float


@dataclass(frozen=True)
class DirectionAuditReport:
    schema_version: int
    accepted: bool
    reason_codes: tuple[str, ...]
    explanation_method: str
    case_count: int
    feature_count: int
    summaries: tuple[FeatureDirectionSummary, ...]
    artifact_sha256: str
    policy_sha256: str
    model_sha256: str
    cohort_sha256: str
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("evidence_sha256")
        result["reason_codes"] = list(self.reason_codes)
        result["summaries"] = [asdict(item) for item in self.summaries]
        return result

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise DirectionArtifactError("NON_CANONICAL_JSON_VALUE") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DirectionArtifactError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def load_json(path: str | Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DirectionArtifactError("JSON_UNREADABLE") from exc
    if len(raw) > max_bytes:
        raise DirectionArtifactError("JSON_BYTE_BUDGET_EXCEEDED")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DirectionArtifactError("JSON_NOT_UTF8") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                DirectionArtifactError("NON_FINITE_JSON_NUMBER")
            ),
        )
    except DirectionArtifactError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DirectionArtifactError("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise DirectionArtifactError("JSON_NOT_OBJECT")
    return payload


def _exact_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise DirectionArtifactError(code)


def _safe_name(value: object, code: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise DirectionArtifactError(code)
    return value


def _sha(value: object, code: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise DirectionArtifactError(code)
    return value


def _finite(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DirectionArtifactError(code)
    return float(value)


def _fraction(value: object, code: str) -> float:
    result = _finite(value, code)
    if not 0.0 <= result <= 1.0:
        raise DirectionArtifactError(code)
    return result


def _integer(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectionArtifactError(code)
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise DirectionArtifactError("INVALID_GENERATED_AT")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DirectionArtifactError("INVALID_GENERATED_AT") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DirectionArtifactError("NAIVE_GENERATED_AT")
    return parsed.astimezone(timezone.utc)


def policy_from_dict(payload: dict[str, Any]) -> DirectionAuditPolicy:
    _exact_keys(payload, POLICY_KEYS, "POLICY_SCHEMA_MISMATCH")
    if _integer(payload["schema_version"], "INVALID_POLICY_SCHEMA") != 1:
        raise DirectionArtifactError("UNSUPPORTED_POLICY_SCHEMA")
    raw_features = payload["features"]
    if not isinstance(raw_features, list) or not raw_features:
        raise DirectionArtifactError("INVALID_FEATURE_CONTRACTS")
    features: list[FeatureDirectionContract] = []
    try:
        for item in raw_features:
            if not isinstance(item, dict):
                raise DirectionArtifactError("FEATURE_CONTRACT_NOT_OBJECT")
            _exact_keys(item, FEATURE_KEYS, "FEATURE_CONTRACT_SCHEMA_MISMATCH")
            features.append(
                FeatureDirectionContract(
                    feature=_safe_name(item["feature"], "INVALID_FEATURE_NAME"),
                    direction=item["direction"],
                    minimum_step=_finite(item["minimum_step"], "INVALID_MINIMUM_STEP"),
                )
            )
        return DirectionAuditPolicy(
            required_explanation_method=_safe_name(
                payload["required_explanation_method"], "INVALID_EXPLANATION_METHOD"
            ),
            features=tuple(features),
            min_cases_per_feature=_integer(payload["min_cases_per_feature"], "INVALID_MIN_CASES"),
            max_cases=_integer(payload["max_cases"], "INVALID_MAX_CASES"),
            max_features=_integer(payload["max_features"], "INVALID_MAX_FEATURES"),
            max_artifact_bytes=_integer(
                payload["max_artifact_bytes"], "INVALID_MAX_ARTIFACT_BYTES"
            ),
            max_age_seconds=_finite(payload["max_age_seconds"], "INVALID_MAX_AGE"),
            max_future_skew_seconds=_finite(
                payload["max_future_skew_seconds"], "INVALID_FUTURE_SKEW"
            ),
            direction_tolerance=_finite(
                payload["direction_tolerance"], "INVALID_DIRECTION_TOLERANCE"
            ),
            max_attribution_error=_finite(
                payload["max_attribution_error"], "INVALID_ATTRIBUTION_ERROR"
            ),
            minimum_material_effect=_finite(
                payload["minimum_material_effect"], "INVALID_MATERIAL_EFFECT"
            ),
            max_model_direction_breach_fraction=_fraction(
                payload["max_model_direction_breach_fraction"],
                "INVALID_MODEL_BREACH_FRACTION",
            ),
            max_attribution_direction_breach_fraction=_fraction(
                payload["max_attribution_direction_breach_fraction"],
                "INVALID_ATTRIBUTION_BREACH_FRACTION",
            ),
            max_attribution_mismatch_fraction=_fraction(
                payload["max_attribution_mismatch_fraction"],
                "INVALID_ATTRIBUTION_MISMATCH_FRACTION",
            ),
            max_flat_effect_fraction=_fraction(
                payload["max_flat_effect_fraction"], "INVALID_FLAT_EFFECT_FRACTION"
            ),
        )
    except DirectionArtifactError:
        raise
    except ValueError as exc:
        raise DirectionArtifactError("INVALID_POLICY_VALUE") from exc


def _validate_artifact(payload: dict[str, Any], policy: DirectionAuditPolicy) -> None:
    if len(_canonical(payload)) > policy.max_artifact_bytes:
        raise DirectionArtifactError("ARTIFACT_BYTE_BUDGET_EXCEEDED")
    _exact_keys(payload, ARTIFACT_KEYS, "ARTIFACT_SCHEMA_MISMATCH")
    if _integer(payload["schema_version"], "INVALID_ARTIFACT_SCHEMA") != 1:
        raise DirectionArtifactError("UNSUPPORTED_ARTIFACT_SCHEMA")
    _timestamp(payload["generated_at"])
    _sha(payload["model_sha256"], "INVALID_MODEL_DIGEST")
    _sha(payload["cohort_sha256"], "INVALID_COHORT_DIGEST")
    method = _safe_name(payload["explanation_method"], "INVALID_EXPLANATION_METHOD")
    if method != policy.required_explanation_method:
        raise DirectionArtifactError("EXPLANATION_METHOD_MISMATCH")
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > policy.max_cases:
        raise DirectionArtifactError("INVALID_CASE_LIST")
    declared = {item.feature for item in policy.features}
    case_ids: set[str] = set()
    for item in cases:
        if not isinstance(item, dict):
            raise DirectionArtifactError("CASE_NOT_OBJECT")
        _exact_keys(item, CASE_KEYS, "CASE_SCHEMA_MISMATCH")
        case_id = _safe_name(item["case_id"], "INVALID_CASE_ID")
        if case_id in case_ids:
            raise DirectionArtifactError("DUPLICATE_CASE_ID")
        case_ids.add(case_id)
        feature = _safe_name(item["feature"], "INVALID_FEATURE_NAME")
        if feature not in declared:
            raise DirectionArtifactError("UNDECLARED_FEATURE")
        _finite(item["baseline_value"], "INVALID_BASELINE_VALUE")
        _finite(item["perturbed_value"], "INVALID_PERTURBED_VALUE")
        baseline_probability = _finite(item["baseline_probability"], "INVALID_BASELINE_PROBABILITY")
        perturbed_probability = _finite(
            item["perturbed_probability"], "INVALID_PERTURBED_PROBABILITY"
        )
        if not 0 <= baseline_probability <= 1 or not 0 <= perturbed_probability <= 1:
            raise DirectionArtifactError("PROBABILITY_OUT_OF_RANGE")
        _finite(item["attribution_delta"], "INVALID_ATTRIBUTION_DELTA")


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def audit_explanation_directions(
    payload: dict[str, Any],
    policy: DirectionAuditPolicy,
    *,
    now: datetime | None = None,
) -> DirectionAuditReport:
    """Check model and local-attribution direction against governed feature contracts."""

    _validate_artifact(payload, policy)
    reasons: set[str] = set()
    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    age = (
        observed_at.astimezone(timezone.utc) - _timestamp(payload["generated_at"])
    ).total_seconds()
    if age > policy.max_age_seconds:
        reasons.add("ARTIFACT_STALE")
    if age < -policy.max_future_skew_seconds:
        reasons.add("ARTIFACT_FROM_FUTURE")

    cases_by_feature: dict[str, list[dict[str, Any]]] = {
        item.feature: [] for item in policy.features
    }
    for item in payload["cases"]:
        cases_by_feature[item["feature"]].append(item)

    summaries: list[FeatureDirectionSummary] = []
    for contract in sorted(policy.features, key=lambda item: item.feature):
        cases = cases_by_feature[contract.feature]
        invalid_steps = 0
        model_breaches = 0
        attribution_breaches = 0
        attribution_mismatches = 0
        flat_effects = 0
        sign = 1.0 if contract.direction == "nondecreasing_risk" else -1.0
        for item in cases:
            input_delta = float(item["perturbed_value"]) - float(item["baseline_value"])
            probability_delta = float(item["perturbed_probability"]) - float(
                item["baseline_probability"]
            )
            attribution_delta = float(item["attribution_delta"])
            if sign * input_delta < contract.minimum_step:
                invalid_steps += 1
            # Every pair moves the governed feature in its risk-increasing direction.
            # The input sign depends on the contract, while the expected risk and
            # attribution deltas are therefore non-negative for both contract types.
            if probability_delta < -policy.direction_tolerance:
                model_breaches += 1
            if attribution_delta < -policy.direction_tolerance:
                attribution_breaches += 1
            if abs(attribution_delta - probability_delta) > policy.max_attribution_error:
                attribution_mismatches += 1
            if abs(probability_delta) < policy.minimum_material_effect:
                flat_effects += 1

        total = len(cases)
        summary = FeatureDirectionSummary(
            feature=contract.feature,
            direction=contract.direction,
            case_count=total,
            invalid_step_count=invalid_steps,
            model_direction_breach_count=model_breaches,
            attribution_direction_breach_count=attribution_breaches,
            attribution_mismatch_count=attribution_mismatches,
            flat_effect_count=flat_effects,
            model_direction_breach_fraction=_rate(model_breaches, total),
            attribution_direction_breach_fraction=_rate(attribution_breaches, total),
            attribution_mismatch_fraction=_rate(attribution_mismatches, total),
            flat_effect_fraction=_rate(flat_effects, total),
        )
        summaries.append(summary)
        if total < policy.min_cases_per_feature:
            reasons.add("FEATURE_SAMPLE_INSUFFICIENT")
        if invalid_steps:
            reasons.add("PERTURBATION_DIRECTION_CONTRACT_BREACH")
        if summary.model_direction_breach_fraction > policy.max_model_direction_breach_fraction:
            reasons.add("MODEL_DIRECTION_BREACH_RATE_EXCEEDED")
        if (
            summary.attribution_direction_breach_fraction
            > policy.max_attribution_direction_breach_fraction
        ):
            reasons.add("ATTRIBUTION_DIRECTION_BREACH_RATE_EXCEEDED")
        if summary.attribution_mismatch_fraction > policy.max_attribution_mismatch_fraction:
            reasons.add("ATTRIBUTION_MISMATCH_RATE_EXCEEDED")
        if summary.flat_effect_fraction > policy.max_flat_effect_fraction:
            reasons.add("FLAT_EFFECT_RATE_EXCEEDED")

    canonical_artifact = {**payload, "cases": sorted(payload["cases"], key=lambda x: x["case_id"])}
    report = DirectionAuditReport(
        schema_version=1,
        accepted=not reasons,
        reason_codes=tuple(sorted(reasons)),
        explanation_method=payload["explanation_method"],
        case_count=len(payload["cases"]),
        feature_count=len(policy.features),
        summaries=tuple(summaries),
        artifact_sha256=_digest(canonical_artifact),
        policy_sha256=_digest(policy.as_dict()),
        model_sha256=payload["model_sha256"],
        cohort_sha256=payload["cohort_sha256"],
        evidence_sha256="",
    )
    return replace(report, evidence_sha256=_digest(report.body()))


def _malformed(code: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "accepted": False,
        "status": "malformed",
        "reason_codes": [code],
    }
    return {**body, "evidence_sha256": _digest(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local explanation direction contracts")
    parser.add_argument("artifact", help="Directional perturbation evidence JSON")
    parser.add_argument("--policy", required=True, help="Server-owned direction policy JSON")
    parser.add_argument("--now", help="Timezone-aware ISO-8601 evaluation time")
    args = parser.parse_args(argv)
    try:
        policy = policy_from_dict(load_json(args.policy, max_bytes=65_536))
        payload = load_json(args.artifact, max_bytes=policy.max_artifact_bytes)
        report = audit_explanation_directions(
            payload,
            policy,
            now=_timestamp(args.now) if args.now else None,
        )
    except (DirectionArtifactError, ValueError) as exc:
        code = exc.code if isinstance(exc, DirectionArtifactError) else "INVALID_POLICY_VALUE"
        print(json.dumps(_malformed(code), sort_keys=True, separators=(",", ":")))
        return 3
    print(json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
