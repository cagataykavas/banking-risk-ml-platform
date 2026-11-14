from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureRule:
    name: str
    kind: FeatureKind
    max_missing_rate: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: tuple[str, ...] = ()
    max_unknown_rate: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name must not be blank")
        for value, label in (
            (self.max_missing_rate, "max_missing_rate"),
            (self.max_unknown_rate, "max_unknown_rate"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be within [0, 1]")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not exceed maximum")
        if any(
            value is not None and not math.isfinite(value)
            for value in (self.minimum, self.maximum)
        ):
            raise ValueError("numeric bounds must be finite")
        if self.kind not in {"numeric", "categorical"}:
            raise ValueError("kind must be numeric or categorical")
        if self.kind == "numeric" and self.allowed_values:
            raise ValueError("numeric features cannot define allowed_values")
        if self.kind == "categorical" and not self.allowed_values:
            raise ValueError("categorical features require allowed_values")
        if len(set(self.allowed_values)) != len(self.allowed_values):
            raise ValueError("allowed_values must be unique")
        if any(not value for value in self.allowed_values):
            raise ValueError("allowed_values must not contain blanks")


@dataclass(frozen=True)
class FeatureContract:
    version: str
    features: tuple[FeatureRule, ...]
    allow_extra_columns: bool = False

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("contract version must not be blank")
        if not self.features:
            raise ValueError("feature contract must contain at least one feature")
        names = [rule.name for rule in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")


@dataclass(frozen=True)
class ContractIssue:
    code: str
    feature: str | None
    count: int
    rate: float
    threshold: float | None = None


@dataclass(frozen=True)
class ContractReport:
    valid: bool
    contract_sha256: str
    row_count: int
    issues: tuple[ContractIssue, ...]
    feature_stats: dict[str, dict[str, float | int]]


def _contract_payload(contract: FeatureContract) -> dict:
    return {
        "version": contract.version,
        "allow_extra_columns": contract.allow_extra_columns,
        "features": [asdict(rule) for rule in contract.features],
    }


def contract_sha256(contract: FeatureContract) -> str:
    payload = json.dumps(
        _contract_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def credit_application_contract() -> FeatureContract:
    return FeatureContract(
        version="credit-application-v1",
        features=(
            FeatureRule("age", "numeric", minimum=18, maximum=100),
            FeatureRule("income", "numeric", minimum=0.01, maximum=10_000_000),
            FeatureRule("debt", "numeric", minimum=0, maximum=10_000_000),
            FeatureRule("utilization", "numeric", minimum=0, maximum=1),
            FeatureRule("late_payments", "numeric", minimum=0, maximum=100),
            FeatureRule("account_age_months", "numeric", minimum=0, maximum=1_200),
            FeatureRule(
                "employment",
                "categorical",
                allowed_values=("salaried", "self_employed", "student", "unemployed"),
            ),
            FeatureRule(
                "housing",
                "categorical",
                allowed_values=("rent", "mortgage", "owned"),
            ),
        ),
    )


def validate_frame(frame: pd.DataFrame, contract: FeatureContract) -> ContractReport:
    issues: list[ContractIssue] = []
    stats: dict[str, dict[str, float | int]] = {}
    row_count = len(frame)

    if row_count == 0:
        issues.append(ContractIssue("empty_batch", None, 0, 1.0))

    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        issues.append(
            ContractIssue("duplicate_columns", None, len(duplicate_columns), 1.0)
        )

    expected = {rule.name for rule in contract.features}
    observed = set(frame.columns)
    for feature in sorted(expected - observed):
        issues.append(ContractIssue("missing_feature", feature, row_count, 1.0))
    if not contract.allow_extra_columns:
        for feature in sorted(observed - expected):
            issues.append(ContractIssue("unexpected_feature", feature, row_count, 1.0))

    for rule in contract.features:
        if rule.name not in frame.columns or rule.name in duplicate_columns:
            continue
        series = frame[rule.name]
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / row_count if row_count else 0.0
        feature_stats: dict[str, float | int] = {
            "missing_count": missing_count,
            "missing_rate": missing_rate,
        }
        stats[rule.name] = feature_stats
        if missing_rate > rule.max_missing_rate:
            issues.append(
                ContractIssue(
                    "missing_rate_exceeded",
                    rule.name,
                    missing_count,
                    missing_rate,
                    rule.max_missing_rate,
                )
            )

        nonmissing = series.dropna()
        if rule.kind == "numeric":
            if not is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype):
                issues.append(
                    ContractIssue("invalid_numeric_dtype", rule.name, row_count, 1.0)
                )
                continue
            values = nonmissing.to_numpy(dtype=float)
            finite = np.isfinite(values)
            nonfinite_count = int((~finite).sum())
            feature_stats["nonfinite_count"] = nonfinite_count
            if nonfinite_count:
                issues.append(
                    ContractIssue(
                        "nonfinite_numeric",
                        rule.name,
                        nonfinite_count,
                        nonfinite_count / max(len(nonmissing), 1),
                        0.0,
                    )
                )
            finite_values = values[finite]
            out_of_range = np.zeros(len(finite_values), dtype=bool)
            if rule.minimum is not None:
                out_of_range |= finite_values < rule.minimum
            if rule.maximum is not None:
                out_of_range |= finite_values > rule.maximum
            out_of_range_count = int(out_of_range.sum())
            feature_stats["out_of_range_count"] = out_of_range_count
            if out_of_range_count:
                issues.append(
                    ContractIssue(
                        "numeric_out_of_range",
                        rule.name,
                        out_of_range_count,
                        out_of_range_count / max(len(finite_values), 1),
                        0.0,
                    )
                )
            continue

        categorical_dtype = (
            is_string_dtype(series.dtype)
            or is_object_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not categorical_dtype:
            issues.append(
                ContractIssue("invalid_categorical_dtype", rule.name, row_count, 1.0)
            )
            continue
        unknown_count = int((~nonmissing.isin(rule.allowed_values)).sum())
        unknown_rate = unknown_count / len(nonmissing) if len(nonmissing) else 0.0
        feature_stats["unknown_count"] = unknown_count
        feature_stats["unknown_rate"] = unknown_rate
        if unknown_rate > rule.max_unknown_rate:
            issues.append(
                ContractIssue(
                    "unknown_category_rate_exceeded",
                    rule.name,
                    unknown_count,
                    unknown_rate,
                    rule.max_unknown_rate,
                )
            )

    return ContractReport(
        valid=not issues,
        contract_sha256=contract_sha256(contract),
        row_count=row_count,
        issues=tuple(issues),
        feature_stats=stats,
    )


def _write_json_atomically(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def persist_contract_evidence(
    output: Path,
    contract: FeatureContract,
    report: ContractReport,
) -> None:
    digest = contract_sha256(contract)
    if report.contract_sha256 != digest:
        raise ValueError("report does not belong to this feature contract")
    contract_payload = {**_contract_payload(contract), "contract_sha256": digest}
    report_payload = asdict(report)
    _write_json_atomically(output / "feature_contract.json", contract_payload)
    _write_json_atomically(output / "feature_contract_validation.json", report_payload)
