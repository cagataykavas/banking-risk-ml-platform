from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class SelectionAuditPolicy:
    min_rows: int = 500
    min_events: int = 30
    min_nonevents: int = 100
    propensity_floor: float = 0.10
    min_effective_sample_fraction: float = 0.50
    max_weight_share: float = 0.02
    max_rows: int = 1_000_000


@dataclass(frozen=True)
class ProbabilityMetrics:
    roc_auc: float
    average_precision: float
    brier_score: float
    event_rate: float
    mean_score: float
    calibration_gap: float


def _numeric_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{name} must not contain booleans")
    try:
        vector = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_policy(policy: SelectionAuditPolicy) -> None:
    if not 2 <= policy.min_rows <= policy.max_rows:
        raise ValueError("min_rows must be between 2 and max_rows")
    if policy.min_events < 1 or policy.min_nonevents < 1:
        raise ValueError("minimum event and non-event counts must be positive")
    if policy.min_events + policy.min_nonevents > policy.max_rows:
        raise ValueError("minimum class counts exceed max_rows")
    if not 0 < policy.propensity_floor <= 1:
        raise ValueError("propensity_floor must be in (0, 1]")
    if not 0 < policy.min_effective_sample_fraction <= 1:
        raise ValueError("min_effective_sample_fraction must be in (0, 1]")
    if not 0 < policy.max_weight_share <= 1:
        raise ValueError("max_weight_share must be in (0, 1]")


def _probability_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> ProbabilityMetrics:
    if sample_weight is None:
        normalized = np.full(len(labels), 1.0 / len(labels))
    else:
        normalized = sample_weight / np.sum(sample_weight)
    event_rate = float(np.sum(normalized * labels))
    mean_score = float(np.sum(normalized * scores))
    return ProbabilityMetrics(
        roc_auc=float(roc_auc_score(labels, scores, sample_weight=sample_weight)),
        average_precision=float(
            average_precision_score(labels, scores, sample_weight=sample_weight)
        ),
        brier_score=float(np.sum(normalized * np.square(labels - scores))),
        event_rate=event_rate,
        mean_score=mean_score,
        calibration_gap=event_rate - mean_score,
    )


def _evidence_digest(report: dict[str, object]) -> str:
    payload = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_selection_weighted_performance(
    y_true: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    observation_propensity: Sequence[float] | np.ndarray,
    *,
    policy: SelectionAuditPolicy | None = None,
) -> dict[str, object]:
    """Compare booked-sample metrics with self-normalized inverse-propensity metrics.

    The caller supplies the probability that each labeled observation entered the
    observed sample under the historical policy. The audit does not infer rejected
    outcomes and does not estimate the propensity model.
    """
    active_policy = policy or SelectionAuditPolicy()
    _validate_policy(active_policy)
    labels = _numeric_vector("y_true", y_true)
    probabilities = _numeric_vector("scores", scores)
    propensities = _numeric_vector("observation_propensity", observation_propensity)

    if not (len(labels) == len(probabilities) == len(propensities)):
        raise ValueError(
            "y_true, scores, and observation_propensity must have equal lengths"
        )
    if len(labels) < 2:
        raise ValueError("at least two observations are required")
    if len(labels) > active_policy.max_rows:
        raise ValueError("observation count exceeds max_rows")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain binary 0/1 labels")
    if np.unique(labels).size != 2:
        raise ValueError("y_true must contain both event and non-event labels")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("scores must be probabilities in [0, 1]")
    if np.any((propensities <= 0) | (propensities > 1)):
        raise ValueError("observation_propensity must be in (0, 1]")

    weights = np.reciprocal(propensities)
    weight_sum = float(np.sum(weights))
    normalized_weights = weights / weight_sum
    effective_sample_size = float(weight_sum**2 / np.sum(np.square(weights)))
    effective_sample_fraction = effective_sample_size / len(labels)
    max_weight_share = float(np.max(normalized_weights))
    event_count = int(np.sum(labels))
    nonevent_count = len(labels) - event_count

    findings: list[str] = []
    if len(labels) < active_policy.min_rows:
        findings.append("INSUFFICIENT_ROWS")
    if event_count < active_policy.min_events:
        findings.append("INSUFFICIENT_EVENTS")
    if nonevent_count < active_policy.min_nonevents:
        findings.append("INSUFFICIENT_NONEVENTS")
    if float(np.min(propensities)) < active_policy.propensity_floor:
        findings.append("PROPENSITY_OVERLAP_FAILURE")
    if effective_sample_fraction < active_policy.min_effective_sample_fraction:
        findings.append("LOW_EFFECTIVE_SAMPLE_SIZE")
    if max_weight_share > active_policy.max_weight_share:
        findings.append("DOMINANT_OBSERVATION_WEIGHT")

    unweighted = _probability_metrics(labels, probabilities)
    weighted = _probability_metrics(labels, probabilities, weights)
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": not findings,
        "findings": findings,
        "policy": asdict(active_policy),
        "sample": {
            "rows": len(labels),
            "events": event_count,
            "nonevents": nonevent_count,
        },
        "weight_diagnostics": {
            "minimum_propensity": float(np.min(propensities)),
            "maximum_propensity": float(np.max(propensities)),
            "maximum_inverse_weight": float(np.max(weights)),
            "maximum_normalized_weight_share": max_weight_share,
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": effective_sample_fraction,
        },
        "unweighted_metrics": asdict(unweighted),
        "self_normalized_ipw_metrics": asdict(weighted),
        "caveat": (
            "Inverse-propensity weighting requires a correct historical selection model, "
            "positivity, and no unmeasured selection drivers. It does not recover rejected outcomes."
        ),
    }
    report["evidence_sha256"] = _evidence_digest(report)
    return report


def _synthetic_demo() -> dict[str, object]:
    rng = np.random.default_rng(2026)
    latent = rng.normal(size=4_000)
    true_probability = 1 / (1 + np.exp(-(-2.0 + 1.1 * latent)))
    labels = rng.binomial(1, true_probability)
    scores = np.clip(true_probability * 0.92 + 0.015, 0.001, 0.999)
    selection_probability = np.clip(0.88 - 0.55 * scores, 0.20, 0.95)
    observed = rng.random(len(labels)) < selection_probability
    return audit_selection_weighted_performance(
        labels[observed],
        scores[observed],
        selection_probability[observed],
    )


if __name__ == "__main__":
    result = _synthetic_demo()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)
