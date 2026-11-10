from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.selection_bias import (
    SelectionAuditPolicy,
    audit_selection_weighted_performance,
)

TEST_POLICY = SelectionAuditPolicy(
    min_rows=4,
    min_events=1,
    min_nonevents=1,
    propensity_floor=0.05,
    min_effective_sample_fraction=0.20,
    max_weight_share=0.80,
    max_rows=100,
)


def test_equal_propensities_preserve_probability_metrics():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    report = audit_selection_weighted_performance(
        labels,
        scores,
        np.full(4, 0.5),
        policy=TEST_POLICY,
    )

    assert report["passed"] is True
    assert report["unweighted_metrics"] == report["self_normalized_ipw_metrics"]
    assert report["weight_diagnostics"]["effective_sample_size"] == 4.0


def test_inverse_propensity_weights_change_selected_sample_event_rate():
    report = audit_selection_weighted_performance(
        [1, 0, 0, 0],
        [0.8, 0.2, 0.1, 0.3],
        [0.2, 0.8, 0.8, 0.8],
        policy=TEST_POLICY,
    )

    assert report["unweighted_metrics"]["event_rate"] == 0.25
    assert report["self_normalized_ipw_metrics"]["event_rate"] == pytest.approx(4 / 7)
    assert report["self_normalized_ipw_metrics"]["event_rate"] > 0.25


def test_overlap_and_weight_concentration_fail_closed():
    policy = SelectionAuditPolicy(
        min_rows=4,
        min_events=1,
        min_nonevents=1,
        propensity_floor=0.10,
        min_effective_sample_fraction=0.80,
        max_weight_share=0.50,
        max_rows=100,
    )
    report = audit_selection_weighted_performance(
        [1, 0, 1, 0],
        [0.8, 0.2, 0.7, 0.1],
        [0.01, 0.9, 0.9, 0.9],
        policy=policy,
    )

    assert report["passed"] is False
    assert report["findings"] == [
        "PROPENSITY_OVERLAP_FAILURE",
        "LOW_EFFECTIVE_SAMPLE_SIZE",
        "DOMINANT_OBSERVATION_WEIGHT",
    ]


@pytest.mark.parametrize(
    ("labels", "scores", "propensities", "message"),
    [
        ([0, 1], [0.1], [0.5, 0.5], "equal lengths"),
        ([0, 2], [0.1, 0.9], [0.5, 0.5], "binary 0/1"),
        ([0, 1], [0.1, np.nan], [0.5, 0.5], "finite"),
        ([0, 1], [0.1, 1.1], [0.5, 0.5], "in [0, 1]"),
        ([0, 1], [0.1, 0.9], [0.0, 0.5], "in (0, 1]"),
        ([False, True], [0.1, 0.9], [0.5, 0.5], "booleans"),
    ],
)
def test_malformed_evidence_is_rejected(labels, scores, propensities, message):
    with pytest.raises(ValueError) as error:
        audit_selection_weighted_performance(
            labels,
            scores,
            propensities,
            policy=TEST_POLICY,
        )
    assert message in str(error.value)


def test_evidence_digest_is_deterministic_and_privacy_reduced():
    args = ([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], [0.4, 0.5, 0.6, 0.7])
    first = audit_selection_weighted_performance(*args, policy=TEST_POLICY)
    second = audit_selection_weighted_performance(*args, policy=TEST_POLICY)

    assert first == second
    evidence_sha256 = first.pop("evidence_sha256")
    canonical = json.dumps(
        first, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == evidence_sha256
    assert "rows" in first["sample"]
    assert "labels" not in first
    assert "scores" not in first
