import numpy as np
import pandas as pd

from src.shadow_evaluation import (
    DecisionPolicy,
    disagreement_slice,
    outcome_comparison,
    shadow_summary,
)


def test_shadow_summary_detects_route_and_latency_changes():
    frame = pd.DataFrame(
        {
            "champion_score": [0.02, 0.10, 0.70, 0.20],
            "challenger_score": [0.03, 0.05, 0.60, 0.75],
            "champion_latency_ms": [20, 20, 20, 20],
            "challenger_latency_ms": [25, 25, 25, 25],
        }
    )
    report = shadow_summary(frame, policy=DecisionPolicy(approve_below=0.08, decline_above=0.65))
    assert report.rows == 4
    assert report.route_disagreement_rate > 0
    assert report.latency_p95_ratio > 1


def test_outcome_comparison_reports_metric_deltas():
    rng = np.random.default_rng(21)
    y = rng.binomial(1, 0.2, 1000)
    champion = np.clip(0.10 + 0.55 * y + rng.normal(0, 0.15, len(y)), 0.001, 0.999)
    challenger = np.clip(0.08 + 0.65 * y + rng.normal(0, 0.12, len(y)), 0.001, 0.999)
    frame = pd.DataFrame(
        {
            "default_label": y,
            "champion_score": champion,
            "challenger_score": challenger,
        }
    )
    result = outcome_comparison(frame)
    assert result.rows == 1000
    assert np.isfinite(result.auc_delta)
    assert np.isfinite(result.brier_delta)


def test_disagreement_slice_returns_only_changed_routes():
    frame = pd.DataFrame(
        {
            "application_id": ["a", "b", "c"],
            "champion_score": [0.02, 0.20, 0.80],
            "challenger_score": [0.03, 0.75, 0.70],
        }
    )
    changed = disagreement_slice(frame)
    assert list(changed["application_id"]) == ["b"]
    assert (changed["champion_route"] != changed["challenger_route"]).all()
