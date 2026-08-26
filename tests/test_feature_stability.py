import pandas as pd

from src.feature_stability import flag_unstable_features, rank_stability_matrix, summarize_importance_runs


def test_stable_feature_ranks_high():
    frame = pd.DataFrame(
        [
            {"run": 0, "feature": "utilization", "importance": 0.50},
            {"run": 0, "feature": "income", "importance": 0.10},
            {"run": 1, "feature": "utilization", "importance": 0.48},
            {"run": 1, "feature": "income", "importance": 0.05},
            {"run": 2, "feature": "utilization", "importance": 0.52},
            {"run": 2, "feature": "income", "importance": -0.12},
        ]
    )
    summary = summarize_importance_runs(frame, top_k=1)
    first = summary.iloc[0]
    assert first["feature"] == "utilization"
    assert first["top_k_frequency"] == 1.0
    assert first["sign_consistency"] == 1.0


def test_unstable_feature_is_flagged():
    frame = pd.DataFrame(
        [
            {"run": 0, "feature": "unstable", "importance": 0.30},
            {"run": 1, "feature": "unstable", "importance": -0.31},
            {"run": 2, "feature": "unstable", "importance": 0.02},
            {"run": 0, "feature": "stable", "importance": 0.60},
            {"run": 1, "feature": "stable", "importance": 0.59},
            {"run": 2, "feature": "stable", "importance": 0.61},
        ]
    )
    summary = summarize_importance_runs(frame, top_k=2)
    flagged = flag_unstable_features(summary, max_cv=0.5, min_sign_consistency=0.8)
    assert "unstable" in set(flagged["feature"])


def test_rank_stability_matrix_is_symmetric():
    frame = pd.DataFrame(
        [
            {"run": 0, "feature": "a", "importance": 0.9},
            {"run": 0, "feature": "b", "importance": 0.4},
            {"run": 1, "feature": "a", "importance": 0.8},
            {"run": 1, "feature": "b", "importance": 0.5},
        ]
    )
    matrix = rank_stability_matrix(frame)
    assert matrix.loc[0, 1] == matrix.loc[1, 0]
    assert matrix.loc[0, 0] == 1.0
