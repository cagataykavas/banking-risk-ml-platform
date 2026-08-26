from src.champion_challenger import ModelSnapshot, PromotionPolicy, compare


def snapshot(**overrides):
    values = {
        "name": "model",
        "version": "1",
        "roc_auc": 0.78,
        "average_precision": 0.36,
        "brier_score": 0.11,
        "expected_cost_per_application": 62.0,
        "p95_latency_ms": 25.0,
        "psi_score": 0.08,
        "review_rate": 0.20,
    }
    values.update(overrides)
    return ModelSnapshot(**values)


def test_good_challenger_promotes():
    champion = snapshot(name="champion")
    challenger = snapshot(
        name="challenger",
        roc_auc=0.79,
        average_precision=0.38,
        brier_score=0.108,
        expected_cost_per_application=58.0,
        p95_latency_ms=28.0,
        psi_score=0.09,
        review_rate=0.22,
    )
    decision = compare(champion, challenger)
    assert decision.promote
    assert decision.reasons == ()


def test_latency_or_review_regression_blocks_promotion():
    champion = snapshot(name="champion")
    challenger = snapshot(
        name="challenger",
        roc_auc=0.80,
        average_precision=0.39,
        expected_cost_per_application=55.0,
        p95_latency_ms=50.0,
        review_rate=0.31,
    )
    decision = compare(champion, challenger)
    assert not decision.promote
    assert any("latency" in reason for reason in decision.reasons)
    assert any("review" in reason for reason in decision.reasons)


def test_policy_can_require_material_cost_gain():
    champion = snapshot(expected_cost_per_application=62.0)
    challenger = snapshot(
        roc_auc=0.79,
        average_precision=0.38,
        expected_cost_per_application=61.5,
    )
    policy = PromotionPolicy(min_cost_improvement=2.0)
    decision = compare(champion, challenger, policy)
    assert not decision.promote
    assert any("business cost" in reason for reason in decision.reasons)
