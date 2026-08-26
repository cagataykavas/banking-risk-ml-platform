import numpy as np

from src.stress_testing import (
    StressScenario,
    default_scenarios,
    stress_matrix,
    stress_probabilities,
)


def test_positive_logit_shift_increases_mean_pd():
    base = np.array([0.02, 0.05, 0.10, 0.25, 0.50])
    stressed = stress_probabilities(base, StressScenario("stress", logit_shift=0.5))
    assert stressed.mean() > base.mean()
    assert np.all((stressed >= 0) & (stressed <= 1))


def test_severe_scenario_increases_expected_loss():
    rng = np.random.default_rng(3)
    base_pd = np.clip(rng.beta(1.2, 8.0, 1500), 0.005, 0.9)
    y = rng.binomial(1, base_pd)
    balances = rng.lognormal(9.0, 0.6, len(y))

    report = stress_matrix(
        y_true=y,
        base_pd=base_pd,
        balances=balances,
        approve_below=0.08,
        decline_above=0.65,
        scenarios=default_scenarios(),
    )

    baseline = report.loc[report["scenario"] == "baseline"].iloc[0]
    severe = report.loc[report["scenario"] == "severe_downturn"].iloc[0]
    assert severe["expected_loss"] > baseline["expected_loss"]
    assert severe["loss_uplift_vs_baseline"] > 0
