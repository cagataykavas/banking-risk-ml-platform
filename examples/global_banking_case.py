from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.banking_metrics import validation_report
from src.decision_economics import EconomicsAssumptions, optimize_thresholds
from src.risk_pipeline import (
    CATEGORICAL,
    NUMERIC,
    TARGET,
    build_pipeline,
    make_dataset,
    temporal_split,
)
from src.stress_testing import default_scenarios, stress_matrix


def run_case(*, rows: int = 8000, seed: int = 42) -> dict[str, Any]:
    """Run a synthetic modeling-to-policy case suitable for portfolio/interview walkthroughs."""
    if rows < 500:
        raise ValueError("rows must be >= 500 so the synthetic holdout is meaningful")

    data = make_dataset(rows, seed=seed)
    train_df, test_df = temporal_split(data)

    model = build_pipeline()
    model.fit(train_df[NUMERIC + CATEGORICAL], train_df[TARGET])
    probability = model.predict_proba(test_df[NUMERIC + CATEGORICAL])[:, 1]
    y_true = test_df[TARGET].to_numpy(dtype=int)

    metrics = validation_report(y_true, probability)
    assumptions = EconomicsAssumptions(
        approved_account_revenue=240.0,
        loss_given_default=3000.0,
        manual_review_cost=22.0,
        false_decline_opportunity_cost=140.0,
        review_default_capture_rate=0.75,
    )
    policies = optimize_thresholds(
        y_true,
        probability,
        assumptions=assumptions,
        max_review_rate=0.30,
    )
    if policies.empty:
        raise RuntimeError("no feasible threshold policy found")

    selected = policies.iloc[0].to_dict()
    approve_below = float(selected["approve_below"])
    reject_above = float(selected["reject_above"])

    # Synthetic debt is used as an exposure proxy solely for this public stress example.
    exposure_proxy = np.maximum(test_df["debt"].to_numpy(dtype=float), 500.0)
    stress = stress_matrix(
        y_true=y_true,
        base_pd=probability,
        balances=exposure_proxy,
        approve_below=approve_below,
        decline_above=reject_above,
        scenarios=default_scenarios(),
    )

    return {
        "scope": {
            "synthetic": True,
            "rows": rows,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "seed": seed,
            "note": (
                "Offline portfolio demonstration only; economics are illustrative and "
                "not a causal or regulatory credit-policy estimate."
            ),
        },
        "model_validation": metrics,
        "economics_assumptions": asdict(assumptions),
        "selected_policy": {key: float(value) for key, value in selected.items()},
        "top_policy_candidates": [
            {key: float(value) for key, value in row.items()}
            for row in policies.head(5).to_dict(orient="records")
        ],
        "stress_scenarios": [
            {
                key: (str(value) if key == "scenario" else float(value))
                for key, value in row.items()
            }
            for row in stress.to_dict(orient="records")
        ],
    }


def markdown_summary(report: dict[str, Any]) -> str:
    metrics = report["model_validation"]
    policy = report["selected_policy"]
    stress = report["stress_scenarios"]

    lines = [
        "# Synthetic Global Banking Decision Case",
        "",
        "> Public portfolio demonstration using synthetic data and illustrative economics.",
        "",
        "## Model validation",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "roc_auc",
        "average_precision",
        "ks",
        "lift_at_10pct",
        "brier_score",
        "ece",
    ):
        lines.append(f"| {key} | {metrics[key]:.4f} |")

    lines.extend(
        [
            "",
            "## Selected operating policy",
            "",
            f"- auto-approve below PD: **{policy['approve_below']:.3f}**",
            f"- auto-decline at/above PD: **{policy['reject_above']:.3f}**",
            f"- approval rate: **{policy['approval_rate']:.1%}**",
            f"- manual-review rate: **{policy['review_rate']:.1%}**",
            f"- decline rate: **{policy['reject_rate']:.1%}**",
            f"- approved bad rate: **{policy['bad_rate_approved']:.1%}**",
            f"- synthetic expected value/application: **{policy['expected_value_per_application']:.2f}**",
            "",
            "## Stress matrix",
            "",
            "| Scenario | Mean PD | Expected loss/account | Policy cost/application | Loss uplift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in stress:
        lines.append(
            "| {scenario} | {mean_pd:.3f} | {expected_loss_per_account:.2f} | "
            "{policy_cost_per_application:.2f} | {loss_uplift_vs_baseline:.1%} |".format(**row)
        )

    interpretation = (
        "The model-selection question and the business-policy question are intentionally separate. "
        "Discrimination/calibration metrics describe probability quality; the two-threshold policy "
        "adds review capacity and synthetic economics; stress testing then asks how a fixed policy "
        "behaves when PD/LGD/exposure assumptions deteriorate."
    )
    limitations = (
        "The debt field is used only as a synthetic exposure proxy in this public stress example. "
        "A real banking implementation would require governed EAD/LGD definitions, selection-bias "
        "analysis, portfolio constraints, mature outcomes and policy approval."
    )
    lines.extend(["", "## Interpretation", "", interpretation, "", limitations])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic Global Banking DS case")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_banking_case.json"))
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/global_banking_case.md"),
    )
    args = parser.parse_args()

    report = run_case(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_summary(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
