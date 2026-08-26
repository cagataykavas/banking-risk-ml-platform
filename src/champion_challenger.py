from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSnapshot:
    name: str
    version: str
    roc_auc: float
    average_precision: float
    brier_score: float
    expected_cost_per_application: float
    p95_latency_ms: float
    psi_score: float
    review_rate: float


@dataclass(frozen=True)
class PromotionPolicy:
    min_auc_gain: float = 0.002
    min_ap_gain: float = 0.003
    max_brier_regression: float = 0.005
    min_cost_improvement: float = 0.0
    max_latency_ratio: float = 1.35
    max_psi: float = 0.25
    max_review_rate_increase: float = 0.05


DEFAULT_PROMOTION_POLICY = PromotionPolicy()


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]
    deltas: dict[str, float]


def compare(
    champion: ModelSnapshot,
    challenger: ModelSnapshot,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    active_policy = policy or DEFAULT_PROMOTION_POLICY
    deltas = {
        "roc_auc": challenger.roc_auc - champion.roc_auc,
        "average_precision": challenger.average_precision - champion.average_precision,
        "brier_score": challenger.brier_score - champion.brier_score,
        "expected_cost_per_application": (
            challenger.expected_cost_per_application - champion.expected_cost_per_application
        ),
        "latency_ratio": challenger.p95_latency_ms / max(champion.p95_latency_ms, 1e-9),
        "review_rate": challenger.review_rate - champion.review_rate,
        "psi_score": challenger.psi_score,
    }

    reasons: list[str] = []

    discrimination_improved = (
        deltas["roc_auc"] >= active_policy.min_auc_gain
        or deltas["average_precision"] >= active_policy.min_ap_gain
    )
    if not discrimination_improved:
        reasons.append("challenger does not materially improve ROC-AUC or average precision")

    if deltas["brier_score"] > active_policy.max_brier_regression:
        reasons.append("calibration regression exceeds Brier tolerance")

    if deltas["expected_cost_per_application"] > -active_policy.min_cost_improvement:
        reasons.append("expected business cost does not improve enough")

    if deltas["latency_ratio"] > active_policy.max_latency_ratio:
        reasons.append("serving latency regression exceeds policy")

    if challenger.psi_score > active_policy.max_psi:
        reasons.append("challenger evaluation population is too unstable")

    if deltas["review_rate"] > active_policy.max_review_rate_increase:
        reasons.append("challenger exceeds human-review capacity tolerance")

    return PromotionDecision(promote=not reasons, reasons=tuple(reasons), deltas=deltas)


def markdown_report(
    champion: ModelSnapshot,
    challenger: ModelSnapshot,
    decision: PromotionDecision,
) -> str:
    status = "PROMOTE" if decision.promote else "HOLD"
    lines = [
        f"# Champion / Challenger Decision — {status}",
        "",
        f"Champion: `{champion.name}:{champion.version}`",
        f"Challenger: `{challenger.name}:{challenger.version}`",
        "",
        "## Metric deltas",
        "",
        "| Metric | Delta |",
        "|---|---:|",
    ]
    for name, value in decision.deltas.items():
        lines.append(f"| {name} | {value:.6f} |")

    lines.extend(["", "## Decision reasons", ""])
    if decision.reasons:
        lines.extend(f"- {reason}" for reason in decision.reasons)
    else:
        lines.append("- All promotion gates passed.")
    return "\n".join(lines)


if __name__ == "__main__":
    champion = ModelSnapshot("hist_gbdt", "1.4.2", 0.781, 0.364, 0.109, 61.2, 27.0, 0.07, 0.22)
    challenger = ModelSnapshot("xgboost", "2.0.0-rc1", 0.789, 0.381, 0.108, 57.8, 31.0, 0.09, 0.23)
    result = compare(champion, challenger)
    print(markdown_report(champion, challenger, result))
