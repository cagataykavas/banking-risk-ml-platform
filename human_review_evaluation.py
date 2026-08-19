from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CaseOutcome:
    model_score: float
    model_decision: int
    human_decision: int
    final_outcome: int
    review_seconds: float


def evaluate_human_agent_system(cases: list[CaseOutcome]) -> dict[str, float]:
    if not cases:
        raise ValueError("cases cannot be empty")

    model = np.array([c.model_decision for c in cases])
    human = np.array([c.human_decision for c in cases])
    truth = np.array([c.final_outcome for c in cases])
    review_time = np.array([c.review_seconds for c in cases], dtype=float)

    model_accuracy = float(np.mean(model == truth))
    human_accuracy = float(np.mean(human == truth))
    override_rate = float(np.mean(model != human))
    useful_override_rate = float(np.mean((model != human) & (human == truth)))
    harmful_override_rate = float(np.mean((model != human) & (model == truth)))

    return {
        "model_accuracy": model_accuracy,
        "human_accuracy": human_accuracy,
        "override_rate": override_rate,
        "useful_override_rate": useful_override_rate,
        "harmful_override_rate": harmful_override_rate,
        "median_review_seconds": float(np.median(review_time)),
        "p95_review_seconds": float(np.quantile(review_time, 0.95)),
    }


def automation_bias_signal(cases: list[CaseOutcome]) -> float:
    """Fraction of model errors that the human reviewer failed to correct."""
    errors = [c for c in cases if c.model_decision != c.final_outcome]
    if not errors:
        return 0.0
    missed = sum(c.human_decision == c.model_decision for c in errors)
    return missed / len(errors)


if __name__ == "__main__":
    demo = [
        CaseOutcome(0.91, 1, 1, 1, 18),
        CaseOutcome(0.63, 1, 0, 0, 44),
        CaseOutcome(0.55, 1, 1, 0, 39),
        CaseOutcome(0.22, 0, 0, 0, 14),
        CaseOutcome(0.49, 0, 1, 1, 52),
    ]
    print(evaluate_human_agent_system(demo))
    print("automation_bias_signal:", automation_bias_signal(demo))
