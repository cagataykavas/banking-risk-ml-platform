from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Step(str, Enum):
    PROFILE = "profile"
    IDENTITY = "identity"
    SCREENING = "screening"
    RISK_REVIEW = "risk_review"
    HUMAN_REVIEW = "human_review"
    COMPLETE = "complete"
    DECLINED = "declined"


@dataclass
class OnboardingCase:
    case_id: str
    step: Step = Step.PROFILE
    risk_score: float | None = None
    confidence: float | None = None
    missing_information: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    def transition(self, target: Step, reason: str) -> None:
        self.history.append(f"{self.step.value} -> {target.value}: {reason}")
        self.step = target


def next_step(case: OnboardingCase) -> Step:
    if case.missing_information:
        return case.step

    if case.step is Step.PROFILE:
        return Step.IDENTITY
    if case.step is Step.IDENTITY:
        return Step.SCREENING
    if case.step is Step.SCREENING:
        return Step.RISK_REVIEW
    if case.step is Step.RISK_REVIEW:
        if case.risk_score is None or case.confidence is None:
            return Step.HUMAN_REVIEW
        if case.confidence < 0.90:
            return Step.HUMAN_REVIEW
        if case.risk_score >= 0.85:
            return Step.HUMAN_REVIEW
        return Step.COMPLETE
    if case.step is Step.HUMAN_REVIEW:
        return Step.HUMAN_REVIEW
    return case.step


def customer_message(case: OnboardingCase) -> str:
    if case.missing_information:
        return "We need a little more information: " + ", ".join(case.missing_information)
    messages = {
        Step.PROFILE: "Tell us about yourself to begin your application.",
        Step.IDENTITY: "Next, verify your identity.",
        Step.SCREENING: "We are checking the information you provided.",
        Step.RISK_REVIEW: "Your application is being assessed.",
        Step.HUMAN_REVIEW: "A specialist is reviewing your application.",
        Step.COMPLETE: "Your onboarding is complete.",
        Step.DECLINED: "We cannot complete the onboarding at this time.",
    }
    return messages[case.step]


if __name__ == "__main__":
    case = OnboardingCase("onb-1042")
    while case.step not in {Step.COMPLETE, Step.DECLINED, Step.HUMAN_REVIEW}:
        target = next_step(case)
        if target is Step.RISK_REVIEW:
            case.risk_score = 0.62
            case.confidence = 0.93
        case.transition(target, "workflow policy")
        print(case.step.value, "-", customer_message(case))
    print(case.history)
