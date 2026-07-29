from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import AuditResult
from .policy import PolicyEngine


@dataclass(frozen=True)
class RewardBreakdown:
    schema: float
    decision: float
    risk_labels: float
    evidence: float
    brief: float
    calibration: float
    hard_policy_violation: bool
    total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    true_positive = len(predicted & expected)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted)
    recall = true_positive / len(expected)
    return 2 * precision * recall / (precision + recall)


def evidence_key(item: Any) -> tuple[Any, ...]:
    bbox = tuple(round(value, 2) for value in item.bbox) if item.bbox else None
    return item.kind, item.source, item.quote, bbox


def score_candidate(payload: dict[str, Any], gold: AuditResult, policy: PolicyEngine) -> RewardBreakdown:
    try:
        candidate = AuditResult.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, -1.0)

    verdict = policy.evaluate(candidate)
    if not verdict.allowed:
        return RewardBreakdown(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, -1.0)

    decision = float(candidate.decision == gold.decision)
    labels = set_f1(set(candidate.risk_labels), set(gold.risk_labels))
    evidence = set_f1(
        {str(evidence_key(item)) for item in candidate.evidence},
        {str(evidence_key(item)) for item in gold.evidence},
    )
    if gold.generation_brief is None:
        brief = float(candidate.generation_brief is None)
    else:
        brief = float(candidate.generation_brief is not None)
    expected_review = gold.review_required
    calibration = float(candidate.review_required == expected_review)
    total = 0.30 * decision + 0.25 * labels + 0.20 * evidence + 0.15 * brief + 0.10 * calibration
    return RewardBreakdown(1.0, decision, labels, evidence, brief, calibration, False, total)

