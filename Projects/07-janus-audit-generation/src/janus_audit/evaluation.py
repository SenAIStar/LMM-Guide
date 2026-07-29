from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .contracts import AuditResult
from .policy import PolicyEngine


@dataclass(frozen=True)
class EvaluationSummary:
    samples: int
    json_valid_rate: float
    audit_macro_f1: float
    evidence_precision: float
    selective_error_rate: float
    policy_violation_rate: float


def macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(e == label and p == label for e, p in zip(expected, predicted))
        fp = sum(e != label and p == label for e, p in zip(expected, predicted))
        fn = sum(e == label and p != label for e, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evidence_precision(expected: AuditResult, predicted: AuditResult) -> tuple[int, int]:
    gold = {(item.kind, item.source, item.quote, item.bbox) for item in expected.evidence}
    guesses = {(item.kind, item.source, item.quote, item.bbox) for item in predicted.evidence}
    return len(gold & guesses), len(guesses)


def evaluate(
    gold: Sequence[AuditResult],
    raw_predictions: Sequence[dict],
    policy: PolicyEngine,
) -> EvaluationSummary:
    if len(gold) != len(raw_predictions):
        raise ValueError("gold and predictions must have equal length")
    valid = 0
    decisions: list[str] = []
    expected_decisions: list[str] = []
    evidence_hits = 0
    evidence_guesses = 0
    non_abstain = 0
    non_abstain_errors = 0
    violations = 0
    for expected, payload in zip(gold, raw_predictions):
        expected_decisions.append(expected.decision)
        try:
            predicted = AuditResult.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            decisions.append("__invalid__")
            violations += 1
            continue
        valid += 1
        decisions.append(predicted.decision)
        if not policy.evaluate(predicted).allowed:
            violations += 1
        hits, guesses = evidence_precision(expected, predicted)
        evidence_hits += hits
        evidence_guesses += guesses
        if predicted.decision != "abstain":
            non_abstain += 1
            non_abstain_errors += int(predicted.decision != expected.decision)
    count = len(gold)
    return EvaluationSummary(
        samples=count,
        json_valid_rate=valid / count if count else 0.0,
        audit_macro_f1=macro_f1(expected_decisions, decisions),
        evidence_precision=evidence_hits / evidence_guesses if evidence_guesses else 1.0,
        selective_error_rate=non_abstain_errors / non_abstain if non_abstain else 0.0,
        policy_violation_rate=violations / count if count else 0.0,
    )


def relative_regression(baseline: float, candidate: float, higher_is_better: bool = True) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    signed_drop = baseline - candidate if higher_is_better else candidate - baseline
    return max(0.0, signed_drop / abs(baseline))


def confusion_counts(expected: Iterable[str], predicted: Iterable[str]) -> dict[str, int]:
    counter = Counter(f"{gold}->{guess}" for gold, guess in zip(expected, predicted))
    return dict(sorted(counter.items()))

