from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contracts import ContractError, parse_audit_output, validate_audit_output
from .policy import evaluate_hard_policy


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    schema: float
    hard_policy: float
    decision: float
    risk_f1: float
    evidence: float
    policy_refs: float
    explanation: float
    gated: bool


WEIGHTS = {
    "decision": 0.35,
    "risk_f1": 0.30,
    "evidence": 0.20,
    "policy_refs": 0.10,
    "explanation": 0.05,
}


def _set_f1(predicted: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 1.0 if left is None and right is None else 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _evidence_score(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    matched_gold: set[int] = set()
    matches = 0.0
    for item in predicted:
        candidates: list[tuple[float, int]] = []
        for index, target in enumerate(gold):
            if index in matched_gold:
                continue
            if item["risk_code"] != target["risk_code"] or item["media_index"] != target["media_index"]:
                continue
            if item["support"] != target["support"]:
                continue
            candidates.append((bbox_iou(item["bbox"], target["bbox"]), index))
        if candidates:
            score, index = max(candidates)
            if score >= 0.5:
                matched_gold.add(index)
                matches += 1.0
    precision = matches / len(predicted)
    recall = matches / len(gold)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _decision_score(prediction: str, gold: str) -> float:
    if prediction == gold:
        return 1.0
    # Missing a reject is more costly than sending a safe item to review.
    costs = {
        ("pass", "reject"): -1.0,
        ("review", "reject"): -0.5,
        ("reject", "pass"): -0.75,
        ("review", "pass"): 0.25,
        ("pass", "review"): 0.0,
        ("reject", "review"): -0.25,
    }
    return costs.get((prediction, gold), -0.5)


def _violates_policy(prediction: dict[str, Any], context: dict[str, Any], policy: dict[str, Any]) -> bool:
    result = evaluate_hard_policy(context, policy)
    if result.decision == "reject" and prediction["decision"] != "reject":
        return True
    required_rules = {hit.rule_id for hit in result.hits if hit.decision == "reject"}
    return not required_rules.issubset(set(prediction["policy_refs"]))


def score_completion(
    completion: Any,
    ground_truth: dict[str, Any],
    policy: dict[str, Any],
    context: dict[str, Any],
) -> RewardBreakdown:
    try:
        prediction = parse_audit_output(completion, image_count=context.get("image_count")).value
        gold = validate_audit_output(ground_truth, image_count=context.get("image_count"))
    except (ContractError, KeyError, TypeError, ValueError):
        return RewardBreakdown(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    if _violates_policy(prediction, context, policy):
        return RewardBreakdown(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    decision = _decision_score(prediction["decision"], gold["decision"])
    risk_f1 = _set_f1(prediction["risk_codes"], gold["risk_codes"])
    evidence = _evidence_score(prediction["evidence"], gold["evidence"])
    policy_refs = _set_f1(prediction["policy_refs"], gold["policy_refs"])
    explanation = 1.0 if 1 <= len(prediction["explanation"]) <= 120 else 0.0
    total = (
        WEIGHTS["decision"] * decision
        + WEIGHTS["risk_f1"] * risk_f1
        + WEIGHTS["evidence"] * evidence
        + WEIGHTS["policy_refs"] * policy_refs
        + WEIGHTS["explanation"] * explanation
    )
    return RewardBreakdown(total, 1.0, 1.0, decision, risk_f1, evidence, policy_refs, explanation, False)


def product_audit_reward(
    completions: list[Any],
    ground_truth: list[str],
    policy_json: list[str],
    context_json: list[str],
    **_: Any,
) -> list[float]:
    """TRL GRPO reward adapter; dataset columns are passed by name."""
    if not (len(completions) == len(ground_truth) == len(policy_json) == len(context_json)):
        raise ValueError("reward inputs must have equal lengths")
    scores: list[float] = []
    for completion, gold_text, policy_text, context_text in zip(
        completions, ground_truth, policy_json, context_json
    ):
        result = score_completion(
            completion=completion,
            ground_truth=json.loads(gold_text),
            policy=json.loads(policy_text),
            context=json.loads(context_text),
        )
        scores.append(result.total)
    return scores

