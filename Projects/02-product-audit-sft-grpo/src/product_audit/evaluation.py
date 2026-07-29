from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .contracts import ContractError, validate_audit_output
from .policy import evaluate_hard_policy


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = 10
) -> float | None:
    if not confidences:
        return None
    if len(confidences) != len(correct) or bins <= 0:
        raise ValueError("invalid calibration inputs")
    total = len(confidences)
    error = 0.0
    for bin_index in range(bins):
        lower, upper = bin_index / bins, (bin_index + 1) / bins
        indices = [
            index
            for index, value in enumerate(confidences)
            if lower <= value <= upper and (bin_index == bins - 1 or value < upper)
        ]
        if not indices:
            continue
        accuracy = sum(1.0 for index in indices if correct[index]) / len(indices)
        confidence = sum(confidences[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(accuracy - confidence)
    return error


def evaluate(
    gold_records: Iterable[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    policy: dict[str, Any],
    high_risk_codes: set[str],
) -> dict[str, Any]:
    records = list(gold_records)
    schema_valid = 0
    risk_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    high_risk_total = high_risk_missed = safe_total = safe_auto_rejected = 0
    manual_review = policy_violations = 0
    confidences: list[float] = []
    confidence_correct: list[bool] = []
    for record in records:
        gold = validate_audit_output(record["ground_truth"], image_count=len(record["media"]))
        envelope = predictions.get(record["sample_id"], {})
        raw_prediction = envelope.get("output")
        try:
            prediction = validate_audit_output(raw_prediction, image_count=len(record["media"]))
            schema_valid += 1
        except (ContractError, TypeError, KeyError):
            prediction = {
                "decision": "review",
                "risk_codes": [],
                "evidence": [],
                "policy_refs": [],
                "explanation": "invalid output",
            }
        gold_risks, predicted_risks = set(gold["risk_codes"]), set(prediction["risk_codes"])
        for risk in gold_risks | predicted_risks:
            if risk in gold_risks and risk in predicted_risks:
                risk_stats[risk]["tp"] += 1
            elif risk in predicted_risks:
                risk_stats[risk]["fp"] += 1
            else:
                risk_stats[risk]["fn"] += 1
        if gold_risks & high_risk_codes:
            high_risk_total += 1
            if prediction["decision"] != "reject":
                high_risk_missed += 1
        if gold["decision"] == "pass":
            safe_total += 1
            if prediction["decision"] == "reject":
                safe_auto_rejected += 1
        if prediction["decision"] == "review":
            manual_review += 1
        context = {
            "title": record["title"],
            "category": record["category"],
            "attributes": record["attributes"],
            "ocr_tokens": record["ocr_tokens"],
        }
        hard_result = evaluate_hard_policy(context, policy)
        if hard_result.decision == "reject" and prediction["decision"] != "reject":
            policy_violations += 1
        confidence = envelope.get("external_confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0 <= confidence <= 1:
            confidences.append(float(confidence))
            confidence_correct.append(prediction["decision"] == gold["decision"])
    f1_by_risk: dict[str, float] = {}
    for risk, values in sorted(risk_stats.items()):
        precision = safe_divide(values["tp"], values["tp"] + values["fp"])
        recall = safe_divide(values["tp"], values["tp"] + values["fn"])
        f1_by_risk[risk] = safe_divide(2 * precision * recall, precision + recall)
    macro_f1 = sum(f1_by_risk.values()) / len(f1_by_risk) if f1_by_risk else 0.0
    return {
        "sample_count": len(records),
        "schema_valid_rate": safe_divide(schema_valid, len(records)),
        "risk_macro_f1": macro_f1,
        "risk_f1": f1_by_risk,
        "high_risk_false_negative_rate": safe_divide(high_risk_missed, high_risk_total),
        "auto_reject_false_positive_rate": safe_divide(safe_auto_rejected, safe_total),
        "manual_review_rate": safe_divide(manual_review, len(records)),
        "policy_violation_rate": safe_divide(policy_violations, len(records)),
        "ece": expected_calibration_error(confidences, confidence_correct),
    }

