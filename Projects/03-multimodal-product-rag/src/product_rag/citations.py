from __future__ import annotations

from typing import Any


def claim_supported(claim: dict[str, Any], context_map: dict[str, dict[str, Any]]) -> bool:
    field = claim["field"]
    if field is None:
        return False
    for chunk_id in claim["citation_ids"]:
        chunk = context_map.get(chunk_id)
        if chunk is not None and chunk.get("facts", {}).get(field) == claim["value"]:
            return True
    return False


def unsupported_claim_ids(answer: dict[str, Any], contexts: list[dict[str, Any]]) -> list[str]:
    context_map = {context["chunk_id"]: context for context in contexts}
    return [
        claim["claim_id"]
        for claim in answer["claims"]
        if not claim_supported(claim, context_map)
    ]


def citation_precision_recall(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> tuple[float, float]:
    predicted_ids = {citation["chunk_id"] for citation in predicted}
    gold_ids = {citation["chunk_id"] for citation in gold}
    if not predicted_ids and not gold_ids:
        return 1.0, 1.0
    true_positive = len(predicted_ids & gold_ids)
    precision = true_positive / len(predicted_ids) if predicted_ids else 0.0
    recall = true_positive / len(gold_ids) if gold_ids else 0.0
    return precision, recall


def field_pairs(answer: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (claim["field"], str(claim["value"]))
        for claim in answer["claims"]
        if claim["field"] is not None
    }


def set_f1(predicted: set[Any], gold: set[Any]) -> float:
    if not predicted and not gold:
        return 1.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
