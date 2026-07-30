from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .citations import citation_precision_recall, field_pairs, set_f1, unsupported_claim_ids
from .contracts import ContractError, parse_answer, validate_answer


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    schema: float
    support: float
    answer_state: float
    fields: float
    citations: float
    abstention: float
    concision: float
    gated: bool


WEIGHTS = {
    "answer_state": 0.25,
    "fields": 0.30,
    "citations": 0.25,
    "abstention": 0.15,
    "concision": 0.05,
}


def score_completion(
    completion: Any,
    gold: dict[str, Any],
    contexts: list[dict[str, Any]],
) -> RewardBreakdown:
    try:
        prediction = parse_answer(completion, contexts).value
        gold_answer = validate_answer(gold, contexts)
    except (ContractError, KeyError, TypeError, ValueError):
        return RewardBreakdown(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)
    if unsupported_claim_ids(prediction, contexts):
        return RewardBreakdown(-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True)

    answer_state = 1.0 if prediction["answer_state"] == gold_answer["answer_state"] else 0.0
    fields = set_f1(field_pairs(prediction), field_pairs(gold_answer))
    citation_precision, citation_recall = citation_precision_recall(
        prediction["citations"], gold_answer["citations"]
    )
    citations = (
        0.0
        if citation_precision + citation_recall == 0
        else 2 * citation_precision * citation_recall / (citation_precision + citation_recall)
    )
    predicted_abstains = prediction["answer_state"] == "not_found"
    gold_abstains = gold_answer["answer_state"] == "not_found"
    abstention = 1.0 if predicted_abstains == gold_abstains else 0.0
    concision = 1.0 if len(prediction["answer"]) <= 160 else 0.0
    total = (
        WEIGHTS["answer_state"] * answer_state
        + WEIGHTS["fields"] * fields
        + WEIGHTS["citations"] * citations
        + WEIGHTS["abstention"] * abstention
        + WEIGHTS["concision"] * concision
    )
    return RewardBreakdown(total, 1.0, 1.0, answer_state, fields, citations, abstention, concision, False)


def product_rag_reward(
    completions: list[Any],
    gold_json: list[str],
    context_json: list[str],
    **_: Any,
) -> list[float]:
    if not (len(completions) == len(gold_json) == len(context_json)):
        raise ValueError("reward inputs must have equal lengths")
    return [
        score_completion(completion, json.loads(gold), json.loads(contexts)).total
        for completion, gold, contexts in zip(completions, gold_json, context_json)
    ]
