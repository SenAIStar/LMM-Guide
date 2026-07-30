from __future__ import annotations

import math
from typing import Any

from .catalog import is_eligible
from .citations import citation_precision_recall, field_pairs, unsupported_claim_ids
from .contracts import ContractError, validate_answer
from .retrieval import HybridRetriever


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _dcg(relevances: list[int]) -> float:
    return sum((2**relevance - 1) / math.log2(index + 2) for index, relevance in enumerate(relevances))


def evaluate_retrieval(
    queries: list[dict[str, Any]], retriever: HybridRetriever, k: int = 10
) -> dict[str, float]:
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    leaks = total_hits = 0
    for query in queries:
        hits = retriever.search(query, final_limit=k)
        ids = [hit.chunk["chunk_id"] for hit in hits]
        relevant = set(query["relevant_chunk_ids"])
        if not relevant:
            empty_result_score = 1.0 if not ids else 0.0
            recalls.append(empty_result_score)
            reciprocal_ranks.append(empty_result_score)
            ndcgs.append(empty_result_score)
            for hit in hits:
                total_hits += 1
                if not is_eligible(hit.chunk, query):
                    leaks += 1
            continue
        recalls.append(len(relevant.intersection(ids)) / len(relevant))
        first_rank = next((index for index, chunk_id in enumerate(ids, start=1) if chunk_id in relevant), None)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        relevances = [1 if chunk_id in relevant else 0 for chunk_id in ids]
        ideal = [1] * min(len(relevant), k)
        ndcgs.append(safe_divide(_dcg(relevances), _dcg(ideal)))
        for hit in hits:
            total_hits += 1
            if not is_eligible(hit.chunk, query):
                leaks += 1
    count = len(queries)
    return {
        f"recall_at_{k}": safe_divide(sum(recalls), count),
        f"mrr_at_{k}": safe_divide(sum(reciprocal_ranks), count),
        f"ndcg_at_{k}": safe_divide(sum(ndcgs), count),
        "acl_leak_rate": safe_divide(leaks, total_hits),
    }


def evaluate_generation(
    queries: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, float]:
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    schema_valid = field_scores = grounded_claims = total_claims = 0.0
    citation_precision_total = citation_recall_total = 0.0
    abstain_tp = abstain_fp = abstain_fn = 0
    for query in queries:
        gold_contexts = [chunk_map[chunk_id] for chunk_id in query["relevant_chunk_ids"]]
        gold = validate_answer(query["gold_output"], gold_contexts)
        envelope = predictions.get(query["query_id"], {})
        raw_output = envelope.get("output")
        predicted_context_ids = envelope.get("context_ids", query["relevant_chunk_ids"])
        contexts = [chunk_map[chunk_id] for chunk_id in predicted_context_ids if chunk_id in chunk_map]
        try:
            prediction = validate_answer(raw_output, contexts)
            schema_valid += 1.0
        except (ContractError, TypeError, KeyError):
            if gold["answer_state"] == "not_found":
                abstain_fn += 1
            continue
        field_scores += 1.0 if field_pairs(prediction) == field_pairs(gold) else 0.0
        unsupported = set(unsupported_claim_ids(prediction, contexts))
        total_claims += len(prediction["claims"])
        grounded_claims += len(prediction["claims"]) - len(unsupported)
        precision, recall = citation_precision_recall(prediction["citations"], gold["citations"])
        citation_precision_total += precision
        citation_recall_total += recall
        predicted_abstain = prediction["answer_state"] == "not_found"
        gold_abstain = gold["answer_state"] == "not_found"
        if predicted_abstain and gold_abstain:
            abstain_tp += 1
        elif predicted_abstain:
            abstain_fp += 1
        elif gold_abstain:
            abstain_fn += 1
    count = len(queries)
    abstain_precision = safe_divide(abstain_tp, abstain_tp + abstain_fp)
    abstain_recall = safe_divide(abstain_tp, abstain_tp + abstain_fn)
    return {
        "schema_valid_rate": safe_divide(schema_valid, count),
        "field_exact_match": safe_divide(field_scores, count),
        "grounded_claim_rate": safe_divide(grounded_claims, total_claims),
        "citation_precision": safe_divide(citation_precision_total, count),
        "citation_recall": safe_divide(citation_recall_total, count),
        "abstention_f1": safe_divide(
            2 * abstain_precision * abstain_recall, abstain_precision + abstain_recall
        ),
    }
