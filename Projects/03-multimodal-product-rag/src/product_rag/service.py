from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .citations import unsupported_claim_ids
from .contracts import ContractError, parse_answer
from .retrieval import HybridRetriever


@dataclass(frozen=True)
class ServiceResult:
    action: str
    reason: str
    output: dict[str, Any] | None
    context_ids: tuple[str, ...]


def answer_query(
    query: dict[str, Any],
    retriever: HybridRetriever,
    generate: Callable[[dict[str, Any], list[dict[str, Any]]], Any],
    requires_dynamic_data: bool = False,
    dynamic_lookup: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> ServiceResult:
    if requires_dynamic_data:
        if dynamic_lookup is None or dynamic_lookup(query) is None:
            return ServiceResult("clarify", "dynamic_source_unavailable", None, ())
    hits = retriever.search(query)
    contexts = [hit.chunk for hit in hits]
    context_ids = tuple(context["chunk_id"] for context in contexts)
    if not contexts:
        return ServiceResult(
            "answer",
            "no_eligible_context",
            {
                "schema_version": "product-qa.v1",
                "answer_state": "not_found",
                "answer": "当前可访问资料中没有足够信息。",
                "claims": [],
                "citations": [],
            },
            (),
        )
    try:
        output = parse_answer(generate(query, contexts), contexts).value
    except (ContractError, TypeError, ValueError):
        return ServiceResult("manual_review", "invalid_model_output", None, context_ids)
    if unsupported_claim_ids(output, contexts):
        return ServiceResult("manual_review", "unsupported_claim", output, context_ids)
    action = "clarify" if output["answer_state"] == "ambiguous" else "answer"
    return ServiceResult(action, "validated_output", output, context_ids)
