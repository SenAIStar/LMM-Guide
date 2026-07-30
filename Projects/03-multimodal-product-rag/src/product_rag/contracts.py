from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


OUTPUT_KEYS = {"schema_version", "answer_state", "answer", "claims", "citations"}
CLAIM_KEYS = {"claim_id", "text", "field", "value", "citation_ids"}
CITATION_KEYS = {"chunk_id", "product_id", "source_revision", "media_index"}
ANSWER_STATES = {"grounded", "not_found", "ambiguous"}


class ContractError(ValueError):
    """Raised when an answer violates the JSON contract."""


@dataclass(frozen=True)
class ParsedAnswer:
    value: dict[str, Any]
    source_text: str


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and isinstance(completion.get("content"), str):
        return completion["content"]
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ContractError("unsupported completion type")


def parse_first_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ContractError("JSON object not found")
    if text[:start].strip():
        raise ContractError("text before JSON object")
    try:
        value, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg}") from exc
    if text[start + end :].strip():
        raise ContractError("trailing text after JSON object")
    if not isinstance(value, dict):
        raise ContractError("answer must be a JSON object")
    return value


def _unique_strings(value: Any, name: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{name} must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ContractError(f"{name} must not be empty")
    if len(value) != len(set(value)):
        raise ContractError(f"{name} must not contain duplicates")
    return value


def validate_answer(
    value: dict[str, Any], contexts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    if set(value) != OUTPUT_KEYS:
        raise ContractError(
            f"output keys mismatch; missing={sorted(OUTPUT_KEYS - set(value))}, "
            f"extra={sorted(set(value) - OUTPUT_KEYS)}"
        )
    if value["schema_version"] != "product-qa.v1":
        raise ContractError("unsupported schema_version")
    state = value["answer_state"]
    if state not in ANSWER_STATES:
        raise ContractError("invalid answer_state")
    answer = value["answer"]
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 500:
        raise ContractError("answer must be a non-empty string no longer than 500 characters")
    claims = value["claims"]
    citations = value["citations"]
    if not isinstance(claims, list) or not isinstance(citations, list):
        raise ContractError("claims and citations must be lists")

    normalized_citations: list[dict[str, Any]] = []
    citation_ids: list[str] = []
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != CITATION_KEYS:
            raise ContractError("citation keys mismatch")
        for name in ("chunk_id", "product_id", "source_revision"):
            if not isinstance(citation[name], str) or not citation[name]:
                raise ContractError(f"citation {name} must be a non-empty string")
        media_index = citation["media_index"]
        if media_index is not None and (
            isinstance(media_index, bool) or not isinstance(media_index, int) or media_index < 0
        ):
            raise ContractError("citation media_index must be a non-negative integer or null")
        citation_ids.append(citation["chunk_id"])
        normalized_citations.append(dict(citation))
    if len(citation_ids) != len(set(citation_ids)):
        raise ContractError("citations must be unique by chunk_id")

    normalized_claims: list[dict[str, Any]] = []
    claim_ids: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != CLAIM_KEYS:
            raise ContractError("claim keys mismatch")
        if not isinstance(claim["claim_id"], str) or not claim["claim_id"]:
            raise ContractError("claim_id must be a non-empty string")
        if not isinstance(claim["text"], str) or not claim["text"] or len(claim["text"]) > 200:
            raise ContractError("claim text must be a non-empty string no longer than 200 characters")
        field = claim["field"]
        if field is not None and (not isinstance(field, str) or not field):
            raise ContractError("claim field must be a non-empty string or null")
        if not isinstance(claim["value"], (str, int, float, bool, type(None))):
            raise ContractError("claim value must be a JSON scalar")
        refs = _unique_strings(claim["citation_ids"], "claim citation_ids", allow_empty=False)
        if not set(refs).issubset(set(citation_ids)):
            raise ContractError("claim references an undeclared citation")
        claim_ids.append(claim["claim_id"])
        normalized_claims.append({**claim, "citation_ids": refs})
    if len(claim_ids) != len(set(claim_ids)):
        raise ContractError("claim_id values must be unique")

    if state == "grounded" and (not claims or not citations):
        raise ContractError("grounded answers require claims and citations")
    if state == "not_found" and (claims or citations):
        raise ContractError("not_found answers must not contain claims or citations")

    if contexts is not None:
        context_map = {context["chunk_id"]: context for context in contexts}
        for citation in normalized_citations:
            context = context_map.get(citation["chunk_id"])
            if context is None:
                raise ContractError("citation is outside retrieved context")
            if citation["product_id"] != context["product_id"]:
                raise ContractError("citation product_id does not match retrieved context")
            if citation["source_revision"] != context["source_revision"]:
                raise ContractError("citation source_revision does not match retrieved context")
            media_index = citation["media_index"]
            if media_index is not None and media_index >= len(context.get("media", [])):
                raise ContractError("citation media_index is outside retrieved context")

    return {
        **value,
        "answer": answer.strip(),
        "claims": normalized_claims,
        "citations": normalized_citations,
    }


def parse_answer(completion: Any, contexts: list[dict[str, Any]] | None = None) -> ParsedAnswer:
    source_text = completion_text(completion)
    return ParsedAnswer(validate_answer(parse_first_json_object(source_text), contexts), source_text)
