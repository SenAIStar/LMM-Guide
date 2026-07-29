from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


DECISIONS = {"pass", "reject", "review"}
SUPPORT_TYPES = {"image_level", "region"}
OUTPUT_KEYS = {
    "schema_version",
    "decision",
    "risk_codes",
    "evidence",
    "policy_refs",
    "explanation",
}
EVIDENCE_KEYS = {"risk_code", "media_index", "support", "bbox", "policy_rule_id"}


class ContractError(ValueError):
    """Raised when model output cannot be consumed safely."""


@dataclass(frozen=True)
class ParsedOutput:
    value: dict[str, Any]
    source_text: str


def completion_text(completion: Any) -> str:
    """Normalize TRL chat completions and plain strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
    raise ContractError("unsupported completion type")


def parse_first_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object while rejecting non-whitespace trailing text."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ContractError("JSON object not found")
    try:
        value, end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc.msg}") from exc
    if text[start + end :].strip():
        raise ContractError("trailing text after JSON object")
    if not isinstance(value, dict):
        raise ContractError("top-level output must be an object")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ContractError(f"{name} must not contain duplicates")
    return value


def _bbox(value: Any, support: str) -> list[float] | None:
    if support == "image_level":
        if value is not None:
            raise ContractError("image-level evidence must use a null bbox")
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ContractError("region evidence requires a four-number bbox")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise ContractError("bbox coordinates must be numbers")
    coords = [float(x) for x in value]
    x1, y1, x2, y2 = coords
    if not all(0.0 <= x <= 1.0 for x in coords) or not (x1 < x2 and y1 < y2):
        raise ContractError("bbox must be normalized and ordered")
    return coords


def validate_audit_output(value: dict[str, Any], image_count: int | None = None) -> dict[str, Any]:
    if set(value) != OUTPUT_KEYS:
        missing = sorted(OUTPUT_KEYS - set(value))
        extra = sorted(set(value) - OUTPUT_KEYS)
        raise ContractError(f"output keys mismatch; missing={missing}, extra={extra}")
    if value["schema_version"] != "audit-output.v1":
        raise ContractError("unsupported schema_version")
    if value["decision"] not in DECISIONS:
        raise ContractError("invalid decision")
    risks = _string_list(value["risk_codes"], "risk_codes")
    refs = _string_list(value["policy_refs"], "policy_refs")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or len(explanation) > 300:
        raise ContractError("explanation must be a string no longer than 300 characters")
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise ContractError("evidence must be a list")
    normalized_evidence: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != EVIDENCE_KEYS:
            raise ContractError("evidence keys mismatch")
        if item["risk_code"] not in risks:
            raise ContractError("evidence risk_code is not declared in risk_codes")
        media_index = item["media_index"]
        if isinstance(media_index, bool) or not isinstance(media_index, int) or media_index < 0:
            raise ContractError("media_index must be a non-negative integer")
        if image_count is not None and media_index >= image_count:
            raise ContractError("media_index is outside the request")
        support = item["support"]
        if support not in SUPPORT_TYPES:
            raise ContractError("invalid evidence support type")
        rule_id = item["policy_rule_id"]
        if rule_id is not None and (not isinstance(rule_id, str) or not rule_id):
            raise ContractError("policy_rule_id must be a non-empty string or null")
        if rule_id is not None and rule_id not in refs:
            raise ContractError("evidence policy_rule_id is not declared in policy_refs")
        normalized_evidence.append({**item, "bbox": _bbox(item["bbox"], support)})
    if value["decision"] == "pass" and (risks or evidence or refs):
        raise ContractError("pass output must not declare risks, evidence, or policy refs")
    if value["decision"] == "reject" and not risks:
        raise ContractError("reject output requires at least one risk code")
    return {**value, "risk_codes": risks, "policy_refs": refs, "evidence": normalized_evidence}


def parse_audit_output(completion: Any, image_count: int | None = None) -> ParsedOutput:
    text = completion_text(completion)
    value = validate_audit_output(parse_first_json_object(text), image_count=image_count)
    return ParsedOutput(value=value, source_text=text)

