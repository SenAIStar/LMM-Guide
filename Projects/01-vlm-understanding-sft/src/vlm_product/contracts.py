import json
import re
from typing import Any


ALLOWED_DECISIONS = {"accept", "review", "reject"}
ATTRIBUTE_KEYS = {"color", "material"}
REQUIRED_OUTPUT_KEYS = {
    "schema_version",
    "product_type",
    "attributes",
    "visible_text",
    "evidence",
    "decision",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_assistant_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("unterminated fenced JSON")
        candidate = "\n".join(lines[1:-1]).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("assistant output must be a JSON object")
    return value


def _validate_string_list(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{path} must be a list of non-empty strings")
        return
    normalized = [item.strip().lower() for item in value]
    if len(normalized) != len(set(normalized)):
        errors.append(f"{path} contains duplicate values")


def validate_prediction(value: dict[str, Any], media_count: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_OUTPUT_KEYS - value.keys()
    extra = value.keys() - REQUIRED_OUTPUT_KEYS
    if missing:
        errors.append(f"missing output keys: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected output keys: {sorted(extra)}")
    if value.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    product_type = value.get("product_type")
    if not isinstance(product_type, str) or not product_type.strip():
        errors.append("product_type must be a non-empty string")
    decision = value.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append("decision is invalid")

    attributes = value.get("attributes")
    if not isinstance(attributes, dict):
        errors.append("attributes must be an object")
        attributes = {}
    else:
        missing_attributes = ATTRIBUTE_KEYS - attributes.keys()
        extra_attributes = attributes.keys() - ATTRIBUTE_KEYS
        if missing_attributes:
            errors.append(f"missing attribute keys: {sorted(missing_attributes)}")
        if extra_attributes:
            errors.append(f"unexpected attribute keys: {sorted(extra_attributes)}")
        for name in sorted(ATTRIBUTE_KEYS):
            _validate_string_list(attributes.get(name), f"attributes.{name}", errors)

    _validate_string_list(value.get("visible_text"), "visible_text", errors)
    evidence = value.get("evidence")
    evidence_fields: set[str] = set()
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
        evidence = []
    else:
        allowed_fields = {"product_type", "visible_text"} | {f"attributes.{name}" for name in ATTRIBUTE_KEYS}
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"evidence[{index}] must be an object")
                continue
            if set(item) != {"field", "media_index", "support"}:
                errors.append(f"evidence[{index}] must contain only field, media_index and support")
            field = item.get("field")
            if field not in allowed_fields:
                errors.append(f"evidence[{index}].field is invalid")
            else:
                evidence_fields.add(field)
            media_index = item.get("media_index")
            if not isinstance(media_index, int) or not 0 <= media_index < media_count:
                errors.append(f"evidence[{index}].media_index is invalid")
            if item.get("support") != "image_level":
                errors.append(f"evidence[{index}].support must be image_level")

    required_evidence: set[str] = set()
    if isinstance(product_type, str) and product_type.strip().lower() != "unknown":
        required_evidence.add("product_type")
    for name in ATTRIBUTE_KEYS:
        if isinstance(attributes.get(name), list) and attributes[name]:
            required_evidence.add(f"attributes.{name}")
    if isinstance(value.get("visible_text"), list) and value["visible_text"]:
        required_evidence.add("visible_text")
    missing_evidence = sorted(required_evidence - evidence_fields)
    if missing_evidence:
        errors.append(f"missing evidence for: {missing_evidence}")

    has_unknown = (
        isinstance(product_type, str)
        and product_type.strip().lower() == "unknown"
    ) or any(isinstance(attributes.get(name), list) and not attributes[name] for name in ATTRIBUTE_KEYS)
    if decision == "accept" and has_unknown:
        errors.append("accept is not allowed when a required field is unknown")
    return errors


def validate_training_record(record: dict[str, Any], check_output: bool = True) -> list[str]:
    errors: list[str] = []
    for key in ("sample_id", "group_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"{key} must be a non-empty string")
    if record.get("split") not in {"train", "eval", "test"}:
        errors.append("split must be train, eval or test")

    images = record.get("images")
    if not isinstance(images, list) or not images or not all(isinstance(item, str) and item.strip() for item in images):
        errors.append("images must be a non-empty string list")
        images = []
    hashes = record.get("media_sha256")
    if not isinstance(hashes, list) or len(hashes) != len(images):
        errors.append("media_sha256 must match images length")
    elif not all(isinstance(item, str) and SHA256_RE.fullmatch(item) for item in hashes):
        errors.append("media_sha256 must contain lowercase SHA-256 values")

    source = record.get("source")
    required_source = {"dataset", "snapshot_id", "license_id", "source_uri"}
    if not isinstance(source, dict) or not required_source.issubset(source):
        errors.append("source must include dataset, snapshot_id, license_id and source_uri")
    if record.get("review_required") is not False:
        errors.append("training-ready records must set review_required to false")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant turns")
        return errors
    image_tags = sum(str(turn.get("content", "")).count("<image>") for turn in messages)
    if image_tags != len(images):
        errors.append(f"image tag count {image_tags} does not match images count {len(images)}")
    for index, turn in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if turn.get("role") != expected_role:
            errors.append(f"messages[{index}].role must be {expected_role}")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            errors.append(f"messages[{index}].content must be a non-empty string")
    if messages[-1].get("role") != "assistant":
        errors.append("the final message must be assistant")
    if any("<image>" in str(turn.get("content", "")) for turn in messages if turn.get("role") == "assistant"):
        errors.append("assistant targets must not contain <image>")
    if check_output and messages[-1].get("role") == "assistant":
        try:
            output = parse_assistant_json(str(messages[-1].get("content", "")))
            errors.extend(validate_prediction(output, len(images)))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"assistant output is invalid: {exc}")
    return errors
