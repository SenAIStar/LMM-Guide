from typing import Any


def structured_reward(output: dict[str, Any], required_fields: set[str]) -> float:
    if not required_fields:
        raise ValueError("required_fields cannot be empty")
    present = sum(field in output and output[field] not in (None, "") for field in required_fields)
    return present / len(required_fields)


def product_audit_reward(output: dict[str, Any], reference: dict[str, Any]) -> float:
    """Deterministic reward used before any learned or judge-model reward."""
    required = {"decision", "risk_type", "evidence", "confidence"}
    schema = structured_reward(output, required)
    decision = float(output.get("decision") == reference.get("decision"))
    risk_type = float(output.get("risk_type") == reference.get("risk_type"))
    evidence = output.get("evidence")
    evidence_score = float(isinstance(evidence, list) and bool(evidence))
    confidence = output.get("confidence")
    calibrated = float(isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0)
    return round(0.30 * schema + 0.30 * decision + 0.20 * risk_type + 0.10 * evidence_score + 0.10 * calibrated, 6)

