from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .contracts import ContractError, parse_audit_output
from .policy import PolicyResult, evaluate_hard_policy


@dataclass(frozen=True)
class ServiceDecision:
    action: str
    reason: str
    policy_version: str
    output: dict[str, Any] | None
    hard_policy: dict[str, Any]


def _serialize_policy(result: PolicyResult) -> dict[str, Any]:
    return {
        "policy_version": result.policy_version,
        "decision": result.decision,
        "hits": [asdict(hit) for hit in result.hits],
    }


def review_product(
    request: dict[str, Any],
    policy: dict[str, Any],
    model_generate: Callable[[dict[str, Any]], Any],
    external_confidence: Callable[[dict[str, Any], dict[str, Any]], float],
    auto_action_threshold: float = 0.95,
) -> ServiceDecision:
    if not 0.0 <= auto_action_threshold <= 1.0:
        raise ValueError("auto_action_threshold must be between zero and one")
    hard_result = evaluate_hard_policy(request, policy)
    serialized = _serialize_policy(hard_result)
    if hard_result.decision == "reject":
        return ServiceDecision("reject", "hard_policy_reject", policy["policy_version"], None, serialized)
    try:
        prediction = parse_audit_output(
            model_generate(request), image_count=len(request.get("images", []))
        ).value
    except (ContractError, TypeError, ValueError):
        return ServiceDecision("manual_review", "invalid_model_output", policy["policy_version"], None, serialized)
    if hard_result.decision == "review" and prediction["decision"] == "pass":
        return ServiceDecision(
            "manual_review", "policy_model_conflict", policy["policy_version"], prediction, serialized
        )
    confidence = external_confidence(request, prediction)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        return ServiceDecision(
            "manual_review", "invalid_external_confidence", policy["policy_version"], prediction, serialized
        )
    if prediction["decision"] == "review" or confidence < auto_action_threshold:
        return ServiceDecision(
            "manual_review", "low_confidence_or_model_review", policy["policy_version"], prediction, serialized
        )
    return ServiceDecision(prediction["decision"], "calibrated_model_action", policy["policy_version"], prediction, serialized)

