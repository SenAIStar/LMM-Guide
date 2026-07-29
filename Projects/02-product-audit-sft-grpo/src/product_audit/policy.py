from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyHit:
    rule_id: str
    risk_code: str
    severity: str
    decision: str


@dataclass(frozen=True)
class PolicyResult:
    policy_version: str
    decision: str
    hits: tuple[PolicyHit, ...]


def load_policy(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        policy = json.load(stream)
    required = {"schema_version", "policy_version", "effective_at", "rules"}
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("policy keys mismatch")
    if policy["schema_version"] != "audit-policy.v1" or not isinstance(policy["rules"], list):
        raise ValueError("unsupported policy schema")
    ids = [rule.get("id") for rule in policy["rules"] if isinstance(rule, dict)]
    if len(ids) != len(policy["rules"]) or len(ids) != len(set(ids)):
        raise ValueError("policy rule ids must be present and unique")
    return policy


def _get_path(context: dict[str, Any], dotted_path: str) -> Any:
    current: Any = context
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "equals":
        return actual == expected
    if operator == "in":
        return isinstance(expected, list) and actual in expected
    if operator == "contains_any":
        if not isinstance(expected, list):
            return False
        values = actual if isinstance(actual, list) else [actual]
        normalized = [str(value).casefold() for value in values if value is not None]
        return any(str(needle).casefold() in value for needle in expected for value in normalized)
    raise ValueError(f"unsupported policy operator: {operator}")


def evaluate_hard_policy(context: dict[str, Any], policy: dict[str, Any]) -> PolicyResult:
    hits: list[PolicyHit] = []
    for rule in policy["rules"]:
        if _matches(_get_path(context, rule["field"]), rule["operator"], rule["value"]):
            hits.append(
                PolicyHit(
                    rule_id=rule["id"],
                    risk_code=rule["risk_code"],
                    severity=rule["severity"],
                    decision=rule["decision"],
                )
            )
    decisions = {hit.decision for hit in hits}
    decision = "reject" if "reject" in decisions else "review" if "review" in decisions else "pass"
    return PolicyResult(policy_version=policy["policy_version"], decision=decision, hits=tuple(hits))

