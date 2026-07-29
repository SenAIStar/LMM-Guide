from __future__ import annotations

from dataclasses import dataclass

from .contracts import AuditResult
from .policy import PolicyEngine


@dataclass(frozen=True)
class GenerationRequest:
    asset_id: str
    policy_version: str
    prompt: str
    prohibited_elements: tuple[str, ...]
    review_required: bool


def build_generation_request(result: AuditResult, policy: PolicyEngine) -> GenerationRequest:
    verdict = policy.evaluate(result)
    if not verdict.allowed or not verdict.generation_allowed:
        raise ValueError(f"generation blocked: {','.join(verdict.reasons) or result.decision}")
    if result.generation_brief is None:
        raise ValueError("approved audit result does not contain a generation brief")
    brief = result.generation_brief
    prompt_parts = [brief.objective]
    if brief.must_include:
        prompt_parts.append("Must include: " + ", ".join(brief.must_include))
    return GenerationRequest(
        asset_id=result.asset_id,
        policy_version=result.policy_version,
        prompt="\n".join(prompt_parts),
        prohibited_elements=brief.must_avoid,
        review_required=brief.human_review_required or verdict.force_human_review,
    )

