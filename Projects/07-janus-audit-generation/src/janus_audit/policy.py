from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import AuditResult


@dataclass(frozen=True)
class PolicyVerdict:
    allowed: bool
    generation_allowed: bool
    force_human_review: bool
    reasons: tuple[str, ...]


class PolicyEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.policy_version = str(config["policy_version"])
        self.risk_labels = dict(config["risk_labels"])
        self.review_min = int(config["manual_review_min_severity"])
        self.generation_block_min = int(config["generation_block_min_severity"])
        self.brief_block_terms = tuple(str(item) for item in config.get("brief_block_terms", []))

    @classmethod
    def from_path(cls, path: str | Path) -> "PolicyEngine":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def evaluate(self, result: AuditResult) -> PolicyVerdict:
        reasons: list[str] = []
        severities: list[int] = []
        if result.policy_version != self.policy_version:
            reasons.append("unknown_policy_version")
        for label in result.risk_labels:
            rule = self.risk_labels.get(label)
            if rule is None:
                reasons.append(f"unknown_risk_label:{label}")
                continue
            severities.append(int(rule["severity"]))
            if result.decision not in rule["allowed_decisions"]:
                reasons.append(f"decision_not_allowed:{label}:{result.decision}")
        max_severity = max(severities, default=3)
        force_review = max_severity >= self.review_min or result.decision == "abstain"
        generation_allowed = result.decision in {"pass", "revise"} and max_severity < self.generation_block_min
        brief = result.generation_brief
        if brief is not None:
            searchable = " ".join((brief.objective, *brief.must_include, *brief.must_avoid))
            for term in self.brief_block_terms:
                if term.casefold() in searchable.casefold():
                    reasons.append(f"blocked_brief_term:{term}")
            if brief.human_review_required:
                force_review = True
        if result.review_required is False and force_review:
            reasons.append("missing_required_human_review")
        if brief is not None and not generation_allowed:
            reasons.append("generation_not_allowed")
        return PolicyVerdict(
            allowed=not reasons,
            generation_allowed=generation_allowed and not reasons,
            force_human_review=force_review,
            reasons=tuple(reasons),
        )

