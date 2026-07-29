from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DECISIONS = {"pass", "revise", "reject", "abstain"}
EVIDENCE_KINDS = {"bbox", "text_span", "global"}


def _require_keys(payload: dict[str, Any], required: set[str], allowed: set[str]) -> None:
    missing = sorted(required - payload.keys())
    extra = sorted(payload.keys() - allowed)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={missing}, extra={extra}")


@dataclass(frozen=True)
class Evidence:
    kind: str
    source: str
    quote: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Evidence":
        allowed = {"kind", "source", "quote", "bbox"}
        _require_keys(payload, {"kind", "source"}, allowed)
        kind = str(payload["kind"])
        if kind not in EVIDENCE_KINDS:
            raise ValueError(f"unsupported evidence kind: {kind}")
        bbox_raw = payload.get("bbox")
        bbox = tuple(float(value) for value in bbox_raw) if bbox_raw is not None else None
        if bbox is not None:
            if len(bbox) != 4 or any(value < 0 or value > 1 for value in bbox):
                raise ValueError("bbox must contain four normalized coordinates")
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("bbox must have positive area")
        if kind == "bbox" and bbox is None:
            raise ValueError("bbox evidence requires bbox coordinates")
        if kind == "text_span" and not payload.get("quote"):
            raise ValueError("text_span evidence requires a quote")
        return cls(kind=kind, source=str(payload["source"]), quote=payload.get("quote"), bbox=bbox)


@dataclass(frozen=True)
class GenerationBrief:
    objective: str
    must_include: tuple[str, ...]
    must_avoid: tuple[str, ...]
    human_review_required: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationBrief":
        keys = {"objective", "must_include", "must_avoid", "human_review_required"}
        _require_keys(payload, keys, keys)
        objective = str(payload["objective"]).strip()
        if not objective:
            raise ValueError("generation objective cannot be empty")
        return cls(
            objective=objective,
            must_include=tuple(str(item) for item in payload["must_include"]),
            must_avoid=tuple(str(item) for item in payload["must_avoid"]),
            human_review_required=bool(payload["human_review_required"]),
        )


@dataclass(frozen=True)
class AuditResult:
    asset_id: str
    policy_version: str
    decision: str
    risk_labels: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    confidence: float
    rationale: str
    generation_brief: GenerationBrief | None
    review_required: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AuditResult":
        keys = {
            "asset_id",
            "policy_version",
            "decision",
            "risk_labels",
            "evidence",
            "confidence",
            "rationale",
            "generation_brief",
            "review_required",
        }
        _require_keys(payload, keys, keys)
        decision = str(payload["decision"])
        if decision not in DECISIONS:
            raise ValueError(f"unsupported decision: {decision}")
        confidence = float(payload["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        labels = tuple(sorted(set(str(item) for item in payload["risk_labels"])))
        if not labels:
            raise ValueError("risk_labels cannot be empty; use ['none'] for clean content")
        evidence = tuple(Evidence.from_dict(item) for item in payload["evidence"])
        if labels != ("none",) and not evidence:
            raise ValueError("risk decisions require grounded evidence")
        brief_raw = payload["generation_brief"]
        brief = GenerationBrief.from_dict(brief_raw) if brief_raw is not None else None
        if decision in {"reject", "abstain"} and brief is not None:
            raise ValueError("reject and abstain decisions cannot trigger generation")
        if decision == "abstain" and not payload["review_required"]:
            raise ValueError("abstain requires human review")
        return cls(
            asset_id=str(payload["asset_id"]),
            policy_version=str(payload["policy_version"]),
            decision=decision,
            risk_labels=labels,
            evidence=evidence,
            confidence=confidence,
            rationale=str(payload["rationale"]),
            generation_brief=brief,
            review_required=bool(payload["review_required"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

