from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping


class RequestValidationError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class AdapterRef:
    adapter_id: str
    revision: str
    artifact_sha256: str
    scale: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdapterRef":
        adapter = cls(
            adapter_id=str(value.get("adapter_id", "")),
            revision=str(value.get("revision", "")),
            artifact_sha256=str(value.get("artifact_sha256", "")),
            scale=float(value.get("scale", 1.0)),
        )
        if not adapter.adapter_id or not adapter.revision:
            raise RequestValidationError("adapter_id and revision are required")
        if len(adapter.artifact_sha256) != 64 or any(c not in "0123456789abcdef" for c in adapter.artifact_sha256.lower()):
            raise RequestValidationError("adapter artifact_sha256 must be a 64-character hex digest")
        if not 0.0 <= adapter.scale <= 2.0:
            raise RequestValidationError("adapter scale must be in [0, 2]")
        return adapter


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    idempotency_key: str
    tenant_id: str
    prompt: str
    model_id: str
    model_revision: str
    width: int
    height: int
    steps: int
    seed: int
    scheduler: str
    guidance_scale: float
    policy_version: str
    deadline_ms: int
    negative_prompt: str = ""
    submitted_at_ms: int = 0
    dtype: str = "float16"
    output_format: str = "png"
    adapters: tuple[AdapterRef, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationRequest":
        try:
            adapters = tuple(sorted(AdapterRef.from_dict(item) for item in value.get("adapters", [])))
            request = cls(
                request_id=str(value.get("request_id", "")),
                idempotency_key=str(value.get("idempotency_key", "")),
                tenant_id=str(value.get("tenant_id", "")),
                prompt=str(value.get("prompt", "")),
                negative_prompt=str(value.get("negative_prompt", "")),
                model_id=str(value.get("model_id", "")),
                model_revision=str(value.get("model_revision", "")),
                width=int(value.get("width", 0)),
                height=int(value.get("height", 0)),
                steps=int(value.get("steps", 0)),
                seed=int(value.get("seed", -1)),
                scheduler=str(value.get("scheduler", "")),
                guidance_scale=float(value.get("guidance_scale", 0.0)),
                policy_version=str(value.get("policy_version", "")),
                deadline_ms=int(value.get("deadline_ms", 0)),
                submitted_at_ms=int(value.get("submitted_at_ms", 0)),
                dtype=str(value.get("dtype", "float16")),
                output_format=str(value.get("output_format", "png")),
                adapters=adapters,
            )
        except (TypeError, ValueError) as exc:
            raise RequestValidationError(f"invalid request field: {exc}") from exc
        request.validate()
        return request

    def validate(self) -> None:
        required = {
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "tenant_id": self.tenant_id,
            "prompt": self.prompt,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "scheduler": self.scheduler,
            "policy_version": self.policy_version,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise RequestValidationError(f"missing required fields: {missing}")
        if self.width <= 0 or self.height <= 0 or self.width % 8 or self.height % 8:
            raise RequestValidationError("width and height must be positive multiples of 8")
        if self.steps <= 0:
            raise RequestValidationError("steps must be positive")
        if self.seed < 0:
            raise RequestValidationError("seed must be non-negative")
        if self.deadline_ms <= 0:
            raise RequestValidationError("deadline_ms must be positive")
        if not 0.0 <= self.guidance_scale <= 30.0:
            raise RequestValidationError("guidance_scale must be in [0, 30]")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise RequestValidationError("dtype must be float16, bfloat16, or float32")
        if self.output_format not in {"png", "webp"}:
            raise RequestValidationError("output_format must be png or webp")

    def generation_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("request_id")
        payload.pop("idempotency_key")
        payload.pop("submitted_at_ms")
        payload.pop("deadline_ms")
        payload["adapters"] = [asdict(adapter) for adapter in self.adapters]
        return payload

    def cache_key(self) -> str:
        canonical = json.dumps(self.generation_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
