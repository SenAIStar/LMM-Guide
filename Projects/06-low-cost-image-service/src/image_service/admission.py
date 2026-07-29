from __future__ import annotations

from dataclasses import dataclass

from .contracts import GenerationRequest


@dataclass(frozen=True)
class AdmissionPolicy:
    allowed_models: frozenset[str]
    max_width: int
    max_height: int
    max_pixels: int
    max_steps: int
    max_adapters: int
    max_queue_depth: int
    gpu_budget_mb: int
    model_resident_mb: int
    activation_mb_per_megapixel: float
    adapter_mb_each: int
    headroom_mb: int

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "AdmissionPolicy":
        return cls(
            allowed_models=frozenset(str(item) for item in value["allowed_models"]),
            max_width=int(value["max_width"]),
            max_height=int(value["max_height"]),
            max_pixels=int(value["max_pixels"]),
            max_steps=int(value["max_steps"]),
            max_adapters=int(value["max_adapters"]),
            max_queue_depth=int(value["max_queue_depth"]),
            gpu_budget_mb=int(value["gpu_budget_mb"]),
            model_resident_mb=int(value["model_resident_mb"]),
            activation_mb_per_megapixel=float(value["activation_mb_per_megapixel"]),
            adapter_mb_each=int(value["adapter_mb_each"]),
            headroom_mb=int(value["headroom_mb"]),
        )


@dataclass(frozen=True)
class AdmissionResult:
    accepted: bool
    estimated_peak_mb: int
    reasons: tuple[str, ...]


class AdmissionController:
    def __init__(self, policy: AdmissionPolicy) -> None:
        self.policy = policy

    def estimate_peak_mb(self, request: GenerationRequest, batch_size: int = 1) -> int:
        megapixels = request.width * request.height / 1_000_000
        activation = self.policy.activation_mb_per_megapixel * megapixels * batch_size
        adapter_memory = self.policy.adapter_mb_each * len(request.adapters)
        return int(self.policy.model_resident_mb + activation + adapter_memory + self.policy.headroom_mb)

    def evaluate(self, request: GenerationRequest, queue_depth: int) -> AdmissionResult:
        reasons: list[str] = []
        if request.model_id not in self.policy.allowed_models:
            reasons.append("model_not_allowed")
        if request.width > self.policy.max_width or request.height > self.policy.max_height:
            reasons.append("dimension_limit")
        if request.width * request.height > self.policy.max_pixels:
            reasons.append("pixel_limit")
        if request.steps > self.policy.max_steps:
            reasons.append("step_limit")
        if len(request.adapters) > self.policy.max_adapters:
            reasons.append("adapter_limit")
        if queue_depth >= self.policy.max_queue_depth:
            reasons.append("queue_full")
        estimate = self.estimate_peak_mb(request)
        if estimate > self.policy.gpu_budget_mb:
            reasons.append("estimated_vram_limit")
        return AdmissionResult(not reasons, estimate, tuple(reasons))

