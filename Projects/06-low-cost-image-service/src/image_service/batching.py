from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

from .contracts import GenerationRequest


def work_units(request: GenerationRequest) -> int:
    megapixels = request.width * request.height / 1_000_000
    return max(1, ceil(megapixels * request.steps))


def compatibility_key(request: GenerationRequest) -> tuple[object, ...]:
    adapters = tuple((a.adapter_id, a.revision, a.artifact_sha256, a.scale) for a in request.adapters)
    return (
        request.model_id,
        request.model_revision,
        request.dtype,
        request.width,
        request.height,
        request.steps,
        request.scheduler,
        request.guidance_scale,
        request.output_format,
        adapters,
    )


@dataclass(frozen=True)
class BatchLimits:
    max_batch_size: int
    max_batch_work_units: int

    def validate(self) -> None:
        if self.max_batch_size <= 0 or self.max_batch_work_units <= 0:
            raise ValueError("batch limits must be positive")


@dataclass(frozen=True)
class MicroBatch:
    requests: tuple[GenerationRequest, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return compatibility_key(self.requests[0])

    @property
    def total_work_units(self) -> int:
        return sum(work_units(request) for request in self.requests)


def plan_batches(requests: Iterable[GenerationRequest], limits: BatchLimits) -> list[MicroBatch]:
    limits.validate()
    buckets: dict[tuple[object, ...], list[GenerationRequest]] = {}
    for request in requests:
        units = work_units(request)
        if units > limits.max_batch_work_units:
            raise ValueError(f"request {request.request_id} exceeds max_batch_work_units")
        buckets.setdefault(compatibility_key(request), []).append(request)

    planned: list[MicroBatch] = []
    for bucket in buckets.values():
        current: list[GenerationRequest] = []
        current_units = 0
        for request in bucket:
            units = work_units(request)
            if current and (len(current) >= limits.max_batch_size or current_units + units > limits.max_batch_work_units):
                planned.append(MicroBatch(tuple(current)))
                current = []
                current_units = 0
            current.append(request)
            current_units += units
        if current:
            planned.append(MicroBatch(tuple(current)))
    return planned

