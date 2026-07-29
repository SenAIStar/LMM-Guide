from __future__ import annotations

from dataclasses import dataclass

from .batching import BatchLimits, MicroBatch, compatibility_key, plan_batches
from .contracts import GenerationRequest


@dataclass(frozen=True)
class DrainResult:
    batches: tuple[MicroBatch, ...]
    expired: tuple[GenerationRequest, ...]


class MicroBatchScheduler:
    def __init__(self, limits: BatchLimits, *, preferred_batch_size: int, max_delay_ms: int) -> None:
        if not 1 <= preferred_batch_size <= limits.max_batch_size:
            raise ValueError("preferred_batch_size must be within max_batch_size")
        if max_delay_ms < 0:
            raise ValueError("max_delay_ms must be non-negative")
        self.limits = limits
        self.preferred_batch_size = preferred_batch_size
        self.max_delay_ms = max_delay_ms
        self._pending: list[GenerationRequest] = []

    @property
    def depth(self) -> int:
        return len(self._pending)

    def enqueue(self, request: GenerationRequest) -> None:
        self._pending.append(request)

    def drain(self, now_ms: int, *, force: bool = False) -> DrainResult:
        expired: list[GenerationRequest] = []
        live: list[GenerationRequest] = []
        for request in self._pending:
            if now_ms - request.submitted_at_ms >= request.deadline_ms:
                expired.append(request)
            else:
                live.append(request)

        buckets: dict[tuple[object, ...], list[GenerationRequest]] = {}
        for request in live:
            buckets.setdefault(compatibility_key(request), []).append(request)

        ready: list[GenerationRequest] = []
        deferred: list[GenerationRequest] = []
        for bucket in buckets.values():
            oldest_wait = max(0, now_ms - bucket[0].submitted_at_ms)
            if force or len(bucket) >= self.preferred_batch_size or oldest_wait >= self.max_delay_ms:
                ready.extend(bucket)
            else:
                deferred.extend(bucket)

        self._pending = deferred
        return DrainResult(tuple(plan_batches(ready, self.limits)), tuple(expired))

