from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .admission import AdmissionController
from .backend import BackendOOM, ImageBackend
from .batching import MicroBatch
from .cache import ContentAddressedCache
from .contracts import GenerationRequest
from .metrics import RequestMetric, ServiceMetrics
from .scheduler import MicroBatchScheduler


@dataclass(frozen=True)
class ServiceResponse:
    request_id: str
    status: str
    cache_key: str
    image_bytes: bytes | None
    reason: str | None
    backend_revision: str | None


class ImageGenerationService:
    def __init__(
        self,
        *,
        admission: AdmissionController,
        scheduler: MicroBatchScheduler,
        cache: ContentAddressedCache | None,
        backend: ImageBackend,
        metrics: ServiceMetrics | None = None,
        oom_split_retries: int = 2,
    ) -> None:
        if oom_split_retries < 0:
            raise ValueError("oom_split_retries must be non-negative")
        self.admission = admission
        self.scheduler = scheduler
        self.cache = cache
        self.backend = backend
        self.metrics = metrics or ServiceMetrics()
        self.oom_split_retries = oom_split_retries

    def submit_many(self, requests: Iterable[GenerationRequest], *, now_ms: int) -> list[ServiceResponse]:
        responses: dict[str, ServiceResponse] = {}
        requests_by_id: dict[str, GenerationRequest] = {}
        for request in requests:
            if request.request_id in requests_by_id:
                raise ValueError(f"duplicate request_id in submission: {request.request_id}")
            requests_by_id[request.request_id] = request
            cache_key = request.cache_key()
            decision = self.admission.evaluate(request, self.scheduler.depth)
            if not decision.accepted:
                responses[request.request_id] = ServiceResponse(
                    request.request_id, "rejected", cache_key, None, ",".join(decision.reasons), None
                )
                self._record_request(request, now_ms, "rejected", cache_hit=False, queue_wait_ms=0)
                continue
            cached = self.cache.get(request.tenant_id, cache_key) if self.cache else None
            if cached is not None:
                responses[request.request_id] = ServiceResponse(
                    request.request_id,
                    "cache_hit",
                    cache_key,
                    cached.image_bytes,
                    None,
                    str(cached.metadata.get("backend_revision", "unknown")),
                )
                self._record_request(request, now_ms, "cache_hit", cache_hit=True, queue_wait_ms=0)
                continue
            self.scheduler.enqueue(request)

        drain = self.scheduler.drain(now_ms, force=True)
        for request in drain.expired:
            responses[request.request_id] = ServiceResponse(
                request.request_id, "deadline_exceeded", request.cache_key(), None, "expired_in_queue", None
            )
            self._record_request(
                request,
                now_ms,
                "deadline_exceeded",
                cache_hit=False,
                queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
            )
        for batch in drain.batches:
            for fitted_batch in self._fit_vram(batch):
                for response in self._execute_batch(
                    fitted_batch, retries_left=self.oom_split_retries, now_ms=now_ms
                ):
                    responses[response.request_id] = response

        return [responses[request_id] for request_id in requests_by_id]

    def _fit_vram(self, batch: MicroBatch) -> list[MicroBatch]:
        estimate = self.admission.estimate_peak_mb(batch.requests[0], batch_size=len(batch.requests))
        if estimate <= self.admission.policy.gpu_budget_mb or len(batch.requests) == 1:
            return [batch]
        midpoint = len(batch.requests) // 2
        return self._fit_vram(MicroBatch(batch.requests[:midpoint])) + self._fit_vram(
            MicroBatch(batch.requests[midpoint:])
        )

    def _execute_batch(self, batch: MicroBatch, *, retries_left: int, now_ms: int) -> list[ServiceResponse]:
        try:
            result = self.backend.generate(batch)
            self.metrics.record_batch_attempt(result.elapsed_ms, len(batch.requests), oom=False)
        except BackendOOM as exc:
            self.metrics.record_batch_attempt(exc.elapsed_ms, len(batch.requests), oom=True)
            if retries_left > 0 and len(batch.requests) > 1:
                midpoint = len(batch.requests) // 2
                left = MicroBatch(batch.requests[:midpoint])
                right = MicroBatch(batch.requests[midpoint:])
                return self._execute_batch(left, retries_left=retries_left - 1, now_ms=now_ms) + self._execute_batch(
                    right, retries_left=retries_left - 1, now_ms=now_ms
                )
            failed: list[ServiceResponse] = []
            for request in batch.requests:
                failed.append(
                    ServiceResponse(request.request_id, "oom_failed", request.cache_key(), None, str(exc), None)
                )
                self._record_request(
                    request,
                    now_ms,
                    "oom_failed",
                    cache_hit=False,
                    queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
                )
            return failed

        artifacts = {artifact.request_id: artifact for artifact in result.artifacts}
        completed: list[ServiceResponse] = []
        for request in batch.requests:
            artifact = artifacts.get(request.request_id)
            if artifact is None:
                completed.append(
                    ServiceResponse(request.request_id, "backend_error", request.cache_key(), None, "missing_artifact", None)
                )
                self._record_request(
                    request,
                    now_ms,
                    "backend_error",
                    cache_hit=False,
                    queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
                )
                continue
            if not artifact.guard_accepted:
                completed.append(
                    ServiceResponse(
                        request.request_id,
                        "guard_rejected",
                        request.cache_key(),
                        None,
                        artifact.guard_reason,
                        artifact.backend_revision,
                    )
                )
                self._record_request(
                    request,
                    now_ms + int(result.elapsed_ms),
                    "guard_rejected",
                    cache_hit=False,
                    queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
                )
                continue
            if self.cache:
                self.cache.put(
                    request.tenant_id,
                    request.cache_key(),
                    artifact.image_bytes,
                    {
                        "request_id": request.request_id,
                        "idempotency_key": request.idempotency_key,
                        "backend_revision": artifact.backend_revision,
                        "policy_version": request.policy_version,
                    },
                )
            completed.append(
                ServiceResponse(
                    request.request_id,
                    "generated",
                    request.cache_key(),
                    artifact.image_bytes,
                    None,
                    artifact.backend_revision,
                )
            )
            self._record_request(
                request,
                now_ms + int(result.elapsed_ms),
                "generated",
                cache_hit=False,
                queue_wait_ms=max(0, now_ms - request.submitted_at_ms),
            )
        return completed

    def _record_request(
        self,
        request: GenerationRequest,
        finished_at_ms: int,
        status: str,
        *,
        cache_hit: bool,
        queue_wait_ms: float,
    ) -> None:
        elapsed = max(0, finished_at_ms - request.submitted_at_ms)
        self.metrics.record_request(RequestMetric(status, elapsed, max(0, queue_wait_ms), cache_hit))
