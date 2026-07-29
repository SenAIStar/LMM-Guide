from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from .batching import MicroBatch


class BackendOOM(RuntimeError):
    def __init__(self, message: str, elapsed_ms: float) -> None:
        super().__init__(message)
        self.elapsed_ms = elapsed_ms


@dataclass(frozen=True)
class GeneratedArtifact:
    request_id: str
    image_bytes: bytes
    backend_revision: str
    guard_accepted: bool = True
    guard_reason: str | None = None


@dataclass(frozen=True)
class BatchResult:
    artifacts: tuple[GeneratedArtifact, ...]
    elapsed_ms: float


class ImageBackend(Protocol):
    def generate(self, batch: MicroBatch) -> BatchResult:
        ...


class FakeBackend:
    """Deterministic backend for control-plane tests; it is not a GPU benchmark."""

    def __init__(
        self,
        *,
        backend_revision: str = "fake-v1",
        base_batch_ms: float = 100.0,
        per_image_ms: float = 20.0,
        oom_over_batch_size: int | None = None,
        guard_reject_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.backend_revision = backend_revision
        self.base_batch_ms = base_batch_ms
        self.per_image_ms = per_image_ms
        self.oom_over_batch_size = oom_over_batch_size
        self.guard_reject_ids = guard_reject_ids
        self.calls = 0
        self.batch_sizes: list[int] = []

    def generate(self, batch: MicroBatch) -> BatchResult:
        self.calls += 1
        batch_size = len(batch.requests)
        self.batch_sizes.append(batch_size)
        elapsed_ms = self.base_batch_ms + self.per_image_ms * batch_size
        if self.oom_over_batch_size is not None and batch_size > self.oom_over_batch_size:
            raise BackendOOM("simulated OOM; not a measured CUDA event", elapsed_ms)
        artifacts: list[GeneratedArtifact] = []
        for request in batch.requests:
            digest = hashlib.sha256(f"{request.cache_key()}:{self.backend_revision}".encode("utf-8")).digest()
            accepted = request.request_id not in self.guard_reject_ids
            artifacts.append(
                GeneratedArtifact(
                    request_id=request.request_id,
                    image_bytes=digest,
                    backend_revision=self.backend_revision,
                    guard_accepted=accepted,
                    guard_reason=None if accepted else "simulated_guard_reject",
                )
            )
        return BatchResult(tuple(artifacts), elapsed_ms)

