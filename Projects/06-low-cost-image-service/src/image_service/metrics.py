from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(quantile * len(ordered)) - 1)
    return ordered[index]


@dataclass(frozen=True)
class RequestMetric:
    status: str
    end_to_end_ms: float
    queue_wait_ms: float
    cache_hit: bool


@dataclass(frozen=True)
class BatchAttemptMetric:
    elapsed_ms: float
    batch_size: int
    oom: bool


class ServiceMetrics:
    def __init__(self) -> None:
        self.requests: list[RequestMetric] = []
        self.batch_attempts: list[BatchAttemptMetric] = []

    def record_request(self, metric: RequestMetric) -> None:
        self.requests.append(metric)

    def record_batch_attempt(self, elapsed_ms: float, batch_size: int, *, oom: bool) -> None:
        self.batch_attempts.append(BatchAttemptMetric(elapsed_ms, batch_size, oom))

    def snapshot(self, *, gpu_hour_price: float | None = None) -> dict[str, Any]:
        total = len(self.requests)
        accepted = sum(item.status in {"generated", "cache_hit"} for item in self.requests)
        cache_hits = sum(item.cache_hit for item in self.requests)
        guard_pass_denominator = sum(item.status in {"generated", "guard_rejected"} for item in self.requests)
        gpu_seconds = sum(item.elapsed_ms for item in self.batch_attempts) / 1000.0
        attempt_count = len(self.batch_attempts)
        snapshot: dict[str, Any] = {
            "request_count": total,
            "accepted_output_count": accepted,
            "generated_output_count": sum(item.status == "generated" for item in self.requests),
            "cache_hit_count": cache_hits,
            "cache_hit_rate": cache_hits / total if total else None,
            "quality_guard_pass_rate": (
                sum(item.status == "generated" for item in self.requests) / guard_pass_denominator
                if guard_pass_denominator
                else None
            ),
            "end_to_end_p50_ms": percentile([item.end_to_end_ms for item in self.requests], 0.50),
            "end_to_end_p95_ms": percentile([item.end_to_end_ms for item in self.requests], 0.95),
            "end_to_end_p99_ms": percentile([item.end_to_end_ms for item in self.requests], 0.99),
            "queue_wait_p95_ms": percentile([item.queue_wait_ms for item in self.requests], 0.95),
            "gpu_seconds_all_attempts": gpu_seconds,
            "batch_attempt_count": attempt_count,
            "oom_attempt_count": sum(item.oom for item in self.batch_attempts),
            "oom_attempt_rate": sum(item.oom for item in self.batch_attempts) / attempt_count if attempt_count else None,
            "mean_attempt_batch_size": (
                sum(item.batch_size for item in self.batch_attempts) / attempt_count if attempt_count else None
            ),
            "result_status": "simulated_or_measured_by_backend_caller",
        }
        if gpu_hour_price is not None:
            total_gpu_cost = gpu_seconds / 3600.0 * gpu_hour_price
            snapshot["gpu_cost"] = total_gpu_cost
            snapshot["gpu_cost_per_accepted_image"] = total_gpu_cost / accepted if accepted else None
        else:
            snapshot["gpu_cost"] = None
            snapshot["gpu_cost_per_accepted_image"] = None
        return snapshot

