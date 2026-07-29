from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class CapacityEstimate:
    accepted_images_per_gpu_hour: float
    sustainable_arrival_rps: float
    required_concurrency_at_p95: int
    gpu_cost_per_accepted_image: float | None


def estimate_capacity(
    *,
    accepted_images: int,
    aggregate_gpu_seconds: float,
    p95_end_to_end_ms: float,
    utilization_target: float,
    gpu_hour_price: float | None,
) -> CapacityEstimate:
    if accepted_images <= 0 or aggregate_gpu_seconds <= 0 or p95_end_to_end_ms < 0:
        raise ValueError("accepted_images and aggregate_gpu_seconds must be positive; latency cannot be negative")
    if not 0 < utilization_target <= 1:
        raise ValueError("utilization_target must be in (0, 1]")
    per_hour = accepted_images / aggregate_gpu_seconds * 3600.0
    sustainable_rps = per_hour / 3600.0 * utilization_target
    concurrency = max(1, ceil(sustainable_rps * p95_end_to_end_ms / 1000.0))
    cost = gpu_hour_price / per_hour if gpu_hour_price is not None else None
    return CapacityEstimate(per_hour, sustainable_rps, concurrency, cost)

