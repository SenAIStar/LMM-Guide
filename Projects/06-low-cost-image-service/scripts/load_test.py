from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_service import (  # noqa: E402
    AdmissionController,
    AdmissionPolicy,
    BatchLimits,
    ContentAddressedCache,
    FakeBackend,
    GenerationRequest,
    ImageGenerationService,
    MicroBatchScheduler,
    ServiceMetrics,
)
from image_service.config import load_json, validate_service_config  # noqa: E402


def weighted_choice(rng: random.Random, items: list[dict[str, object]]) -> dict[str, object]:
    point = rng.random() * sum(float(item["weight"]) for item in items)
    cumulative = 0.0
    for item in items:
        cumulative += float(item["weight"])
        if point <= cumulative:
            return item
    return items[-1]


def build_request(index: int, submitted_at_ms: int, mix: dict[str, object], tenant_count: int) -> GenerationRequest:
    return GenerationRequest.from_dict(
        {
            "request_id": f"load-{index:05d}",
            "idempotency_key": f"asset-{index:05d}",
            "tenant_id": f"tenant-{index % tenant_count}",
            "prompt": f"catalog product photograph, sample {index}",
            "negative_prompt": "text, watermark, duplicate object",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "model_revision": "PIN_A_COMMIT_SHA_BEFORE_GPU_RUN",
            "width": mix["width"],
            "height": mix["height"],
            "steps": mix["steps"],
            "seed": 100000 + index,
            "scheduler": mix["scheduler"],
            "guidance_scale": 6.5,
            "policy_version": "image-policy-v1",
            "deadline_ms": 30000,
            "submitted_at_ms": submitted_at_ms,
            "adapters": [],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic control-plane simulation; this is not a GPU benchmark.")
    parser.add_argument("--service-config", type=Path, default=ROOT / "configs" / "service.json")
    parser.add_argument("--load-config", type=Path, default=ROOT / "configs" / "load_test.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    service_config = load_json(args.service_config)
    validate_service_config(service_config)
    load_config = load_json(args.load_config)
    rng = random.Random(int(load_config["seed"]))
    admission = AdmissionController(AdmissionPolicy.from_dict(service_config["admission"]))
    batching = service_config["batching"]
    scheduler = MicroBatchScheduler(
        BatchLimits(int(batching["max_batch_size"]), int(batching["max_batch_work_units"])),
        preferred_batch_size=int(batching["preferred_batch_size"]),
        max_delay_ms=int(batching["max_delay_ms"]),
    )
    fake = load_config["fake_backend"]
    backend = FakeBackend(
        base_batch_ms=float(fake["base_batch_ms"]),
        per_image_ms=float(fake["per_image_ms"]),
        oom_over_batch_size=int(fake["oom_over_batch_size"]),
    )
    metrics = ServiceMetrics()
    cache = ContentAddressedCache(ROOT / "artifacts" / "cache")
    service = ImageGenerationService(
        admission=admission,
        scheduler=scheduler,
        cache=cache,
        backend=backend,
        metrics=metrics,
        oom_split_retries=int(batching["oom_split_retries"]),
    )

    interval_ms = int(1000 / float(load_config["arrival_rate_rps"]))
    previous: list[GenerationRequest] = []
    window: list[GenerationRequest] = []
    all_responses = []
    for index in range(int(load_config["request_count"])):
        submitted_at_ms = index * interval_ms
        if previous and rng.random() < float(load_config["duplicate_ratio"]):
            source = rng.choice(previous)
            request = replace(
                source,
                request_id=f"load-{index:05d}",
                idempotency_key=f"retry-{index:05d}",
                submitted_at_ms=submitted_at_ms,
            )
        else:
            request = build_request(
                index,
                submitted_at_ms,
                weighted_choice(rng, load_config["request_mix"]),
                int(load_config["tenant_count"]),
            )
        previous.append(request)
        window.append(request)
        if len(window) == 8:
            all_responses.extend(service.submit_many(window, now_ms=submitted_at_ms))
            window = []
    if window:
        all_responses.extend(service.submit_many(window, now_ms=window[-1].submitted_at_ms))

    report = metrics.snapshot(gpu_hour_price=service_config["pricing"]["gpu_hour_price"])
    report.update(
        {
            "result_status": "simulated_not_gpu_measured",
            "backend": "deterministic_fake_backend",
            "response_status_counts": {
                status: sum(response.status == status for response in all_responses)
                for status in sorted({response.status for response in all_responses})
            },
            "warning": "Timing, OOM, throughput, memory, quality, and cost values from this run are synthetic.",
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

