from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

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
    compatibility_key,
    estimate_capacity,
    plan_batches,
)
from image_service.config import load_json, validate_service_config  # noqa: E402
from scripts.validate_requests import validate_jsonl  # noqa: E402


def make_request(request_id: str, **overrides: object) -> GenerationRequest:
    payload: dict[str, object] = {
        "request_id": request_id,
        "idempotency_key": f"job-{request_id}",
        "tenant_id": "tenant-a",
        "prompt": "studio photo of a red shoe",
        "negative_prompt": "text, watermark",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_revision": "model-sha",
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "seed": 7,
        "scheduler": "euler",
        "guidance_scale": 6.5,
        "policy_version": "policy-v1",
        "deadline_ms": 1000,
        "submitted_at_ms": 100,
        "adapters": [],
    }
    payload.update(overrides)
    return GenerationRequest.from_dict(payload)


def admission() -> AdmissionController:
    return AdmissionController(
        AdmissionPolicy(
            allowed_models=frozenset({"stabilityai/stable-diffusion-xl-base-1.0"}),
            max_width=1536,
            max_height=1536,
            max_pixels=1536 * 1024,
            max_steps=50,
            max_adapters=2,
            max_queue_depth=32,
            gpu_budget_mb=24000,
            model_resident_mb=12000,
            activation_mb_per_megapixel=1500,
            adapter_mb_each=300,
            headroom_mb=2000,
        )
    )


def scheduler(max_batch_size: int = 4) -> MicroBatchScheduler:
    return MicroBatchScheduler(
        BatchLimits(max_batch_size=max_batch_size, max_batch_work_units=160),
        preferred_batch_size=max_batch_size,
        max_delay_ms=40,
    )


class ContractTests(unittest.TestCase):
    def test_cache_key_ignores_transport_identity_but_tracks_generation_inputs(self) -> None:
        base = make_request("a")
        retried = replace(base, request_id="b", idempotency_key="retry-b", submitted_at_ms=200)
        changed_seed = replace(base, request_id="c", seed=8)
        self.assertEqual(base.cache_key(), retried.cache_key())
        self.assertNotEqual(base.cache_key(), changed_seed.cache_key())

    def test_model_revision_is_part_of_batch_compatibility(self) -> None:
        self.assertNotEqual(compatibility_key(make_request("a")), compatibility_key(make_request("b", model_revision="other")))

    def test_checked_in_sample_requests_are_valid(self) -> None:
        requests = validate_jsonl(ROOT / "data" / "sample" / "requests.jsonl")
        self.assertEqual([request.request_id for request in requests], ["req-001", "req-002"])


class BatchingTests(unittest.TestCase):
    def test_incompatible_shapes_are_never_batched_together(self) -> None:
        batches = plan_batches(
            [make_request("a"), make_request("b"), make_request("c", width=768)],
            BatchLimits(max_batch_size=4, max_batch_work_units=160),
        )
        self.assertEqual(sorted(len(batch.requests) for batch in batches), [1, 2])

    def test_work_budget_splits_an_otherwise_compatible_bucket(self) -> None:
        batches = plan_batches(
            [make_request(str(index)) for index in range(4)],
            BatchLimits(max_batch_size=4, max_batch_work_units=70),
        )
        self.assertEqual([len(batch.requests) for batch in batches], [2, 2])

    def test_deadline_is_checked_before_execution(self) -> None:
        queue = scheduler()
        queue.enqueue(make_request("expired", deadline_ms=50))
        drained = queue.drain(now_ms=150, force=True)
        self.assertEqual([item.request_id for item in drained.expired], ["expired"])
        self.assertEqual(drained.batches, ())


class AdmissionTests(unittest.TestCase):
    def test_oversized_request_is_rejected_with_reasons(self) -> None:
        decision = admission().evaluate(make_request("large", width=2048, height=2048), queue_depth=0)
        self.assertFalse(decision.accepted)
        self.assertIn("dimension_limit", decision.reasons)
        self.assertIn("pixel_limit", decision.reasons)


class ServiceTests(unittest.TestCase):
    @staticmethod
    def temporary_directory():
        temporary_root = ROOT / "artifacts" / "test-tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=temporary_root)

    def test_exact_cache_hit_bypasses_backend(self) -> None:
        with self.temporary_directory() as directory:
            backend = FakeBackend()
            service = ImageGenerationService(
                admission=admission(),
                scheduler=scheduler(),
                cache=ContentAddressedCache(directory),
                backend=backend,
            )
            first = service.submit_many([make_request("first")], now_ms=100)[0]
            second_request = make_request("second", idempotency_key="new-transport-key", submitted_at_ms=200)
            second = service.submit_many([second_request], now_ms=200)[0]
            self.assertEqual(first.status, "generated")
            self.assertEqual(second.status, "cache_hit")
            self.assertEqual(backend.calls, 1)

    def test_oom_is_split_with_a_bounded_retry_budget(self) -> None:
        backend = FakeBackend(oom_over_batch_size=2, base_batch_ms=100, per_image_ms=10)
        metrics = ServiceMetrics()
        service = ImageGenerationService(
            admission=admission(),
            scheduler=scheduler(),
            cache=None,
            backend=backend,
            metrics=metrics,
            oom_split_retries=1,
        )
        responses = service.submit_many([make_request(str(index)) for index in range(4)], now_ms=100)
        self.assertEqual([response.status for response in responses], ["generated"] * 4)
        self.assertEqual(backend.batch_sizes, [4, 2, 2])
        snapshot = metrics.snapshot(gpu_hour_price=10.0)
        self.assertEqual(snapshot["oom_attempt_count"], 1)
        self.assertGreater(snapshot["gpu_cost_per_accepted_image"], 0)

    def test_batch_is_split_by_vram_estimate_before_backend_execution(self) -> None:
        policy = admission().policy
        constrained = AdmissionController(replace(policy, gpu_budget_mb=17000))
        backend = FakeBackend()
        service = ImageGenerationService(
            admission=constrained,
            scheduler=scheduler(),
            cache=None,
            backend=backend,
        )
        responses = service.submit_many([make_request(str(index)) for index in range(4)], now_ms=100)
        self.assertEqual([response.status for response in responses], ["generated"] * 4)
        self.assertEqual(backend.batch_sizes, [1, 1, 1, 1])

    def test_guard_rejected_output_is_not_cached(self) -> None:
        with self.temporary_directory() as directory:
            backend = FakeBackend(guard_reject_ids=frozenset({"blocked"}))
            service = ImageGenerationService(
                admission=admission(),
                scheduler=scheduler(),
                cache=ContentAddressedCache(directory),
                backend=backend,
            )
            response = service.submit_many([make_request("blocked")], now_ms=100)[0]
            self.assertEqual(response.status, "guard_rejected")
            self.assertIsNone(ContentAddressedCache(directory).get("tenant-a", response.cache_key))


class ReportingTests(unittest.TestCase):
    def test_capacity_uses_accepted_outputs_and_all_gpu_seconds(self) -> None:
        estimate = estimate_capacity(
            accepted_images=100,
            aggregate_gpu_seconds=200,
            p95_end_to_end_ms=4000,
            utilization_target=0.7,
            gpu_hour_price=12.0,
        )
        self.assertEqual(estimate.accepted_images_per_gpu_hour, 1800.0)
        self.assertAlmostEqual(estimate.gpu_cost_per_accepted_image, 12.0 / 1800.0)

    def test_checked_in_config_is_explicitly_not_measured(self) -> None:
        config = load_json(ROOT / "configs" / "service.json")
        validate_service_config(config)
        project = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
        self.assertEqual(config["result_status"], "not_measured")
        self.assertEqual(project["result_status"], "not_measured")


if __name__ == "__main__":
    unittest.main()
