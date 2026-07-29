from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from pathlib import Path
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
)
from image_service.config import load_json, validate_service_config  # noqa: E402
from image_service.diffusers_backend import DiffusersBackend, DiffusersBackendConfig  # noqa: E402
from image_service.gateway import AsyncBatchingGateway  # noqa: E402


def build_app(config_path: Path, backend_name: str):
    from fastapi import FastAPI, HTTPException

    config = load_json(config_path)
    validate_service_config(config)
    admission = AdmissionController(AdmissionPolicy.from_dict(config["admission"]))
    batch = config["batching"]
    scheduler = MicroBatchScheduler(
        BatchLimits(int(batch["max_batch_size"]), int(batch["max_batch_work_units"])),
        preferred_batch_size=int(batch["preferred_batch_size"]),
        max_delay_ms=int(batch["max_delay_ms"]),
    )
    if backend_name == "fake":
        backend = FakeBackend()
    else:
        backend = DiffusersBackend(DiffusersBackendConfig(**config["model"]))
    service = ImageGenerationService(
        admission=admission,
        scheduler=scheduler,
        cache=ContentAddressedCache(ROOT / config["cache"]["root"]),
        backend=backend,
        oom_split_retries=int(batch["oom_split_retries"]),
    )
    gateway = AsyncBatchingGateway(
        service,
        max_delay_ms=int(batch["max_delay_ms"]),
        max_queue_size=int(config["admission"]["max_queue_depth"]),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await gateway.start()
        yield
        await gateway.close()

    app = FastAPI(title="Low-cost image service", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "backend": backend_name, "result_status": config["result_status"]}

    @app.get("/metrics/snapshot")
    async def metrics_snapshot() -> dict[str, object]:
        return service.metrics.snapshot(gpu_hour_price=config["pricing"]["gpu_hour_price"])

    @app.post("/v1/images/generate")
    async def generate(payload: dict[str, object]) -> dict[str, object]:
        try:
            request = GenerationRequest.from_dict(payload)
            response = await gateway.generate(request)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "request_id": response.request_id,
            "status": response.status,
            "cache_key": response.cache_key,
            "reason": response.reason,
            "backend_revision": response.backend_revision,
            "image_base64": base64.b64encode(response.image_bytes).decode("ascii") if response.image_bytes else None,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the image generation gateway.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "service.json")
    parser.add_argument("--backend", choices=["fake", "diffusers"], default="fake")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(build_app(args.config, args.backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

