from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import time
from typing import Any

from .backend import BackendOOM, BatchResult, GeneratedArtifact
from .batching import MicroBatch
from .contracts import AdapterRef


@dataclass(frozen=True)
class DiffusersBackendConfig:
    model_id: str
    model_revision: str
    dtype: str = "float16"
    device: str = "cuda"
    compile_unet: bool = False
    compile_mode: str = "reduce-overhead"
    lora_mode: str = "dynamic"
    vae_slicing: bool = True
    vae_tiling: bool = False
    model_cpu_offload: bool = False


class DiffusersBackend:
    """Optional real backend. Pin model and adapter artifacts before using it for a measured run."""

    def __init__(self, config: DiffusersBackendConfig, adapter_catalog: dict[tuple[str, str], Path] | None = None) -> None:
        if config.compile_unet and config.lora_mode == "dynamic":
            raise ValueError("compile_unet requires fixed LoRA mode to avoid untracked recompilation")
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.torch = torch
        self.config = config
        self.adapter_catalog = adapter_catalog or {}
        dtype = getattr(torch, config.dtype)
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        if config.model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(config.device)
        if config.vae_slicing:
            self.pipeline.enable_vae_slicing()
        if config.vae_tiling:
            self.pipeline.enable_vae_tiling()
        if config.compile_unet:
            self.pipeline.unet = torch.compile(self.pipeline.unet, mode=config.compile_mode, fullgraph=True)
        self._active_adapters: tuple[AdapterRef, ...] | None = None
        self.backend_revision = f"{config.model_id}@{config.model_revision}"

    def generate(self, batch: MicroBatch) -> BatchResult:
        request = batch.requests[0]
        self._activate_scheduler(request.scheduler)
        self._activate_adapters(request.adapters)
        generators = [self.torch.Generator(device=self.config.device).manual_seed(item.seed) for item in batch.requests]
        if self.config.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            output = self.pipeline(
                prompt=[item.prompt for item in batch.requests],
                negative_prompt=[item.negative_prompt for item in batch.requests],
                width=request.width,
                height=request.height,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                generator=generators,
            )
            if self.config.device.startswith("cuda"):
                self.torch.cuda.synchronize()
        except self.torch.cuda.OutOfMemoryError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.torch.cuda.empty_cache()
            raise BackendOOM("CUDA OOM; the service may split this batch within its bounded retry budget", elapsed_ms) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        artifacts: list[GeneratedArtifact] = []
        for item, image in zip(batch.requests, output.images, strict=True):
            buffer = BytesIO()
            image.save(buffer, format=item.output_format.upper())
            artifacts.append(GeneratedArtifact(item.request_id, buffer.getvalue(), self.backend_revision))
        return BatchResult(tuple(artifacts), elapsed_ms)

    def _activate_scheduler(self, name: str) -> None:
        from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

        scheduler_types: dict[str, Any] = {
            "euler": EulerDiscreteScheduler,
            "dpmpp-2m": DPMSolverMultistepScheduler,
        }
        if name not in scheduler_types:
            raise ValueError(f"unsupported scheduler: {name}")
        desired_type = scheduler_types[name]
        if not isinstance(self.pipeline.scheduler, desired_type):
            self.pipeline.scheduler = desired_type.from_config(self.pipeline.scheduler.config)

    def _activate_adapters(self, adapters: tuple[AdapterRef, ...]) -> None:
        if adapters == self._active_adapters:
            return
        if self.config.lora_mode == "fixed" and self._active_adapters is not None:
            raise ValueError("fixed LoRA backend cannot change adapter set after warmup")
        if self._active_adapters:
            self.pipeline.unload_lora_weights()
        names: list[str] = []
        weights: list[float] = []
        for index, adapter in enumerate(adapters):
            path = self.adapter_catalog.get((adapter.adapter_id, adapter.revision))
            if path is None:
                raise ValueError(f"adapter is not in the local approved catalog: {adapter.adapter_id}@{adapter.revision}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != adapter.artifact_sha256:
                raise ValueError(f"adapter hash mismatch: {adapter.adapter_id}@{adapter.revision}")
            name = f"adapter_{index}"
            self.pipeline.load_lora_weights(path.parent, weight_name=path.name, adapter_name=name)
            names.append(name)
            weights.append(adapter.scale)
        if names:
            self.pipeline.set_adapters(names, adapter_weights=weights)
        self._active_adapters = adapters

