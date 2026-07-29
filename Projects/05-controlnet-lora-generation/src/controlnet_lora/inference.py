from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .provenance import make_run_record


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_model", "controlnet_model", "prompt", "negative_prompt"):
        if not str(config.get(key, "")).strip():
            raise ValueError(f"missing inference config key: {key}")
    for key in ("controlnet_conditioning_scale", "lora_scale"):
        value = float(config[key])
        if not 0 <= value <= 2:
            raise ValueError(f"{key} must be in [0, 2]")
    return config


def generate(
    *,
    config_path: Path,
    control_image_path: Path,
    lora_path: str,
    output_path: Path,
) -> dict[str, Any]:
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("install requirements-ml.txt before model inference") from exc

    config = load_config(config_path)
    dtype_name = str(config.get("dtype", "float16"))
    dtype = getattr(torch, dtype_name)
    controlnet = ControlNetModel.from_pretrained(
        config["controlnet_model"],
        revision=config.get("controlnet_revision"),
        torch_dtype=dtype,
    )
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        config["base_model"],
        revision=config.get("base_revision"),
        controlnet=controlnet,
        torch_dtype=dtype,
    ).to(config.get("device", "cuda"))
    pipeline.load_lora_weights(
        lora_path,
        weight_name=config.get("lora_weight_name", "pytorch_lora_weights.safetensors"),
        adapter_name="brand",
    )
    pipeline.set_adapters(["brand"], adapter_weights=[float(config["lora_scale"])])

    control_image = Image.open(control_image_path).convert("RGB").resize(
        (int(config["width"]), int(config["height"]))
    )
    generator = torch.Generator(device=config.get("device", "cuda")).manual_seed(int(config["seed"]))
    result = pipeline(
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        image=control_image,
        controlnet_conditioning_scale=float(config["controlnet_conditioning_scale"]),
        guidance_scale=float(config["guidance_scale"]),
        num_inference_steps=int(config["num_inference_steps"]),
        generator=generator,
        width=int(config["width"]),
        height=int(config["height"]),
    ).images[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    run_record = make_run_record(
        config,
        seed=int(config["seed"]),
        base_revision=str(config.get("base_revision", "UNPINNED")),
        controlnet_revision=str(config.get("controlnet_revision", "UNPINNED")),
        lora_revision=str(config.get("lora_revision", "LOCAL_UNPINNED")),
    )
    output_path.with_suffix(".run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_record
