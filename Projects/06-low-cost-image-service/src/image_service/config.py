from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_service_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "config_revision", "result_status", "model", "admission", "batching", "cache", "pricing"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"missing service config keys: {sorted(missing)}")
    if config["result_status"] != "not_measured":
        raise ValueError("the checked-in reference config must remain not_measured")
    model = config["model"]
    if not model.get("model_revision") or str(model["model_revision"]).lower() in {"main", "latest"}:
        raise ValueError("model_revision must be pinned; floating main/latest is not allowed")
    if model.get("compile_unet") and model.get("lora_mode") == "dynamic":
        raise ValueError("compile_unet cannot be combined with dynamic LoRA loading in the reference backend")
    batching = config["batching"]
    if int(batching["preferred_batch_size"]) > int(batching["max_batch_size"]):
        raise ValueError("preferred_batch_size cannot exceed max_batch_size")
    if int(batching["max_delay_ms"]) < 0 or int(batching["oom_split_retries"]) < 0:
        raise ValueError("batch delay and retry count must be non-negative")
    pricing = config["pricing"]
    if pricing.get("gpu_hour_price") is not None and float(pricing["gpu_hour_price"]) <= 0:
        raise ValueError("gpu_hour_price must be positive when set")

