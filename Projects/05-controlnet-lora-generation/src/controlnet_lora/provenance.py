from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_run_record(
    config: dict[str, Any],
    *,
    seed: int,
    base_revision: str,
    controlnet_revision: str,
    lora_revision: str,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": canonical_hash(config),
        "seed": seed,
        "base_revision": base_revision,
        "controlnet_revision": controlnet_revision,
        "lora_revision": lora_revision,
        "python": platform.python_version(),
        "result_status": "generated_not_evaluated",
    }
