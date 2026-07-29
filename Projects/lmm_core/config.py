import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = {
    "project_id",
    "status",
    "model_stack",
    "data_contract",
    "evaluation",
    "acceptance_targets",
}


def load_project_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_project_config(config)
    return config


def validate_project_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - config.keys()
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if config["status"] not in {"scaffold", "validated", "trained"}:
        raise ValueError("status must be scaffold, validated, or trained")
    if not isinstance(config["model_stack"], list) or not config["model_stack"]:
        raise ValueError("model_stack must be a non-empty list")
    if not isinstance(config["evaluation"], list) or not config["evaluation"]:
        raise ValueError("evaluation must be a non-empty list")
    targets = config["acceptance_targets"]
    if not isinstance(targets, dict) or not targets:
        raise ValueError("acceptance_targets must be a non-empty object")
    for name, value in targets.items():
        if not isinstance(value, dict) or "operator" not in value or "value" not in value:
            raise ValueError(f"invalid acceptance target: {name}")
        if value["operator"] not in {">=", "<=", "=="}:
            raise ValueError(f"invalid operator for {name}")
        if not isinstance(value["value"], (int, float)):
            raise ValueError(f"target value must be numeric: {name}")

