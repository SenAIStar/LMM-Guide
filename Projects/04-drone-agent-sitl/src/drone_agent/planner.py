from __future__ import annotations

import json
from dataclasses import dataclass

from .contracts import MissionPlan


@dataclass(frozen=True)
class Observation:
    observation_id: str
    image_sha256: str
    telemetry_timestamp_ms: int
    mission_intent: str


def parse_plan_json(payload: str) -> MissionPlan:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("planner output must be a JSON object")
    return MissionPlan.from_dict(raw)


def planner_prompt(observation: Observation) -> str:
    return (
        "Return one drone-plan.v1 JSON object. Use only takeoff, goto, inspect, "
        "hold, rtl, and land. Never emit actuator, attitude-rate, PWM, shell, or "
        f"MAVLink commands. observation_id={observation.observation_id}; "
        f"image_sha256={observation.image_sha256}; "
        f"telemetry_timestamp_ms={observation.telemetry_timestamp_ms}; "
        f"mission_intent={observation.mission_intent}"
    )
