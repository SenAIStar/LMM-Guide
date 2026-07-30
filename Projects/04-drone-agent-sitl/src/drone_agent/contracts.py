from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContractError(ValueError):
    pass


class Action(str, Enum):
    TAKEOFF = "takeoff"
    GOTO = "goto"
    INSPECT = "inspect"
    HOLD = "hold"
    RTL = "rtl"
    LAND = "land"


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    action: Action
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    relative_altitude_m: float | None = None
    speed_m_s: float | None = None
    timeout_s: float = 30.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanStep":
        if not isinstance(raw, dict):
            raise ContractError("step must be an object")
        allowed = {
            "step_id",
            "action",
            "latitude_deg",
            "longitude_deg",
            "relative_altitude_m",
            "speed_m_s",
            "timeout_s",
        }
        extra = set(raw) - allowed
        if extra:
            raise ContractError(f"unexpected step fields: {sorted(extra)}")
        try:
            step_id = raw["step_id"]
            action = raw["action"]
            if not isinstance(step_id, str) or not isinstance(action, str):
                raise ContractError("step_id and action must be strings")
            step = cls(
                step_id=step_id,
                action=Action(action),
                latitude_deg=_optional_float(raw.get("latitude_deg"), "latitude_deg"),
                longitude_deg=_optional_float(raw.get("longitude_deg"), "longitude_deg"),
                relative_altitude_m=_optional_float(
                    raw.get("relative_altitude_m"), "relative_altitude_m"
                ),
                speed_m_s=_optional_float(raw.get("speed_m_s"), "speed_m_s"),
                timeout_s=_required_float(raw.get("timeout_s", 30.0), "timeout_s"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"invalid step: {exc}") from exc
        step.validate()
        return step

    def validate(self) -> None:
        if not self.step_id.strip():
            raise ContractError("step_id must not be empty")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ContractError("timeout_s must be positive")
        if self.action in {Action.GOTO, Action.INSPECT}:
            if self.latitude_deg is None or self.longitude_deg is None:
                raise ContractError(f"{self.action.value} requires latitude and longitude")
        if self.action in {Action.TAKEOFF, Action.GOTO, Action.INSPECT}:
            if self.relative_altitude_m is None:
                raise ContractError(f"{self.action.value} requires relative_altitude_m")
        if self.latitude_deg is not None and not -90 <= self.latitude_deg <= 90:
            raise ContractError("latitude_deg out of range")
        if self.longitude_deg is not None and not -180 <= self.longitude_deg <= 180:
            raise ContractError("longitude_deg out of range")


@dataclass(frozen=True)
class MissionPlan:
    schema_version: str
    mission_id: str
    observation_id: str
    telemetry_timestamp_ms: int
    steps: tuple[PlanStep, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MissionPlan":
        if not isinstance(raw, dict):
            raise ContractError("mission plan must be an object")
        allowed = {
            "schema_version",
            "mission_id",
            "observation_id",
            "telemetry_timestamp_ms",
            "steps",
        }
        extra = set(raw) - allowed
        if extra:
            raise ContractError(f"unexpected plan fields: {sorted(extra)}")
        try:
            for field_name in ("schema_version", "mission_id", "observation_id"):
                if not isinstance(raw[field_name], str):
                    raise ContractError(f"{field_name} must be a string")
            timestamp_ms = raw["telemetry_timestamp_ms"]
            if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
                raise ContractError("telemetry_timestamp_ms must be an integer")
            if not isinstance(raw["steps"], list):
                raise ContractError("steps must be a list")
            plan = cls(
                schema_version=raw["schema_version"],
                mission_id=raw["mission_id"],
                observation_id=raw["observation_id"],
                telemetry_timestamp_ms=timestamp_ms,
                steps=tuple(PlanStep.from_dict(item) for item in raw["steps"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ContractError):
                raise
            raise ContractError(f"invalid mission plan: {exc}") from exc
        if plan.schema_version != "drone-plan.v1":
            raise ContractError("unsupported schema_version")
        if not plan.mission_id or not plan.observation_id:
            raise ContractError("mission_id and observation_id are required")
        if plan.telemetry_timestamp_ms < 0:
            raise ContractError("telemetry_timestamp_ms must be non-negative")
        if not 1 <= len(plan.steps) <= 16:
            raise ContractError("steps must contain 1 to 16 items")
        step_ids = [step.step_id for step in plan.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ContractError("step_id values must be unique")
        return plan


@dataclass(frozen=True)
class Telemetry:
    timestamp_ms: int
    connected: bool
    armed: bool
    in_air: bool
    battery_remaining: float
    position_valid: bool
    latitude_deg: float
    longitude_deg: float
    absolute_altitude_m: float
    home_absolute_altitude_m: float
    flight_mode: str = "HOLD"

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ms, bool) or not isinstance(self.timestamp_ms, int):
            raise ContractError("timestamp_ms must be an integer")
        if self.timestamp_ms < 0:
            raise ContractError("timestamp_ms must be non-negative")
        for name in ("connected", "armed", "in_air", "position_valid"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"{name} must be a boolean")
        numeric_fields = (
            "battery_remaining",
            "latitude_deg",
            "longitude_deg",
            "absolute_altitude_m",
            "home_absolute_altitude_m",
        )
        for name in numeric_fields:
            _required_float(getattr(self, name), name)
        if not 0.0 <= self.battery_remaining <= 1.0:
            raise ContractError("battery_remaining must be between 0 and 1")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ContractError("latitude_deg out of range")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ContractError("longitude_deg out of range")
        if not isinstance(self.flight_mode, str) or not self.flight_mode.strip():
            raise ContractError("flight_mode must be a non-empty string")

    @property
    def relative_altitude_m(self) -> float:
        return self.absolute_altitude_m - self.home_absolute_altitude_m


@dataclass(frozen=True)
class Geofence:
    center_latitude_deg: float
    center_longitude_deg: float
    radius_m: float
    max_relative_altitude_m: float

    def __post_init__(self) -> None:
        if not -90.0 <= _required_float(self.center_latitude_deg, "center_latitude_deg") <= 90.0:
            raise ContractError("center_latitude_deg out of range")
        if not -180.0 <= _required_float(
            self.center_longitude_deg, "center_longitude_deg"
        ) <= 180.0:
            raise ContractError("center_longitude_deg out of range")
        if _required_float(self.radius_m, "radius_m") <= 0:
            raise ContractError("radius_m must be positive")
        if _required_float(self.max_relative_altitude_m, "max_relative_altitude_m") <= 0:
            raise ContractError("max_relative_altitude_m must be positive")


@dataclass(frozen=True)
class SafetyPolicy:
    geofence: Geofence
    max_telemetry_age_ms: int = 1500
    min_motion_battery: float = 0.25
    max_speed_m_s: float = 5.0
    require_takeoff_approval: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_telemetry_age_ms, bool) or not isinstance(
            self.max_telemetry_age_ms, int
        ):
            raise ContractError("max_telemetry_age_ms must be an integer")
        if self.max_telemetry_age_ms <= 0:
            raise ContractError("max_telemetry_age_ms must be positive")
        if not 0.0 <= _required_float(self.min_motion_battery, "min_motion_battery") <= 1.0:
            raise ContractError("min_motion_battery must be between 0 and 1")
        if _required_float(self.max_speed_m_s, "max_speed_m_s") <= 0:
            raise ContractError("max_speed_m_s must be positive")
        if not isinstance(self.require_takeoff_approval, bool):
            raise ContractError("require_takeoff_approval must be a boolean")


def _required_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ContractError(f"{name} must be finite")
    return parsed


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _required_float(value, name)
