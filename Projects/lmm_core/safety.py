from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reasons: tuple[str, ...]


class FlightSafetyGate:
    """A deterministic guard that must run after the model planner."""

    def __init__(self, max_altitude_m: float = 30.0, min_battery_pct: float = 25.0) -> None:
        self.max_altitude_m = max_altitude_m
        self.min_battery_pct = min_battery_pct

    def evaluate(self, command: dict[str, Any], telemetry: dict[str, Any]) -> SafetyDecision:
        reasons: list[str] = []
        action = command.get("action")
        if action not in {"arm", "takeoff", "goto", "land", "hold", "return_to_launch"}:
            reasons.append("unsupported action")
        altitude = command.get("altitude_m", 0.0)
        if not isinstance(altitude, (int, float)) or altitude < 0 or altitude > self.max_altitude_m:
            reasons.append("altitude outside policy")
        battery = telemetry.get("battery_pct")
        if not isinstance(battery, (int, float)) or battery < self.min_battery_pct:
            reasons.append("battery below policy")
        if telemetry.get("gps_fix") is not True and action in {"takeoff", "goto", "return_to_launch"}:
            reasons.append("gps fix required")
        if command.get("human_approved") is not True and action in {"arm", "takeoff", "goto"}:
            reasons.append("human approval required")
        return SafetyDecision(allowed=not reasons, reasons=tuple(reasons))

