from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import Action, MissionPlan, PlanStep, SafetyPolicy, Telemetry


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason_code: str


class SafetyGate:
    def __init__(self, policy: SafetyPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        plan: MissionPlan,
        step: PlanStep,
        telemetry: Telemetry,
        *,
        now_ms: int,
        human_approved: bool,
        executed_step_ids: frozenset[str] = frozenset(),
    ) -> Decision:
        if step.step_id in executed_step_ids:
            return Decision(False, "duplicate_step")
        if not telemetry.connected:
            return Decision(False, "vehicle_disconnected")
        if now_ms < telemetry.timestamp_ms:
            return Decision(False, "telemetry_from_future")
        if now_ms - telemetry.timestamp_ms > self.policy.max_telemetry_age_ms:
            return Decision(False, "stale_telemetry")
        if plan.telemetry_timestamp_ms != telemetry.timestamp_ms:
            return Decision(False, "plan_telemetry_mismatch")
        if step.action is Action.TAKEOFF and self.policy.require_takeoff_approval and not human_approved:
            return Decision(False, "approval_required")
        if step.action in {Action.TAKEOFF, Action.GOTO, Action.INSPECT}:
            if telemetry.battery_remaining < self.policy.min_motion_battery:
                return Decision(False, "battery_too_low")
            if not telemetry.position_valid:
                return Decision(False, "position_invalid")
        if step.speed_m_s is not None:
            if step.speed_m_s <= 0 or step.speed_m_s > self.policy.max_speed_m_s:
                return Decision(False, "speed_out_of_policy")
        if step.relative_altitude_m is not None:
            if not 0 < step.relative_altitude_m <= self.policy.geofence.max_relative_altitude_m:
                return Decision(False, "altitude_out_of_policy")
        if step.action in {Action.GOTO, Action.INSPECT}:
            distance_m = haversine_m(
                self.policy.geofence.center_latitude_deg,
                self.policy.geofence.center_longitude_deg,
                step.latitude_deg or 0.0,
                step.longitude_deg or 0.0,
            )
            if distance_m > self.policy.geofence.radius_m:
                return Decision(False, "outside_geofence")
        if telemetry.battery_remaining < self.policy.min_motion_battery:
            if step.action not in {Action.HOLD, Action.RTL, Action.LAND}:
                return Decision(False, "recovery_action_required")
        return Decision(True, "allowed")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
