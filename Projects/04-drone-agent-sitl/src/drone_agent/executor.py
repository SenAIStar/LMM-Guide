from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import Action, MissionPlan, PlanStep, Telemetry
from .safety import SafetyGate
from .state_machine import MissionStateMachine


class VehiclePort(Protocol):
    async def takeoff(self, relative_altitude_m: float) -> None: ...
    async def goto(
        self,
        latitude_deg: float,
        longitude_deg: float,
        absolute_altitude_m: float,
    ) -> None: ...
    async def hold(self) -> None: ...
    async def rtl(self) -> None: ...
    async def land(self) -> None: ...


@dataclass
class TraceVehicle:
    calls: list[str] = field(default_factory=list)

    async def takeoff(self, relative_altitude_m: float) -> None:
        self.calls.append(f"takeoff:{relative_altitude_m:.1f}")

    async def goto(self, latitude_deg: float, longitude_deg: float, absolute_altitude_m: float) -> None:
        self.calls.append(f"goto:{latitude_deg:.6f},{longitude_deg:.6f},{absolute_altitude_m:.1f}")

    async def hold(self) -> None:
        self.calls.append("hold")

    async def rtl(self) -> None:
        self.calls.append("rtl")

    async def land(self) -> None:
        self.calls.append("land")


@dataclass(frozen=True)
class ExecutionResult:
    step_id: str
    status: str
    reason_code: str


class MissionExecutor:
    def __init__(self, vehicle: VehiclePort, gate: SafetyGate) -> None:
        self.vehicle = vehicle
        self.gate = gate
        self.state_machine = MissionStateMachine()
        self.executed_step_ids: set[str] = set()

    async def execute_step(
        self,
        plan: MissionPlan,
        step: PlanStep,
        telemetry: Telemetry,
        *,
        now_ms: int,
        human_approved: bool,
    ) -> ExecutionResult:
        decision = self.gate.evaluate(
            plan,
            step,
            telemetry,
            now_ms=now_ms,
            human_approved=human_approved,
            executed_step_ids=frozenset(self.executed_step_ids),
        )
        if not decision.allowed:
            return ExecutionResult(step.step_id, "blocked", decision.reason_code)
        try:
            if not self.executed_step_ids:
                self.state_machine.sync_from_telemetry(telemetry)
            self.state_machine.next_state(step.action)
            await asyncio.wait_for(self._dispatch(step, telemetry), timeout=step.timeout_s)
            self.state_machine.accept(step.action)
            self.executed_step_ids.add(step.step_id)
            return ExecutionResult(step.step_id, "completed", "allowed")
        except Exception:
            self.state_machine.abort()
            try:
                await self.vehicle.hold()
            except Exception:
                pass
            return ExecutionResult(step.step_id, "failed", "executor_error")

    async def _dispatch(self, step: PlanStep, telemetry: Telemetry) -> None:
        if step.action is Action.TAKEOFF:
            await self.vehicle.takeoff(step.relative_altitude_m or 0.0)
        elif step.action in {Action.GOTO, Action.INSPECT}:
            absolute_altitude_m = telemetry.home_absolute_altitude_m + (step.relative_altitude_m or 0.0)
            await self.vehicle.goto(
                step.latitude_deg or 0.0,
                step.longitude_deg or 0.0,
                absolute_altitude_m,
            )
        elif step.action is Action.HOLD:
            await self.vehicle.hold()
        elif step.action is Action.RTL:
            await self.vehicle.rtl()
        elif step.action is Action.LAND:
            await self.vehicle.land()
