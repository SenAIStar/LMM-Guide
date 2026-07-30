from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import Action, Telemetry


class MissionState(str, Enum):
    IDLE = "idle"
    AIRBORNE = "airborne"
    HOLDING = "holding"
    RETURNING = "returning"
    LANDED = "landed"
    ABORTED = "aborted"


@dataclass
class MissionStateMachine:
    state: MissionState = MissionState.IDLE

    def sync_from_telemetry(self, telemetry: Telemetry) -> MissionState:
        if self.state is not MissionState.IDLE or not telemetry.in_air:
            return self.state
        mode = telemetry.flight_mode.strip().upper()
        if mode in {"RTL", "RETURN", "RETURN_TO_LAUNCH"}:
            self.state = MissionState.RETURNING
        elif mode in {"HOLD", "LOITER", "POSCTL"}:
            self.state = MissionState.HOLDING
        else:
            self.state = MissionState.AIRBORNE
        return self.state

    def next_state(self, action: Action) -> MissionState:
        transitions = {
            (MissionState.IDLE, Action.TAKEOFF): MissionState.AIRBORNE,
            (MissionState.AIRBORNE, Action.GOTO): MissionState.AIRBORNE,
            (MissionState.AIRBORNE, Action.INSPECT): MissionState.AIRBORNE,
            (MissionState.AIRBORNE, Action.HOLD): MissionState.HOLDING,
            (MissionState.HOLDING, Action.GOTO): MissionState.AIRBORNE,
            (MissionState.HOLDING, Action.RTL): MissionState.RETURNING,
            (MissionState.AIRBORNE, Action.RTL): MissionState.RETURNING,
            (MissionState.AIRBORNE, Action.LAND): MissionState.LANDED,
            (MissionState.HOLDING, Action.LAND): MissionState.LANDED,
            (MissionState.RETURNING, Action.LAND): MissionState.LANDED,
        }
        key = (self.state, action)
        if key not in transitions:
            raise ValueError(f"illegal transition: {self.state.value} -> {action.value}")
        return transitions[key]

    def accept(self, action: Action) -> MissionState:
        self.state = self.next_state(action)
        return self.state

    def abort(self) -> MissionState:
        self.state = MissionState.ABORTED
        return self.state
