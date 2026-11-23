"""Unified SafeEntry state machine: IDLE -> WATCH -> INTRUSION -> EMERGENCY."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class SystemState(str, Enum):
    IDLE = "IDLE"            # quiet, door plate dim
    WATCH = "WATCH"          # something at the door (knock / visitor), heightened attention
    INTRUSION = "INTRUSION"  # forced entry / glass break detected, deterrent active
    EMERGENCY = "EMERGENCY"  # SOS confirmed, beacon on, responders being guided


class EmergencyType(str, Enum):
    NONE = "NONE"
    MEDICAL = "MEDICAL"   # blinking red beacon
    POLICE = "POLICE"     # split red / blue beacon
    FIRE = "FIRE"         # orange / white beacon


# Allowed transitions. EMERGENCY can only be left via an explicit resolve().
ALLOWED = {
    SystemState.IDLE: {SystemState.WATCH, SystemState.INTRUSION, SystemState.EMERGENCY},
    SystemState.WATCH: {SystemState.IDLE, SystemState.INTRUSION, SystemState.EMERGENCY},
    SystemState.INTRUSION: {SystemState.EMERGENCY, SystemState.IDLE},
    SystemState.EMERGENCY: {SystemState.EMERGENCY},  # upgrade type only
}

# Severity ordering: the machine never silently de-escalates.
SEVERITY = {SystemState.IDLE: 0, SystemState.WATCH: 1, SystemState.INTRUSION: 2, SystemState.EMERGENCY: 3}


@dataclass
class Event:
    t: float
    source: str
    kind: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"t": round(self.t, 3), "source": self.source, "kind": self.kind, "detail": self.detail}


class InvalidTransition(Exception):
    pass


class StateMachine:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.state = SystemState.IDLE
        self.emergency = EmergencyType.NONE
        self.entered_at = clock()
        self.history: list[Event] = []
        self._listeners: list[Callable[[SystemState, SystemState, EmergencyType], None]] = []

    def on_change(self, fn):
        self._listeners.append(fn)
        return fn

    def _set(self, new: SystemState, reason: str, emergency: EmergencyType = EmergencyType.NONE):
        old = self.state
        self.state = new
        self.emergency = emergency if new == SystemState.EMERGENCY else EmergencyType.NONE
        self.entered_at = self.clock()
        self.history.append(Event(self.entered_at, "state", "transition",
                                  {"from": old.value, "to": new.value, "reason": reason,
                                   "emergency": self.emergency.value}))
        for fn in self._listeners:
            fn(old, new, self.emergency)

    def escalate(self, target: SystemState, reason: str, emergency: EmergencyType = EmergencyType.NONE) -> bool:
        """Move to a state of equal or higher severity. Returns True if state changed."""
        if target == SystemState.EMERGENCY and emergency == EmergencyType.NONE:
            raise InvalidTransition("EMERGENCY requires an EmergencyType")
        if target == self.state:
            if target == SystemState.EMERGENCY and emergency != self.emergency:
                # Allow upgrading the emergency type (e.g. MEDICAL -> POLICE if intruder found)
                self._set(target, reason, emergency)
                return True
            return False
        if SEVERITY[target] < SEVERITY[self.state]:
            return False  # never de-escalate implicitly
        if target not in ALLOWED[self.state]:
            raise InvalidTransition(f"{self.state} -> {target}")
        self._set(target, reason, emergency)
        return True

    def relax(self, reason: str) -> bool:
        """WATCH times out back to IDLE. Only WATCH may auto-relax."""
        if self.state == SystemState.WATCH:
            self._set(SystemState.IDLE, reason)
            return True
        return False

    def resolve(self, reason: str = "resolved by owner/responder") -> None:
        """Explicit, authenticated reset from any state."""
        if self.state != SystemState.IDLE:
            self._set(SystemState.IDLE, reason)

    def time_in_state(self) -> float:
        return self.clock() - self.entered_at
