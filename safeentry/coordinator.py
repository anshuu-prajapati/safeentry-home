"""SafeEntry coordinator — wires sentinel, entry guard, camera responder and
beacon to the unified state machine."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .audio import SoundClass
from .calls import analyse_transcript
from .environment import CyberwaveBridge, HomeEnvironment
from .states import Event, EmergencyType, StateMachine, SystemState
from .subsystems import DoorPlateBeacon, DoorPlateSentinel, EntryGuard, IndoorCameraResponder

WATCH_TIMEOUT_S = 30.0


class SafeEntry:
    def __init__(self, env: HomeEnvironment | None = None, detector=None,
                 clock: Callable[[], float] = time.monotonic):
        self.env = env or HomeEnvironment()
        self.clock = clock
        self.sm = StateMachine(clock)
        self.sentinel = DoorPlateSentinel()
        self.guard = EntryGuard(self.env)
        self.cameras = IndoorCameraResponder(self.env, detector)
        self.beacon = DoorPlateBeacon()
        self.bridge = CyberwaveBridge(self.env)
        self.events: list[Event] = []
        self.dispatch: dict | None = None
        self.sm.on_change(self._on_change)

    # ------------------------------------------------------------ helpers
    def log(self, source, kind, **detail):
        e = Event(self.clock(), source, kind, detail)
        self.events.append(e)
        self.bridge.publish(f"safeentry/{source}", e.as_dict())
        return e

    def _on_change(self, old, new, emergency):
        self.log("state", "transition", frm=old.value, to=new.value, emergency=emergency.value,
                 beacon=self.beacon.describe(new, emergency))

    def occupancy(self, frames=None) -> dict:
        dets = self.cameras.sweep(frames)
        occ = {d.room: {"persons": d.persons, "lying": d.lying, "detector": d.source} for d in dets}
        self.log("cameras", "sweep", occupancy=occ)
        return occ

    def _build_dispatch(self, emergency: EmergencyType, reason: str, frames=None):
        occ = self.occupancy(frames)
        priority = sorted(occ, key=lambda r: (-occ[r]["lying"], -occ[r]["persons"]))
        self.dispatch = {
            "emergency": emergency.value,
            "reason": reason,
            "occupied_rooms": [r for r in priority if occ[r]["persons"]],
            "priority_room": priority[0] if occ[priority[0]]["persons"] else None,
            "occupancy": occ,
            "beacon": self.beacon.describe(SystemState.EMERGENCY, emergency),
            "door": "unlocked for responders" if emergency == EmergencyType.MEDICAL else "locked",
        }
        self.log("backend", "dispatch", **self.dispatch)
        return self.dispatch

    # ------------------------------------------------------------ inputs
    def on_audio(self, clip: np.ndarray, frames=None) -> dict:
        c = self.sentinel.listen(clip)
        self.log("sentinel", "sound", label=c.label.value, confidence=round(c.confidence, 2))
        if c.label == SoundClass.KNOCK:
            if self.sm.escalate(SystemState.WATCH, "knock at door"):
                self.log("guard", "announce", **self.guard.announce("visitor"))
        elif c.label in (SoundClass.FORCED_ENTRY, SoundClass.GLASS_BREAK):
            self.env.door_forced = c.label == SoundClass.FORCED_ENTRY
            if self.sm.escalate(SystemState.INTRUSION, c.label.value.lower()):
                self.guard.lock()
                self.log("guard", "announce", **self.guard.announce("deterrent"))
                self.check_intrusion(frames)
        elif c.label == SoundClass.SOS_KNOCK:
            self.trigger_sos(EmergencyType.POLICE, "SOS knock code at door", frames)
        return c.as_dict()

    def check_intrusion(self, frames=None):
        """In INTRUSION, sweep cameras. Someone inside while residents are away
        (or a non-resident present) escalates to a POLICE emergency."""
        if self.sm.state != SystemState.INTRUSION:
            return None
        occ = self.occupancy(frames)
        inside = sum(v["persons"] for v in occ.values())
        residents = sum(1 for p in self.env.people.values() if p.role == "resident" and p.room != "OUTSIDE")
        if inside > residents:
            self.trigger_sos(EmergencyType.POLICE, f"{inside - residents} unknown person(s) inside after forced entry", frames)
        return occ

    def trigger_sos(self, emergency: EmergencyType, reason: str, frames=None) -> dict:
        self.sm.escalate(SystemState.EMERGENCY, reason, emergency)
        if emergency == EmergencyType.MEDICAL:
            self.guard.unlock_for_responders()
            self.log("guard", "announce", **self.guard.announce("responder"))
        else:
            self.guard.lock()
        return self._build_dispatch(emergency, reason, frames)

    def on_call_transcript(self, text: str, frames=None) -> dict:
        v = analyse_transcript(text)
        self.log("calls", "transcript", **v.as_dict())
        if v.distress:
            self.trigger_sos(v.emergency, "distress detected in call: " + ", ".join(v.matches), frames)
        return v.as_dict()

    def tick(self):
        if self.sm.state == SystemState.WATCH and self.sm.time_in_state() > WATCH_TIMEOUT_S:
            self.sm.relax("no follow-up after knock")
            self.guard.playing = None

    def resolve(self, who="owner"):
        self.sm.resolve(f"resolved by {who}")
        self.guard.lock()
        self.guard.playing = None
        self.env.door_forced = False
        self.dispatch = None

    def beacon_rgb(self, t: float | None = None):
        return self.beacon.pattern(self.sm.state, self.sm.emergency, self.clock() if t is None else t)

    def snapshot(self) -> dict:
        l, r = self.beacon_rgb()
        return {
            "state": self.sm.state.value, "emergency": self.sm.emergency.value,
            "beacon": {"left": l, "right": r, "pattern": self.beacon.describe(self.sm.state, self.sm.emergency)},
            "door": {"locked": self.env.door_locked, "forced": self.env.door_forced},
            "guard_audio": self.guard.playing,
            "people": [p.__dict__ for p in self.env.people.values()],
            "dispatch": self.dispatch,
            "cyberwave_online": self.bridge.online,
            "events": [e.as_dict() for e in self.events[-25:]],
        }
