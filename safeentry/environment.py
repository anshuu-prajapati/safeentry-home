"""4-room simulated home (Cyberwave-style digital twin).

`HomeEnvironment` is the local twin used for tests and the demo. `CyberwaveBridge`
mirrors the same interface onto a Cyberwave environment when the SDK and an API
key are present; otherwise everything runs fully offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

ROOMS = ["ENTRY_HALL", "LIVING_ROOM", "BEDROOM", "KITCHEN"]

# floor plan in metres (x, y, w, h) — used by renderer and camera frustums
LAYOUT = {
    "ENTRY_HALL": (0.0, 0.0, 3.0, 4.0),
    "LIVING_ROOM": (3.0, 0.0, 5.0, 4.0),
    "BEDROOM": (0.0, 4.0, 4.0, 4.0),
    "KITCHEN": (4.0, 4.0, 4.0, 4.0),
}
FRONT_DOOR = (0.0, 2.0)


@dataclass
class Person:
    pid: str
    room: str
    role: str = "resident"   # resident | visitor | intruder | responder
    posture: str = "standing"  # standing | lying (fallen)


@dataclass
class HomeEnvironment:
    people: dict = field(default_factory=dict)
    door_locked: bool = True
    door_forced: bool = False
    residents_home: bool = True

    def add(self, p: Person):
        self.people[p.pid] = p

    def move(self, pid: str, room: str):
        assert room in ROOMS or room == "OUTSIDE"
        self.people[pid].room = room

    def remove(self, pid: str):
        self.people.pop(pid, None)

    def occupants(self, room: str) -> list[Person]:
        return [p for p in self.people.values() if p.room == room]

    # ---- simulated camera feed --------------------------------------
    def camera_frame(self, room: str, size=(240, 320)) -> np.ndarray:
        """Render a crude top-down frame. Used by the stand-in detector and the demo."""
        h, w = size
        img = np.full((h, w, 3), 40, np.uint8)
        for i, p in enumerate(self.occupants(room)):
            x = 40 + (i * 90) % (w - 80)
            col = {"resident": (80, 200, 80), "intruder": (60, 60, 230),
                   "visitor": (200, 180, 60), "responder": (230, 120, 40)}[p.role]
            if p.posture == "lying":
                img[h - 60:h - 30, x:x + 70] = col
            else:
                img[60:h - 20, x:x + 30] = col
        return img


class CyberwaveBridge:
    """Thin adapter: pushes state/beacon/occupancy into a Cyberwave twin if available."""

    def __init__(self, env: HomeEnvironment):
        self.env = env
        self.client = None
        key = os.getenv("CYBERWAVE_API_KEY")
        if key:
            try:  # pragma: no cover - optional dependency
                import cyberwave  # type: ignore
                self.client = cyberwave.Cyberwave(api_key=key)
            except Exception:
                self.client = None

    @property
    def online(self) -> bool:
        return self.client is not None

    def publish(self, topic: str, payload: dict):
        if self.client is None:
            return False
        try:  # pragma: no cover
            self.client.publish(topic, payload)
            return True
        except Exception:
            return False
