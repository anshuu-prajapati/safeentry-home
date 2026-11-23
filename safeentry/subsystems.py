"""The three linked subsystems + the door-plate beacon."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio import SR, AudioClassifier, Classification, SoundClass
from .environment import ROOMS, HomeEnvironment
from .states import EmergencyType, SystemState


# ------------------------------------------------------------- 1. sentinel
class DoorPlateSentinel:
    """Acoustic door plate: listens to the door surface + porch mic."""

    def __init__(self, classifier: AudioClassifier | None = None, min_conf: float = 0.6):
        self.clf = classifier or AudioClassifier()
        self.min_conf = min_conf

    def listen(self, clip: np.ndarray) -> Classification:
        c = self.clf.classify(clip)
        if c.confidence < self.min_conf:
            c.label = SoundClass.AMBIENT
        return c


# ---------------------------------------------------------- 2. entry guard
class EntryGuard:
    """Smart lock + outward speaker. Plays deterrent audio on intrusion."""

    MESSAGES = {
        "visitor": "Hello! The residents have been notified. Please wait.",
        "deterrent": "Warning. You are being recorded. Police have been alerted.",
        "responder": "Emergency responders: door unlocking. Patient location shown on the plate.",
    }

    def __init__(self, env: HomeEnvironment):
        self.env = env
        self.playing: str | None = None

    def announce(self, kind: str) -> dict:
        self.playing = kind
        return {"message": kind, "text": self.MESSAGES[kind]}

    def siren(self, dur=1.0) -> np.ndarray:
        t = np.arange(int(dur * SR)) / SR
        f = 900 + 500 * np.sin(2 * np.pi * 2 * t)  # 2 Hz wail
        return 0.8 * np.sin(2 * np.pi * np.cumsum(f) / SR)

    def lock(self):
        self.env.door_locked = True

    def unlock_for_responders(self):
        self.env.door_locked = False


# ------------------------------------------------------ 3. camera responder
@dataclass
class Detection:
    room: str
    persons: int
    lying: int
    source: str


class StandInDetector:
    """Deterministic detector for the simulated top-down feed (colour blobs)."""
    name = "stand-in"

    def count(self, frame: np.ndarray) -> tuple[int, int]:
        mask = (frame.max(axis=2) > 70)
        cols = mask.any(axis=0)
        # count contiguous column runs = people; wide-short blobs = lying
        persons = lying = 0
        x, w = 0, frame.shape[1]
        while x < w:
            if cols[x]:
                s = x
                while x < w and cols[x]:
                    x += 1
                persons += 1
                rows = mask[:, s:x].any(axis=1).sum()
                if (x - s) > rows:
                    lying += 1
            x += 1
        return persons, lying


class YoloDetector:
    """YOLOv8 person detector (COCO class 0). Used on real/rendered RGB feeds."""
    name = "yolov8n"

    def __init__(self, weights="yolov8n.pt", conf=0.4):
        from ultralytics import YOLO  # lazy import, optional dependency
        self.model = YOLO(weights)
        self.conf = conf

    def boxes(self, frame: np.ndarray):
        r = self.model(frame, classes=[0], conf=self.conf, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy().tolist()

    def count(self, frame: np.ndarray) -> tuple[int, int]:
        bxs = self.boxes(frame)
        lying = sum(1 for x1, y1, x2, y2 in bxs if (x2 - x1) > 1.3 * (y2 - y1))
        return len(bxs), lying


class IndoorCameraResponder:
    def __init__(self, env: HomeEnvironment, detector=None):
        self.env = env
        self.detector = detector or StandInDetector()

    def sweep(self, frames: dict | None = None) -> list[Detection]:
        out = []
        for room in ROOMS:
            frame = frames[room] if frames and room in frames else self.env.camera_frame(room)
            n, lying = self.detector.count(frame)
            out.append(Detection(room, n, lying, self.detector.name))
        return out


# ------------------------------------------------------------- beacon
OFF, WHITE, AMBER, RED, BLUE, ORANGE = (0, 0, 0), (255, 255, 255), (255, 170, 0), (255, 0, 0), (0, 80, 255), (255, 110, 0)


class DoorPlateBeacon:
    """Returns (left_half_rgb, right_half_rgb) for time t (seconds).

    MEDICAL  : whole plate blinks red at 2 Hz
    POLICE   : split plate, red|blue alternating at 2.5 Hz
    FIRE     : orange/white alternate at 3 Hz
    INTRUSION: rapid red/white strobe (deterrent, 5 Hz)
    WATCH    : soft amber breathing
    IDLE     : dim white house-number backlight
    """

    def pattern(self, state: SystemState, emergency: EmergencyType, t: float):
        if state == SystemState.EMERGENCY:
            if emergency == EmergencyType.MEDICAL:
                on = (t * 2) % 1 < 0.5
                c = RED if on else OFF
                return c, c
            if emergency == EmergencyType.POLICE:
                phase = (t * 2.5) % 1 < 0.5
                return (RED, OFF) if phase else (OFF, BLUE)
            if emergency == EmergencyType.FIRE:
                c = ORANGE if (t * 3) % 1 < 0.5 else WHITE
                return c, c
        if state == SystemState.INTRUSION:
            c = RED if (t * 5) % 1 < 0.5 else WHITE
            return c, c
        if state == SystemState.WATCH:
            k = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t))
            c = tuple(int(v * k) for v in AMBER)
            return c, c
        dim = (60, 60, 60)
        return dim, dim

    @staticmethod
    def describe(state: SystemState, emergency: EmergencyType) -> str:
        if state == SystemState.EMERGENCY:
            return {"MEDICAL": "blinking red (2 Hz)", "POLICE": "split red | blue (2.5 Hz)",
                    "FIRE": "orange/white (3 Hz)"}[emergency.value]
        return {"INTRUSION": "red/white strobe (5 Hz)", "WATCH": "amber breathing",
                "IDLE": "dim backlight"}[state.value]
