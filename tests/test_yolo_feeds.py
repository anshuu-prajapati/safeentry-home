"""YOLOv8 person detection on camera-feed stand-ins (skipped if ultralytics absent).

Uses the sample photos that ship inside the ultralytics package as simulated
room camera frames, and a blank frame for empty rooms.
"""
import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")

from ultralytics.utils import ASSETS  # noqa: E402

from safeentry.subsystems import YoloDetector  # noqa: E402


@pytest.fixture(scope="module")
def yolo():
    return YoloDetector()


def _img(name):
    import cv2
    return cv2.imread(str(ASSETS / name))


def test_yolo_counts_people(yolo):
    n, _ = yolo.count(_img("bus.jpg"))
    assert n >= 3
    assert yolo.count(np.full((480, 640, 3), 110, np.uint8))[0] == 0


def test_occupancy_sweep_with_yolo_feeds(yolo):
    from safeentry.coordinator import SafeEntry
    from safeentry.environment import HomeEnvironment
    from safeentry.states import EmergencyType
    se = SafeEntry(HomeEnvironment(), detector=yolo)
    empty = np.full((480, 640, 3), 110, np.uint8)
    frames = {"ENTRY_HALL": empty, "LIVING_ROOM": _img("bus.jpg"), "BEDROOM": _img("bus.jpg")[380:1000, :400], "KITCHEN": empty}
    d = se.trigger_sos(EmergencyType.POLICE, "test", frames)
    assert set(d["occupied_rooms"]) == {"LIVING_ROOM", "BEDROOM"}
    assert d["occupancy"]["KITCHEN"]["detector"] == "yolov8n"
