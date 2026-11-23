import pytest
from fastapi.testclient import TestClient

from api.main import app
from safeentry.audio import SYNTH, AudioClassifier, SoundClass, from_wav_bytes, to_wav_bytes
from safeentry.calls import analyse_transcript
from safeentry.scenarios import SCENARIOS, check
from safeentry.states import EmergencyType, InvalidTransition, StateMachine, SystemState
from safeentry.subsystems import BLUE, OFF, RED, DoorPlateBeacon


# ---------------------------------------------------------------- state machine
def test_escalation_and_no_implicit_deescalation():
    sm = StateMachine(clock=lambda: 0)
    assert sm.escalate(SystemState.WATCH, "knock")
    assert sm.escalate(SystemState.INTRUSION, "bang")
    assert not sm.escalate(SystemState.WATCH, "knock again")  # never goes down
    assert sm.state == SystemState.INTRUSION
    sm.escalate(SystemState.EMERGENCY, "sos", EmergencyType.MEDICAL)
    assert sm.escalate(SystemState.EMERGENCY, "intruder found", EmergencyType.POLICE)
    assert sm.emergency == EmergencyType.POLICE
    sm.resolve()
    assert sm.state == SystemState.IDLE and sm.emergency == EmergencyType.NONE


def test_emergency_requires_type():
    with pytest.raises(InvalidTransition):
        StateMachine().escalate(SystemState.EMERGENCY, "x")


def test_only_watch_relaxes():
    sm = StateMachine(clock=lambda: 0)
    sm.escalate(SystemState.INTRUSION, "bang")
    assert not sm.relax("timeout")


# ---------------------------------------------------------------- audio AI
@pytest.mark.parametrize("label", list(SoundClass))
def test_audio_classifier_accuracy(label):
    clf = AudioClassifier()
    hits = sum(clf.classify(SYNTH[label](seed=s)).label == label for s in range(50))
    assert hits >= 48, f"{label}: {hits}/50"


def test_wav_roundtrip():
    x = SYNTH[SoundClass.KNOCK](seed=1)
    assert AudioClassifier().classify(from_wav_bytes(to_wav_bytes(x))).label == SoundClass.KNOCK


# ---------------------------------------------------------------- calls
@pytest.mark.parametrize("text,distress,etype", [
    ("Papa fell down and is unconscious, send an ambulance", True, "MEDICAL"),
    ("someone is breaking in, call the police!", True, "POLICE"),
    ("bachao! aag lagi hai", True, "FIRE"),
    ("no need for an ambulance, he's fine", False, "NONE"),
    ("can you pick up milk on the way home", False, "NONE"),
])
def test_call_detection(text, distress, etype):
    v = analyse_transcript(text)
    assert v.distress == distress and v.emergency.value == etype


# ---------------------------------------------------------------- beacon
def test_medical_beacon_blinks_red():
    b = DoorPlateBeacon()
    assert b.pattern(SystemState.EMERGENCY, EmergencyType.MEDICAL, 0.1) == (RED, RED)
    assert b.pattern(SystemState.EMERGENCY, EmergencyType.MEDICAL, 0.3) == (OFF, OFF)


def test_police_beacon_splits_red_blue():
    b = DoorPlateBeacon()
    assert b.pattern(SystemState.EMERGENCY, EmergencyType.POLICE, 0.05) == (RED, OFF)
    assert b.pattern(SystemState.EMERGENCY, EmergencyType.POLICE, 0.25) == (OFF, BLUE)


# ---------------------------------------------------------------- scenarios
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios(name):
    r = check(name)
    assert r["passed"], r


# ---------------------------------------------------------------- API
def test_api_flow():
    c = TestClient(app)
    c.post("/reset")
    assert c.get("/state").json()["state"] == "IDLE"
    assert c.post("/simulate/sound/KNOCK").json()["label"] == "KNOCK"
    assert c.get("/state").json()["state"] == "WATCH"
    d = c.post("/sos", json={"type": "MEDICAL"}).json()
    assert d["emergency"] == "MEDICAL" and "LIVING_ROOM" in d["occupied_rooms"]
    assert c.get("/state").json()["door"]["locked"] is False  # unlocked for paramedics
    v = c.post("/call/detect", json={"transcript": "there is an intruder, police please"}).json()
    assert v["emergency"] == "POLICE"
    assert c.get("/state").json()["emergency"] == "POLICE"
    c.post("/resolve")
    assert c.get("/state").json()["state"] == "IDLE"
    wav = c.get("/simulate/sound/FORCED_ENTRY.wav").content
    assert c.post("/audio/classify", files={"file": ("a.wav", wav, "audio/wav")}).json()["label"] == "FORCED_ENTRY"
    assert c.get("/scenarios/check").json()["passed"] == len(SCENARIOS)
