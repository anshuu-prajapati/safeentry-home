"""SafeEntry FastAPI backend: SOS, call detection, audio classification, live state.

Run:  uvicorn api.main:app --reload   then open http://localhost:8000
"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from safeentry.audio import SYNTH, SoundClass, from_wav_bytes, to_wav_bytes
from safeentry.coordinator import SafeEntry
from safeentry.environment import ROOMS, HomeEnvironment, Person
from safeentry.scenarios import SCENARIOS, check
from safeentry.states import EmergencyType

app = FastAPI(title="SafeEntry Home", version="0.1.0",
              description="Smart-home security & emergency-response backend")
_lock = threading.Lock()


def _fresh() -> SafeEntry:
    env = HomeEnvironment()
    env.add(Person("asha", "LIVING_ROOM"))
    env.add(Person("dadaji", "BEDROOM"))
    return SafeEntry(env)


SE = _fresh()


class SOSIn(BaseModel):
    type: EmergencyType = Field(EmergencyType.MEDICAL, description="MEDICAL | POLICE | FIRE")
    source: str = "app"


class CallIn(BaseModel):
    transcript: str = Field(..., max_length=2000)


class MoveIn(BaseModel):
    pid: str
    room: str
    role: str = "resident"
    posture: str = "standing"


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "dashboard.html").read_text()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/state")
def state():
    with _lock:
        SE.tick()
        return SE.snapshot()


@app.post("/sos")
def sos(body: SOSIn):
    if body.type == EmergencyType.NONE:
        raise HTTPException(422, "type must be MEDICAL, POLICE or FIRE")
    with _lock:
        return SE.trigger_sos(body.type, f"SOS from {body.source}")


@app.post("/call/detect")
def call_detect(body: CallIn):
    with _lock:
        return SE.on_call_transcript(body.transcript)


@app.post("/audio/classify")
async def audio_classify(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > 5_000_000:
        raise HTTPException(413, "clip too large")
    try:
        clip = from_wav_bytes(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"bad wav: {e}")
    with _lock:
        return SE.on_audio(clip)


@app.post("/simulate/sound/{label}")
def simulate_sound(label: SoundClass, seed: int = 0):
    with _lock:
        return SE.on_audio(SYNTH[label](seed=seed))


@app.get("/simulate/sound/{label}.wav")
def sound_wav(label: SoundClass, seed: int = 0):
    return Response(to_wav_bytes(SYNTH[label](seed=seed)), media_type="audio/wav")


@app.post("/simulate/person")
def simulate_person(body: MoveIn):
    if body.room not in ROOMS + ["OUTSIDE"]:
        raise HTTPException(422, f"room must be one of {ROOMS}")
    with _lock:
        if body.room == "OUTSIDE":
            SE.env.remove(body.pid)
        else:
            SE.env.add(Person(body.pid, body.room, body.role, body.posture))
            SE.check_intrusion()
        return SE.snapshot()


@app.post("/resolve")
def resolve():
    with _lock:
        SE.resolve("owner (app)")
        return SE.snapshot()


@app.post("/reset")
def reset():
    global SE
    with _lock:
        SE = _fresh()
        return SE.snapshot()


@app.get("/scenarios")
def scenarios():
    return [{"name": s.name, "title": s.title} for s in SCENARIOS.values()]


@app.get("/scenarios/check")
def scenarios_check():
    res = [check(n) for n in SCENARIOS]
    return {"passed": sum(r["passed"] for r in res), "total": len(res), "results": res}
