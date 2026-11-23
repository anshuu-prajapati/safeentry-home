"""Scripted scenarios used by automated checks, the API and the demo renderer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .audio import synth_forced_entry, synth_glass_break, synth_knock, synth_sos_knock, synth_ambient
from .coordinator import SafeEntry
from .environment import HomeEnvironment, Person
from .states import EmergencyType


class SimClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@dataclass
class Step:
    at: float
    caption: str
    fn: Callable[[SafeEntry], object]


@dataclass
class Scenario:
    name: str
    title: str
    setup: Callable[[HomeEnvironment], None]
    steps: list
    expect_state: str
    expect_emergency: str = "NONE"
    expect_priority_room: str | None = None
    tags: list = field(default_factory=list)


def _family(env):
    env.add(Person("asha", "LIVING_ROOM"))
    env.add(Person("dadaji", "BEDROOM"))


def _alone_elder(env):
    env.add(Person("dadaji", "KITCHEN"))


def _empty(env):
    env.residents_home = False


def _fall(se):
    se.env.people["dadaji"].posture = "lying"


def _intruder_enters(se):
    se.env.add(Person("x1", "LIVING_ROOM", role="intruder"))


SCENARIOS = {s.name: s for s in [
    Scenario("everyday_knock", "Everyday knock — courier at the door", _family, [
        Step(1, "Ambient household noise", lambda se: se.on_audio(synth_ambient(seed=1))),
        Step(3, "Three polite knocks", lambda se: se.on_audio(synth_knock(seed=2))),
        Step(40, "No follow-up — WATCH times out", lambda se: se.tick()),
    ], "IDLE"),
    Scenario("break_in_empty_home", "Break-in while family is away", _empty, [
        Step(1, "Heavy blows on the door", lambda se: se.on_audio(synth_forced_entry(seed=3))),
        Step(4, "Intruder steps into living room", _intruder_enters),
        Step(5, "Camera sweep finds unknown person", lambda se: se.check_intrusion()),
    ], "EMERGENCY", "POLICE", "LIVING_ROOM"),
    Scenario("glass_break_deterred", "Window smashed, intruder flees", _family, [
        Step(1, "Glass breaks", lambda se: se.on_audio(synth_glass_break(seed=4))),
        Step(8, "Owner confirms all clear in app", lambda se: se.resolve("owner")),
    ], "IDLE"),
    Scenario("medical_fall", "Elder falls in kitchen, presses SOS", _alone_elder, [
        Step(1, "Dadaji falls in the kitchen", _fall),
        Step(2, "Wearable/app SOS: MEDICAL", lambda se: se.trigger_sos(EmergencyType.MEDICAL, "SOS button")),
    ], "EMERGENCY", "MEDICAL", "KITCHEN"),
    Scenario("police_call", "Call detection: 'someone is breaking in, call police'", _family, [
        Step(1, "Resident on phone", lambda se: se.on_call_transcript(
            "please help, someone is breaking in, call the police")),
    ], "EMERGENCY", "POLICE", "LIVING_ROOM"),
    Scenario("silent_sos_knock", "Hostage-safe SOS knock code (3x3)", _family, [
        Step(1, "Resident knocks 3-3-3 on inside of door", lambda se: se.on_audio(synth_sos_knock(seed=5))),
    ], "EMERGENCY", "POLICE"),
    Scenario("negated_call", "Call says 'no need for an ambulance'", _family, [
        Step(1, "Casual phone call", lambda se: se.on_call_transcript(
            "it's fine, no need for an ambulance, he just tripped")),
    ], "IDLE"),
]}


def run(name: str, detector=None):
    sc = SCENARIOS[name]
    clock = SimClock()
    env = HomeEnvironment()
    sc.setup(env)
    se = SafeEntry(env, detector=detector, clock=clock)
    trace = []
    for st in sc.steps:
        clock.t = st.at
        st.fn(se)
        trace.append({"at": st.at, "caption": st.caption, "state": se.sm.state.value,
                      "emergency": se.sm.emergency.value})
    return se, trace


def check(name: str, detector=None) -> dict:
    sc = SCENARIOS[name]
    se, trace = run(name, detector)
    pr = se.dispatch["priority_room"] if se.dispatch else None
    ok = se.sm.state.value == sc.expect_state and se.sm.emergency.value == sc.expect_emergency
    if sc.expect_priority_room:
        ok = ok and pr == sc.expect_priority_room
    return {"scenario": name, "title": sc.title, "passed": ok, "final_state": se.sm.state.value,
            "emergency": se.sm.emergency.value, "priority_room": pr, "trace": trace}
