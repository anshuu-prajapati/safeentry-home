"""Call / voice distress detection.

Works on transcripts coming from the phone-call or in-home voice pipeline (any
ASR). Bilingual keyword + phrase scoring (English and romanised Hindi), with
negation guarding so "no need for an ambulance" doesn't fire.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .states import EmergencyType

LEXICON = {
    EmergencyType.MEDICAL: {
        "ambulance": 3, "heart attack": 3, "chest pain": 3, "can't breathe": 3, "cannot breathe": 3,
        "unconscious": 3, "fainted": 2, "fell down": 2, "fallen": 2, "bleeding": 2, "stroke": 3,
        "seizure": 3, "doctor": 1, "dard": 1, "behosh": 3, "saans nahi": 3,
    },
    EmergencyType.POLICE: {
        "police": 3, "intruder": 3, "break in": 3, "breaking in": 3, "someone is inside": 3,
        "thief": 3, "robbery": 3, "gun": 3, "knife": 2, "chor": 3, "attack": 2,
    },
    EmergencyType.FIRE: {
        "fire": 3, "smoke": 2, "burning": 2, "gas leak": 3, "aag": 3,
    },
}
GENERIC = {"help": 1, "help me": 2, "emergency": 2, "sos": 3, "bachao": 3, "please hurry": 1}
NEGATIONS = re.compile(r"\b(no|not|don't|dont|never|nahi)\b(\W+\w+){0,3}\W*$")


@dataclass
class CallVerdict:
    distress: bool
    emergency: EmergencyType
    score: int
    matches: list

    def as_dict(self):
        return {"distress": self.distress, "emergency": self.emergency.value,
                "score": self.score, "matches": self.matches}


def _hits(text: str, lex: dict):
    out = []
    for phrase, w in lex.items():
        for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text):
            if NEGATIONS.search(text[: m.start()]):
                continue
            out.append((phrase, w))
    return out


def analyse_transcript(text: str, threshold: int = 3) -> CallVerdict:
    t = text.lower().replace("’", "'")
    scores, matches = {}, []
    for et, lex in LEXICON.items():
        h = _hits(t, lex)
        scores[et] = sum(w for _, w in h)
        matches += [p for p, _ in h]
    g = _hits(t, GENERIC)
    gen = sum(w for _, w in g)
    matches += [p for p, _ in g]
    best = max(scores, key=scores.get)
    total = scores[best] + gen
    if scores[best] == 0:
        # distress with no category -> treat as POLICE-grade generic SOS only if strong
        return CallVerdict(gen >= threshold, EmergencyType.POLICE if gen >= threshold else EmergencyType.NONE, gen, matches)
    return CallVerdict(total >= threshold, best if total >= threshold else EmergencyType.NONE, total, matches)
