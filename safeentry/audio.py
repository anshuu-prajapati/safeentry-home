"""Acoustic AI for the door-plate sentinel.

Two parts:
  * synth_*  : deterministic generators used as simulated microphone stand-ins
  * classify : onset + spectral feature pipeline that separates everyday knocks
               from forced entry, glass break and the SOS knock code.

The classifier is a transparent feature/rule model (no training data needed) so
it can run on a microcontroller-class door plate; `AudioClassifier` exposes the
same interface a learned model (e.g. YAMNet fine-tune) would plug into.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

SR = 16_000
FRAME = 320  # 20 ms
HOP = 160    # 10 ms


class SoundClass(str, Enum):
    AMBIENT = "AMBIENT"
    KNOCK = "KNOCK"
    FORCED_ENTRY = "FORCED_ENTRY"
    GLASS_BREAK = "GLASS_BREAK"
    SOS_KNOCK = "SOS_KNOCK"   # 3 groups of 3 knocks -> silent distress code


# ----------------------------------------------------------------- synthesis
def _rng(seed):
    return np.random.default_rng(seed)


def _hit(rng, dur, f0, amp, decay, noise_mix, hp=False):
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.exp(-t / decay)
    tone = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * f0 * 2.3 * t)
    noise = rng.standard_normal(n)
    if hp:
        noise = np.diff(noise, prepend=0)
        noise = np.diff(noise, prepend=0)
        noise /= np.max(np.abs(noise)) + 1e-9
    else:  # crude low-pass
        noise = np.convolve(noise, np.ones(8) / 8, mode="same")
        noise /= np.max(np.abs(noise)) + 1e-9
    return amp * env * ((1 - noise_mix) * tone + noise_mix * noise)


def _place(buf, sig, at):
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def background(dur, seed=0, level=0.01):
    return level * _rng(seed).standard_normal(int(dur * SR))


def synth_knock(n=3, seed=0, gap=0.22, start=0.3, dur=2.5, amp=0.30):
    rng = _rng(seed)
    buf = background(dur, seed)
    t = start
    for _ in range(n):
        _place(buf, _hit(rng, 0.12, 210 + rng.uniform(-30, 30), amp * rng.uniform(0.85, 1.1), 0.025, 0.45), t)
        t += gap * rng.uniform(0.85, 1.15)
    return buf


def synth_sos_knock(seed=0, dur=4.5):
    rng = _rng(seed)
    buf = background(dur, seed)
    t = 0.3
    for g in range(3):
        for _ in range(3):
            _place(buf, _hit(rng, 0.12, 210, 0.30 * rng.uniform(0.9, 1.1), 0.025, 0.45), t)
            t += 0.18 * rng.uniform(0.9, 1.1)
        t += 0.85 * rng.uniform(0.9, 1.1)
    return buf


def synth_forced_entry(seed=0, dur=3.0, hits=4):
    rng = _rng(seed)
    buf = background(dur, seed)
    t = 0.25
    for _ in range(hits):
        _place(buf, _hit(rng, 0.45, 75 + rng.uniform(-10, 10), 0.95, 0.11, 0.6), t)
        t += 0.6 * rng.uniform(0.8, 1.2)
    return np.clip(buf, -1, 1)


def synth_glass_break(seed=0, dur=2.0):
    rng = _rng(seed)
    buf = background(dur, seed)
    _place(buf, _hit(rng, 0.6, 5200, 0.9, 0.12, 0.85, hp=True), 0.3)
    for _ in range(12):  # tinkling shards
        _place(buf, _hit(rng, 0.05, rng.uniform(4000, 7500), 0.25, 0.01, 0.3, hp=True), 0.35 + rng.uniform(0, 1.2))
    return np.clip(buf, -1, 1)


def synth_ambient(seed=0, dur=2.5):
    rng = _rng(seed)
    buf = background(dur, seed, level=0.02)
    t = np.arange(len(buf)) / SR
    buf += 0.03 * np.sin(2 * np.pi * 120 * t)  # fridge / mains hum
    return buf


SYNTH = {
    SoundClass.AMBIENT: synth_ambient,
    SoundClass.KNOCK: synth_knock,
    SoundClass.SOS_KNOCK: synth_sos_knock,
    SoundClass.FORCED_ENTRY: synth_forced_entry,
    SoundClass.GLASS_BREAK: synth_glass_break,
}


# ------------------------------------------------------------------ features
@dataclass
class Features:
    onsets: list            # seconds
    peak_db: float
    centroid_hz: float      # mean spectral centroid at onsets
    hf_ratio: float         # energy > 4 kHz / total, at onsets
    groups: list            # knock counts per group (gap > 0.5 s splits)

    def as_dict(self):
        return {"onsets": [round(o, 3) for o in self.onsets], "peak_db": round(self.peak_db, 1),
                "centroid_hz": round(self.centroid_hz), "hf_ratio": round(self.hf_ratio, 3),
                "groups": self.groups}


def _frames(x):
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def extract(x: np.ndarray) -> Features:
    x = np.asarray(x, dtype=np.float64)
    fr = _frames(x)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    floor = np.median(rms)
    thr = max(floor * 6, 0.02)
    onsets = []
    last = -1e9
    for i in range(1, len(rms)):
        if rms[i] > thr and rms[i - 1] <= thr and (i - last) * HOP / SR > 0.08:
            onsets.append(i)
            last = i
    win = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / SR)
    cents, hfs = [], []
    for i in onsets:
        seg = fr[i: i + 3].mean(axis=0) if i + 3 <= len(fr) else fr[i]
        spec = np.abs(np.fft.rfft(fr[i] * win)) ** 2 + np.abs(np.fft.rfft(seg * win)) ** 2
        tot = spec.sum() + 1e-12
        cents.append(float((freqs * spec).sum() / tot))
        hfs.append(float(spec[freqs > 4000].sum() / tot))
    on_s = [i * HOP / SR for i in onsets]
    groups = []
    for k, t in enumerate(on_s):
        if k == 0 or t - on_s[k - 1] > 0.5:
            groups.append(1)
        else:
            groups[-1] += 1
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return Features(on_s, float(20 * np.log10(peak + 1e-9)), float(np.mean(cents)) if cents else 0.0,
                    float(np.mean(hfs)) if hfs else 0.0, groups)


@dataclass
class Classification:
    label: SoundClass
    confidence: float
    features: Features

    def as_dict(self):
        return {"label": self.label.value, "confidence": round(self.confidence, 2), "features": self.features.as_dict()}


class AudioClassifier:
    """Feature/rule classifier. Order matters: most dangerous classes checked first."""

    def classify(self, x: np.ndarray) -> Classification:
        f = extract(x)
        if not f.onsets:
            return Classification(SoundClass.AMBIENT, 0.9, f)
        if f.hf_ratio > 0.45 and f.peak_db > -12:
            return Classification(SoundClass.GLASS_BREAK, min(0.99, 0.6 + f.hf_ratio / 2), f)
        if f.peak_db > -3 and f.centroid_hz < 1500 and len(f.onsets) >= 2:
            return Classification(SoundClass.FORCED_ENTRY, min(0.99, 0.7 + 0.05 * len(f.onsets)), f)
        if len(f.groups) >= 3 and all(g == 3 for g in f.groups[:3]):
            return Classification(SoundClass.SOS_KNOCK, 0.95, f)
        if 1 <= len(f.onsets) <= 8:
            return Classification(SoundClass.KNOCK, 0.85, f)
        return Classification(SoundClass.AMBIENT, 0.5, f)


def to_wav_bytes(x: np.ndarray) -> bytes:
    import io
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


def from_wav_bytes(data: bytes) -> np.ndarray:
    import io
    import wave
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("16-bit PCM WAV required")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768
        if w.getnchannels() > 1:
            raw = raw.reshape(-1, w.getnchannels()).mean(axis=1)
        sr = w.getframerate()
    if sr != SR:  # linear resample
        n = int(len(raw) * SR / sr)
        raw = np.interp(np.linspace(0, len(raw) - 1, n), np.arange(len(raw)), raw)
    return raw
