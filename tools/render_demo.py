"""Render the demo video: runs the real SafeEntry engine through scripted
scenarios and draws the 4-room twin, door-plate beacon, camera feed and
event log frame by frame. Synthetic sensor audio is mixed into the soundtrack.

    python tools/render_demo.py docs/safeentry_demo.mp4
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safeentry.audio import SR, synth_forced_entry, synth_glass_break, synth_knock, synth_sos_knock  # noqa: E402
from safeentry.environment import LAYOUT, ROOMS, HomeEnvironment  # noqa: E402
from safeentry.scenarios import SCENARIOS, SimClock, check  # noqa: E402
from safeentry.coordinator import SafeEntry  # noqa: E402
from safeentry.subsystems import EntryGuard  # noqa: E402

W, H, FPS = 1280, 720, 30
F = "/usr/share/fonts/opentype/inter/Inter-%s.otf"


def font(sz, w="Regular"):
    try:
        return ImageFont.truetype(F % w, sz)
    except OSError:
        return ImageFont.load_default()


f12, f14, f16, f20, f28, f44 = font(13), font(15), font(17, "Medium"), font(21, "SemiBold"), font(30, "Bold"), font(48, "Bold")
BG, CARD, LINE, INK, MUTE = (15, 17, 21), (24, 27, 34), (42, 47, 58), (232, 234, 240), (139, 145, 161)
STATE_COL = {"IDLE": (70, 76, 90), "WATCH": (245, 165, 36), "INTRUSION": (240, 68, 56), "EMERGENCY": (180, 35, 24)}
ROLE_COL = {"resident": (62, 207, 142), "intruder": (240, 68, 56), "visitor": (245, 200, 60), "responder": (80, 140, 255)}

# YOLO on simulated feed (optional)
try:
    from ultralytics.utils import ASSETS
    from safeentry.subsystems import YoloDetector
    import cv2
    YOLO = YoloDetector("yolov8n.pt")  # auto-downloaded by ultralytics
    FEED_IMG = cv2.imread(str(ASSETS / "bus.jpg"))[380:1000]  # ultralytics sample photo as stand-in feed
    FEED_BOXES = YOLO.boxes(FEED_IMG)
except Exception as e:  # noqa: BLE001
    print("YOLO unavailable:", e)
    YOLO = None


def feed_tile(intruder: bool, w=300, h=230):
    if YOLO is None:
        return None
    img = cv2.cvtColor(FEED_IMG, cv2.COLOR_BGR2RGB)
    sy, sx = h / img.shape[0], w / img.shape[1]
    tile = Image.fromarray(img).resize((w, h))
    if not intruder:
        return Image.new("RGB", (w, h), (35, 38, 45))
    d = ImageDraw.Draw(tile)
    for x1, y1, x2, y2 in FEED_BOXES:
        d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(255, 60, 60), width=3)
        d.text((x1 * sx + 4, y1 * sy + 2), "person", font=f12, fill=(255, 80, 80))
    return tile


def draw(se: SafeEntry, t: float, title: str, caption: str, step_i: int, n_steps: int, show_feed: bool):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    st, em = se.sm.state.value, se.sm.emergency.value
    # header
    d.text((28, 20), "SafeEntry Home", font=f28, fill=INK)
    d.rounded_rectangle([290, 24, 290 + 18 * len(st) + 30, 56], 14, fill=STATE_COL[st])
    d.text((305, 28), st, font=f20, fill=(255, 255, 255))
    if em != "NONE":
        x0 = 290 + 18 * len(st) + 42
        d.rounded_rectangle([x0, 24, x0 + 16 * len(em) + 30, 56], 14, fill=(52, 68, 68))
        d.text((x0 + 14, 28), em, font=f20, fill=INK)
    d.text((W - 28, 30), title, font=f16, fill=MUTE, anchor="ra")
    # floor plan
    ox, oy, s = 28, 90, 62
    d.text((ox, oy - 4), "4-ROOM HOME TWIN", font=f12, fill=MUTE)
    oy += 20
    prio = se.dispatch["priority_room"] if se.dispatch else None
    for r in ROOMS:
        x, y, w, h = LAYOUT[r]
        box = [ox + x * s, oy + y * s, ox + (x + w) * s - 6, oy + (y + h) * s - 6]
        d.rounded_rectangle(box, 8, fill=(18, 21, 27), outline=(240, 68, 56) if r == prio else LINE,
                            width=4 if r == prio else 1)
        d.text((box[0] + 10, box[1] + 8), r.replace("_", " "), font=f12, fill=MUTE)
        for i, p in enumerate(se.env.occupants(r)):
            cx, cy = box[0] + 50 + i * 90, box[1] + 120
            c = ROLE_COL[p.role]
            if p.posture == "lying":
                d.rounded_rectangle([cx - 30, cy + 10, cx + 30, cy + 30], 8, fill=c)
            else:
                d.ellipse([cx - 14, cy - 34, cx + 14, cy - 6], fill=c)
                d.rounded_rectangle([cx - 18, cy - 2, cx + 18, cy + 50], 8, fill=c)
            d.text((cx, cy + 58), p.pid, font=f12, fill=INK, anchor="ma")
        if r == prio:
            d.text((box[2] - 10, box[1] + 8), "PRIORITY ROOM", font=f12, fill=(255, 90, 80), anchor="ra")
    # front door
    dx, dy = ox - 6, oy + 2.0 * s - 40
    d.rectangle([dx, dy, dx + 10, dy + 80], fill=(240, 68, 56) if se.env.door_forced else (
        (62, 207, 142) if not se.env.door_locked else (120, 125, 140)))
    door = "FORCED" if se.env.door_forced else ("UNLOCKED for responders" if not se.env.door_locked else "locked")
    d.text((ox, oy + 8 * s + 4), f"Front door: {door}", font=f14, fill=INK)
    # right column: beacon
    rx = 560
    d.text((rx, 90), "DOOR-PLATE BEACON (as seen from the street)", font=f12, fill=MUTE)
    l, r_ = se.beacon.pattern(se.sm.state, se.sm.emergency, t)
    d.rounded_rectangle([rx, 112, rx + 330, 222], 12, fill=(30, 30, 30))
    d.rounded_rectangle([rx + 8, 120, rx + 164, 214], 8, fill=tuple(l))
    d.rounded_rectangle([rx + 166, 120, rx + 322, 214], 8, fill=tuple(r_))
    d.text((rx + 165, 155), "42", font=f28, fill=(255, 255, 255) if max(l) < 200 or max(r_) < 200 else (0, 0, 0), anchor="ma")
    d.text((rx, 230), se.beacon.describe(se.sm.state, se.sm.emergency), font=f16, fill=INK)
    # speaker
    sp = EntryGuard.MESSAGES.get(se.guard.playing, "—") if se.guard.playing else "—"
    d.text((rx, 266), "ENTRY GUARD SPEAKER", font=f12, fill=MUTE)
    d.multiline_text((rx, 284), _wrap(sp, 44), font=f14, fill=INK)
    # camera feed
    if show_feed:
        tile = feed_tile(any(p.role == "intruder" for p in se.env.people.values()))
        if tile is not None:
            im.paste(tile, (rx + 360 - 10, 112))
            d.text((rx + 350, 90), "LIVING ROOM CAM · YOLOv8n", font=f12, fill=MUTE)
    # dispatch
    d.text((rx, 340), "SOS BACKEND DISPATCH", font=f12, fill=MUTE)
    if se.dispatch:
        dp = se.dispatch
        lines = [f"emergency: {dp['emergency']}", f"priority room: {dp['priority_room']}",
                 f"occupied: {', '.join(dp['occupied_rooms']) or 'none'}", f"door: {dp['door']}"]
        d.multiline_text((rx, 358), "\n".join(lines), font=f14, fill=(255, 210, 200), spacing=6)
    else:
        d.text((rx, 358), "—", font=f14, fill=MUTE)
    # event log
    d.text((rx, 470), "EVENT LOG", font=f12, fill=MUTE)
    evs = [e for e in se.events if e.kind != "sweep"][-6:]
    for i, e in enumerate(evs):
        det = e.detail
        msg = det.get("label") or (f"{det.get('frm')} → {det.get('to')} {det.get('emergency') if det.get('emergency') != 'NONE' else ''}" if e.kind == "transition" else "") \
            or det.get("message") or (", ".join(det.get("matches", [])) if e.kind == "transcript" else "") or det.get("emergency", "")
        d.text((rx, 490 + i * 20), f"t={e.t:5.1f}s  {e.source:<9} {e.kind:<11} {msg}"[:84], font=f12, fill=(197, 202, 214))
    # caption bar
    d.rectangle([0, H - 70, W, H], fill=CARD)
    d.text((28, H - 52), caption, font=f20, fill=INK)
    d.text((W - 28, H - 48), f"step {step_i}/{n_steps}", font=f14, fill=MUTE, anchor="ra")
    return im


def _wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    out.append(line)
    return "\n".join(out)


def card(lines, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 260), lines, font=f44, fill=INK, anchor="ma")
    if sub:
        d.multiline_text((W // 2, 340), sub, font=f20, fill=MUTE, anchor="ma", align="center", spacing=10)
    return im


AUDIO_FOR = {"Three polite knocks": lambda: synth_knock(seed=2), "Heavy blows on the door": lambda: synth_forced_entry(seed=3),
             "Glass breaks": lambda: synth_glass_break(seed=4),
             "Resident knocks 3-3-3 on inside of door": lambda: synth_sos_knock(seed=5)}

DEMO = ["everyday_knock", "break_in_empty_home", "medical_fall", "police_call", "silent_sos_knock"]
STEP_SECONDS, HOLD_SECONDS = 3.2, 3.5


def main(out="docs/safeentry_demo.mp4"):
    tmp = Path(tempfile.mkdtemp())
    audio = []
    count = [0]
    vp = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                           "-preset", "medium", str(tmp / "v.mp4")], stdin=subprocess.PIPE)

    def add(img, secs, snd=None):
        n = int(secs * FPS)
        if snd is not None:
            audio.append((count[0] / FPS, snd))
        for k in range(n):
            f = img(k / FPS) if callable(img) else img
            vp.stdin.write(f.tobytes())
        count[0] += n

    results = [check(n) for n in SCENARIOS]
    add(card("SafeEntry Home", "Acoustic door-plate sentinel · Entry guard · Indoor camera responder\n"
             "Unified state machine  IDLE → WATCH → INTRUSION → EMERGENCY  ·  FastAPI SOS backend"), 4)
    for name in DEMO:
        sc = SCENARIOS[name]
        add(card(sc.title, f"Scenario: {name}"), 2.2)
        clock = SimClock()
        env = HomeEnvironment()
        sc.setup(env)
        se = SafeEntry(env, clock=clock)
        show_feed = name == "break_in_empty_home"
        add(lambda tt: draw(se, tt, sc.title, "Initial state", 0, len(sc.steps), show_feed), 1.5)
        for i, stp in enumerate(sc.steps, 1):
            clock.t = stp.at
            stp.fn(se)
            snd = AUDIO_FOR.get(stp.caption, lambda: None)()
            if se.guard.playing == "deterrent" and snd is not None:
                snd = np.concatenate([snd, se.guard.siren(1.5) * 0.4])
            hold = HOLD_SECONDS if i == len(sc.steps) else STEP_SECONDS
            # snapshot state for this segment (engine is not mutated while drawing)
            add(lambda tt, se=se, i=i, c=stp.caption: draw(se, tt, sc.title, c, i, len(sc.steps), show_feed), hold, snd)
    passed = sum(r["passed"] for r in results)
    add(card(f"{passed}/{len(results)} automated scenario checks passed",
             "\n".join(("PASS  " if r["passed"] else "FAIL  ") + r["scenario"] for r in results)), 5)

    vp.stdin.close()
    vp.wait()
    total = count[0] / FPS
    mix = np.zeros(int(total * SR) + SR)
    for at, snd in audio:
        i = int(at * SR)
        mix[i:i + len(snd)] += snd[: len(mix) - i]
    mix = np.clip(mix * 0.8, -1, 1)
    with wave.open(str(tmp / "a.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / "v.mp4"), "-i", str(tmp / "a.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}: {total:.1f}s, {passed}/{len(results)} scenarios passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
