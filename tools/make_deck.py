"""Build the Build-2036 submission deck (PDF) for SafeEntry Home."""
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
W, H = 960, 540
RED, INK, MUTE, BG, CARD = HexColor("#D7261E"), HexColor("#16181D"), HexColor("#5B6170"), HexColor("#FFFFFF"), HexColor("#F4F5F7")
try:
    pdfmetrics.registerFont(TTFont("R", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    R, B = "R", "B"
except Exception:
    R, B = "Helvetica", "Helvetica-Bold"

REPO = "https://github.com/anshuu-prajapati/safeentry-home"


def header(c, title, n):
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(RED); c.rect(0, H - 8, W, 8, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont(B, 26); c.drawString(40, H - 60, title)
    c.setFillColor(MUTE); c.setFont(R, 10)
    c.drawString(40, 18, "Build 2036 · Applied AI & Agentic Intelligence · Team Nova AI")
    c.drawRightString(W - 40, 18, str(n))


def para(c, x, y, text, size=13, width=420, lead=1.4, font=None, color=INK):
    c.setFont(font or R, size); c.setFillColor(color)
    for raw in text.split("\n"):
        line = ""
        for w in raw.split(" "):
            t = (line + " " + w).strip()
            if c.stringWidth(t, font or R, size) > width:
                c.drawString(x, y, line); y -= size * lead; line = w
            else:
                line = t
        c.drawString(x, y, line); y -= size * lead
    return y


def bullets(c, x, y, items, size=13, width=420):
    for it in items:
        c.setFillColor(RED); c.circle(x + 3, y + size * 0.35, 2.6, fill=1, stroke=0)
        y = para(c, x + 14, y, it, size, width - 14) - 6
    return y


def card(c, x, y, w, h, title, body, size=11.5):
    c.setFillColor(CARD); c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(RED); c.rect(x, y + h - 4, w, 4, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont(B, 13.5); c.drawString(x + 14, y + h - 28, title)
    para(c, x + 14, y + h - 50, body, size, w - 28)


def main(out, repo=REPO, video=None):
    c = canvas.Canvas(out, pagesize=(W, H))
    c.setTitle("SafeEntry Home — Build 2036 prototype")
    # 1 — title / template fields
    c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(RED); c.rect(0, 0, 14, H, fill=1, stroke=0)
    c.setFillColor(INK); c.setFont(B, 46); c.drawString(60, 380, "SafeEntry Home")
    para(c, 60, 340, "A smart door plate that can tell a knock from a break-in, and lights up so help finds you faster.", 17, 820, color=MUTE)
    c.setFont(B, 14); c.setFillColor(INK)
    rows = [("Team Name", "Nova AI"), ("Team Leader Name", "Sarvesh Swami"),
            ("Problem Statement", "Homes can't tell everyday door activity from break-ins or medical crises, and responders lose critical minutes finding the right house and the right room."),
            ("Challenge", "Applied AI & Agentic Intelligence")]
    y = 250
    for k, v in rows:
        c.setFont(B, 13); c.setFillColor(INK); c.drawString(60, y, k + ":")
        y = para(c, 230, y, v, 13, 660) - 10
    c.showPage()

    # 2 — problem
    header(c, "The problem", 2)
    bullets(c, 40, 440, [
        "Doorbell cameras and alarms alert on everything. Constant false alarms teach people to ignore them.",
        "In a medical emergency (an elderly person falling, a cardiac event), the person who needs help often can't get to the door or say where they are.",
        "Ambulances and police lose time finding the house: house numbers are faded, unlit, or missing, especially at night and in dense Indian neighbourhoods.",
        "Once inside, responders don't know which room the person is in, or whether an intruder is still there.",
    ], 15, 560)
    card(c, 640, 210, 280, 240, "Who it's for",
         "Elderly people living alone, families with patients at home, single women and working parents, and the emergency responders who serve them.")
    c.showPage()

    # 3 — solution
    header(c, "Solution: three linked subsystems, one brain", 3)
    cw = 210
    for i, (t, b) in enumerate([
        ("1 · Door-plate sentinel", "Acoustic AI on the door plate tells an everyday knock apart from forced entry, glass break and a covert 3·3·3 SOS knock code."),
        ("2 · Entry guard", "Smart lock and outward speaker. Greets visitors, plays deterrent audio and a siren on intrusion, and unlocks for paramedics."),
        ("3 · Camera responder", "Sweeps all rooms with YOLOv8 person detection during a crisis: who is where, and is anyone lying down?"),
        ("Door-plate beacon", "Medical: blinking red. Police: split red | blue. Fire: orange/white. Responders can spot the house from the street."),
    ]):
        card(c, 40 + i * (cw + 16), 230, cw, 220, t, b, 11.5)
    para(c, 40, 190, "All four are coordinated by a unified state machine (IDLE → WATCH → INTRUSION → EMERGENCY) and a FastAPI SOS and call-detection backend that builds a dispatch packet: emergency type, occupied rooms, priority room, and door status.", 13.5, 880)
    c.showPage()

    # 4 — state machine
    header(c, "Unified state machine", 4)
    states = [("IDLE", "#6B7280"), ("WATCH", "#F5A524"), ("INTRUSION", "#F04438"), ("EMERGENCY", "#B42318")]
    for i, (s, col) in enumerate(states):
        x = 50 + i * 225
        c.setFillColor(HexColor(col)); c.roundRect(x, 330, 170, 70, 10, fill=1, stroke=0)
        c.setFillColor(HexColor("#FFFFFF")); c.setFont(B, 18); c.drawCentredString(x + 85, 358, s)
        if i < 3:
            c.setStrokeColor(INK); c.setLineWidth(2); c.line(x + 175, 365, x + 220, 365)
            c.line(x + 212, 371, x + 220, 365); c.line(x + 212, 359, x + 220, 365)
    for i, lab in enumerate(["knock", "forced entry /\nglass break", "unknown person\ninside"]):
        para(c, 225 + i * 225 - 2, 425, lab, 10.5, 110, color=MUTE)
    para(c, 50, 300, "SOS button · 3·3·3 knock code · distress detected in a call  →  straight to EMERGENCY {MEDICAL | POLICE | FIRE}", 13, 860, font=B)
    bullets(c, 50, 250, [
        "Severity never drops on its own: only WATCH times out (30 s). INTRUSION and EMERGENCY need an explicit resolve.",
        "An emergency can be upgraded, e.g. MEDICAL → POLICE if the camera sweep finds an unknown person.",
        "Every transition drives the beacon, the entry guard and the dispatch packet together, so the subsystems never disagree.",
    ], 13, 860)
    c.showPage()

    # 5 — architecture
    header(c, "Architecture & tech stack", 5)
    boxes = [(40, 330, "Door-plate mic", "onset + spectral\nfeature classifier"), (40, 220, "Indoor cameras ×4", "YOLOv8n person\ndetection"),
             (40, 110, "Phone / app / wearable", "SOS + call\ntranscript"),
             (360, 220, "SafeEntry coordinator", "state machine ·\nevent bus · dispatch"),
             (680, 330, "Entry guard", "lock · deterrent\naudio · siren"), (680, 220, "Door-plate beacon", "red / red|blue /\norange patterns"),
             (680, 110, "FastAPI backend", "/sos · /call/detect ·\n/state · dashboard")]
    for x, y, t, b in boxes:
        c.setFillColor(CARD if x != 360 else HexColor("#FDECEA")); c.roundRect(x, y, 230, 80, 8, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont(B, 13); c.drawString(x + 12, y + 56, t)
        para(c, x + 12, y + 36, b, 11, 210, 1.3, color=MUTE)
    c.setStrokeColor(MUTE); c.setLineWidth(1.5)
    for y in (370, 260, 150):
        c.line(270, y, 360, 260)
        c.line(590, 260, 680, y)
    para(c, 40, 80, "Python 3 · NumPy signal processing · Ultralytics YOLOv8n + OpenCV · FastAPI + Uvicorn · Pydantic · pytest · 4-room Cyberwave-style digital twin (optional Cyberwave bridge) · HTML/JS live dashboard · FFmpeg demo renderer", 12, 880, color=INK)
    c.showPage()

    # 6 — screenshots
    header(c, "Prototype: simulation and live dashboard", 6)
    for i, (img, cap) in enumerate([("docs/demo_breakin.png", "Break-in: deterrent strobe, YOLOv8 finds a person, POLICE"),
                                    ("docs/demo_medical.png", "Medical fall: blinking red beacon, priority room = kitchen, door unlocked")]):
        c.drawImage(ImageReader(str(ROOT / img)), 40 + i * 450, 170, 430, 242)
        para(c, 40 + i * 450, 150, cap, 11.5, 430, color=MUTE)
    c.drawImage(ImageReader(str(ROOT / "docs/dashboard.png")), 40, 32, 160, 95)
    para(c, 215, 105, "FastAPI dashboard: inject sounds, SOS presses and call transcripts, and watch the state machine, beacon and dispatch packet update live.", 12, 680)
    c.showPage()

    # 7 — validation
    header(c, "Tested with automated scenario checks", 7)
    rows = [("everyday_knock", "IDLE → WATCH → IDLE", "no false alarm"),
            ("break_in_empty_home", "→ INTRUSION → EMERGENCY/POLICE", "priority: living room"),
            ("glass_break_deterred", "→ INTRUSION → resolved", "deterrent played"),
            ("medical_fall", "→ EMERGENCY/MEDICAL", "priority: kitchen, door unlocked"),
            ("police_call", "call transcript → POLICE", "keywords + negation guard"),
            ("silent_sos_knock", "3·3·3 code → POLICE", "covert distress"),
            ("negated_call", "'no need for an ambulance'", "stays IDLE")]
    y = 430
    c.setFont(B, 12); c.setFillColor(MUTE)
    c.drawString(40, y, "SCENARIO"); c.drawString(260, y, "FLOW"); c.drawString(530, y, "CHECK"); c.drawString(800, y, "RESULT")
    for s, f, k in rows:
        y -= 30
        c.setFont(R, 12.5); c.setFillColor(INK)
        c.drawString(40, y, s); c.drawString(260, y, f); c.drawString(530, y, k)
        c.setFillColor(HexColor("#12924F")); c.setFont(B, 12.5); c.drawString(800, y, "PASS")
    bullets(c, 40, 160, [
        "26 pytest checks: state machine rules, audio classifier per class (100/100 on seeded clips), call detection, beacon timing, all 7 scenarios, the full API flow, and YOLOv8 occupancy on camera-feed stand-ins.",
        "Sensors are simulated stand-ins (synthetic audio, rendered/sample camera frames). That's honest prototype scope, and the hardware interfaces are already in place.",
    ], 12.5, 880)
    c.showPage()

    # 8 — impact / roadmap
    header(c, "Impact, feasibility & next steps", 8)
    card(c, 40, 200, 280, 250, "Why it matters",
         "Fewer false alarms, so people keep it switched on. A visible beacon and room-level location save minutes in the golden hour. Silent SOS works when calling out loud isn't safe.")
    card(c, 340, 200, 280, 250, "Feasible & affordable",
         "Built from cheap parts: a contact mic + MCU, LED strip, smart lock and existing IP cameras. Edge-first audio model with no cloud audio upload, which protects privacy.")
    card(c, 640, 200, 280, 250, "Roadmap",
         "1. Field-record knock/impact dataset and train a TinyML model.\n2. ESP32 door-plate hardware pilot.\n3. ERSS-112 / RWA security integration.\n4. Fall detection from pose estimation.")
    y = para(c, 40, 160, "Code: " + repo, 13, 880, font=B)
    if video:
        para(c, 40, y - 4, "Demo video: " + video, 13, 880, font=B)
    c.showPage()
    c.save()


if __name__ == "__main__":
    main(*sys.argv[1:])
