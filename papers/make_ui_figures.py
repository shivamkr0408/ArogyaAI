"""
Render the interface figures for the Arogya AI paper.

These are faithful renderings of the shipped client, drawn from the real
component source in frontend/src/components and the real content in
rag/ayurveda_kb.csv and backend/app. Colours are the project's own Tailwind
tokens (brand ramp from tailwind.config.js), the copy is the literal strings
from the components, and the numbers are those reported in the paper.

They are vector-clean renderings rather than photographic captures. If the
running build drifts from this, replace the PNGs with real captures of the
same views, keeping the filenames.

Run with:  python make_ui_figures.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

S = 2  # supersampling factor

# --- tokens, straight out of tailwind.config.js and the components -------
BRAND = {50: "#eef9f6", 100: "#d8f1ea", 200: "#b5e2d7", 300: "#89cdbd",
         400: "#5cb2a0", 500: "#3e9987", 600: "#2e7d6f", 700: "#28645a",
         800: "#23514a", 900: "#1f443e"}
GRAY = {50: "#f9fafb", 100: "#f3f4f6", 200: "#e5e7eb", 300: "#d1d5db",
        400: "#9ca3af", 500: "#6b7280", 600: "#4b5563", 700: "#374151",
        800: "#1f2937", 900: "#111827"}
INDIGO600, VIOLET600 = "#4f46e5", "#7c3aed"
SLATE400, SLATE600 = "#94a3b8", "#475569"
RED50, RED100, RED200, RED500, RED600, RED700, RED800, RED900 = (
    "#fef2f2", "#fee2e2", "#fecaca", "#ef4444", "#dc2626", "#b91c1c",
    "#991b1b", "#7f1d1d")
ROSE700 = "#be123c"
AMBER50, AMBER100, AMBER200, AMBER700, AMBER800, AMBER900 = (
    "#fffbeb", "#fef3c7", "#fde68a", "#b45309", "#92400e", "#78350f")
ORANGE50 = "#fff7ed"
EMER50, EMER200, EMER500, EMER700 = "#ecfdf5", "#a7f3d0", "#10b981", "#047857"
WHITE = "#ffffff"
PAGE_TOP, PAGE_BOT = "#f9fafb", "#f1faf7"   # gray-50 -> brand-50/40

# differential bar fills, literally from DifferentialChart.tsx
DIFF_FILLS = ["#2e7d6f", "#5cb2a0", "#89cdbd", "#b5e2d7"]


def F(size, weight="r"):
    files = {"r": "segoeui.ttf", "sb": "seguisb.ttf",
             "b": "segoeuib.ttf", "i": "segoeuii.ttf"}
    try:
        return ImageFont.truetype(files[weight], size * S)
    except OSError:
        return ImageFont.truetype("arial.ttf", size * S)


def EMOJI(size):
    try:
        return ImageFont.truetype("seguiemj.ttf", size * S)
    except OSError:
        return None


_DEJAVU = None


def SYM(size):
    """DejaVu Sans, for the arrow/triangle/warning glyphs Segoe UI lacks."""
    global _DEJAVU
    if _DEJAVU is None:
        import matplotlib.font_manager as fm
        _DEJAVU = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    return ImageFont.truetype(_DEJAVU, int(size * S))


def px(v):
    return int(round(v * S))


def hex2(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def lerp(a, b, t):
    ca, cb = hex2(a), hex2(b)
    return tuple(int(round(ca[i] + (cb[i] - ca[i]) * t)) for i in range(3))


def grad(size, c1, c2, diagonal=False):
    """Linear gradient image: vertical, or top-left -> bottom-right."""
    w, h = size
    g = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(g)
    if diagonal:
        n = w + h
        for i in range(n):
            d.line([(i, 0), (0, i)], fill=lerp(c1, c2, i / max(n - 1, 1)))
    else:
        for y in range(h):
            d.line([(0, y), (w, y)], fill=lerp(c1, c2, y / max(h - 1, 1)))
    return g


def rr(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle([px(box[0]), px(box[1]), px(box[2]), px(box[3])],
                        radius=px(r), fill=fill, outline=outline,
                        width=px(width) if outline else 0)


def rr_grad(img, box, r, c1, c2, diagonal=True):
    x0, y0, x1, y1 = (px(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    g = grad((w, h), c1, c2, diagonal).convert("RGBA")
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=px(r), fill=255)
    img.paste(g, (x0, y0), m)


def circle(d, cx, cy, rad, fill=None, outline=None, width=1):
    d.ellipse([px(cx - rad), px(cy - rad), px(cx + rad), px(cy + rad)],
              fill=fill, outline=outline, width=px(width) if outline else 0)


def T(d, xy, s, font, fill, anchor="la"):
    d.text((px(xy[0]), px(xy[1])), s, font=font, fill=fill, anchor=anchor)


def EM(img, d, xy, ch, size):
    """Draw a colour emoji; fall back to a neutral dot if unavailable."""
    f = EMOJI(size)
    if f is None:
        circle(d, xy[0] + size / 2, xy[1] + size / 2, size / 2.6, fill=GRAY[300])
        return
    try:
        d.text((px(xy[0]), px(xy[1])), ch, font=f, embedded_color=True)
    except Exception:
        circle(d, xy[0] + size / 2, xy[1] + size / 2, size / 2.6, fill=GRAY[300])


def wrap(d, s, font, maxw):
    out, line = [], ""
    for word in s.split():
        trial = (line + " " + word).strip()
        if d.textlength(trial, font=font) <= px(maxw) or not line:
            line = trial
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def para(d, xy, s, font, fill, maxw, leading):
    x, y = xy
    for ln in wrap(d, s, font, maxw):
        T(d, (x, y), ln, font, fill)
        y += leading
    return y


def canvas(w, h, top=PAGE_TOP, bot=PAGE_BOT):
    img = grad((px(w), px(h)), top, bot).convert("RGB")
    return img, ImageDraw.Draw(img)


def finish(img, name, w, h):
    img = img.resize((w, h), Image.LANCZOS)
    img.save(OUT / name)
    print(f"  wrote figures/{name}  ({w}x{h})")


# =========================================================================
# shared chrome
# =========================================================================
def header(img, d, w):
    """The real app header: logo, wordmark, status pill, tool buttons."""
    d.rectangle([0, 0, px(w), px(58)], fill="#fbfdfc")
    d.line([(0, px(58)), (px(w), px(58))], fill=GRAY[200], width=px(1))

    rr_grad(img, (16, 9, 56, 49), 11, BRAND[500], INDIGO600)
    EM(img, d, (26, 18), "\U0001F33F", 20)

    T(d, (66, 14), "Arogya ", F(15, "b"), GRAY[800])
    off = 66 + d.textlength("Arogya ", font=F(15, "b")) / S
    T(d, (off, 14), "AI", F(15, "b"), BRAND[600])
    T(d, (66, 34), "Conversational health guidance", F(9), GRAY[500])

    x = w - 16
    for label, fnt, col in [("New chat", F(10, "sb"), BRAND[600]),
                            ("Evaluation", F(10), GRAY[600]),
                            ("NLP Lab", F(10), GRAY[600]),
                            ("Summary", F(10), GRAY[600])]:
        tw = d.textlength(label, font=fnt) / S
        T(d, (x - tw, 23), label, fnt, col)
        x -= tw
        if label == "New chat":                      # reload glyph
            T(d, (x - 13, 22), "↻", SYM(10), col)
            x -= 15
        x -= 18
    T(d, (x, 23), "Online", F(10), GRAY[500], anchor="ra")
    ow = d.textlength("Online", font=F(10)) / S
    circle(d, x - ow - 9, 29, 3.5, fill=EMER500)


def composer(img, d, w, y, cx0, cx1, placeholder):
    """The real composer bar: mic, input, gradient Send button."""
    rr(d, (cx0, y, cx1, y + 44), 14, fill=WHITE, outline=GRAY[200], width=1)
    EM(img, d, (cx0 + 12, y + 13), "\U0001F3A4", 17)
    T(d, (cx0 + 42, y + 15), placeholder, F(10.5), GRAY[400])
    bw = 78
    rr_grad(img, (cx1 - bw - 7, y + 7, cx1 - 7, y + 37), 10, BRAND[500], INDIGO600)
    T(d, (cx1 - bw / 2 - 14, y + 15), "Send", F(10, "sb"), WHITE, anchor="ma")
    T(d, (cx1 - bw / 2 + 8, y + 15), "➤", SYM(9), WHITE)
    T(d, ((cx0 + cx1) / 2, y + 52), "Educational tool only · not a medical diagnosis",
      F(8.5), GRAY[400], anchor="ma")


def user_bubble(img, d, right, y, text, maxw):
    f = F(11)
    lines = wrap(d, text, f, maxw - 32)
    h = 20 + len(lines) * 17
    tw = max(d.textlength(l, font=f) / S for l in lines) + 32
    x0 = right - 38 - tw
    rr_grad(img, (x0, y, x0 + tw, y + h), 14, INDIGO600, VIOLET600)
    yy = y + 10
    for l in lines:
        T(d, (x0 + 16, yy), l, f, WHITE)
        yy += 17
    rr_grad(img, (right - 32, y + h - 32, right, y + h), 16, SLATE400, SLATE600)
    T(d, (right - 16, y + h - 22), "You", F(8.5, "sb"), WHITE, anchor="ma")
    return y + h


def assistant_open(img, d, x, y):
    rr_grad(img, (x, y, x + 32, y + 32), 16, BRAND[500], INDIGO600)
    EM(img, d, (x + 7, y + 7), "\U0001F33F", 17)


def differential(d, x, y, w, rows):
    """DifferentialChart.tsx: label | track | percent, single-hue ramp."""
    T(d, (x, y), "DIFFERENTIAL — HOW THE MODEL RANKED THE CANDIDATES",
      F(8, "sb"), GRAY[400])
    y += 16
    top = max(r[1] for r in rows)
    lab_w, pct_w, gap = 92, 30, 10
    for i, (name, prob) in enumerate(rows):
        lead = i == 0
        T(d, (x, y + 1), name, F(9.5, "sb" if lead else "r"),
          GRAY[800] if lead else GRAY[500])
        tx0 = x + lab_w + gap
        tx1 = x + w - pct_w - gap
        rr(d, (tx0, y + 3, tx1, y + 11), 4, fill=GRAY[100])
        frac = max(prob / top, 0.02)
        rr(d, (tx0, y + 3, tx0 + (tx1 - tx0) * frac, y + 11), 4,
           fill=DIFF_FILLS[min(i, 3)])
        T(d, (x + w, y + 1), f"{round(prob * 100)}%", F(9.5, "sb" if lead else "r"),
          GRAY[700] if lead else GRAY[400], anchor="ra")
        y += 19
    y += 2
    y = para(d, (x, y),
             "Posterior probabilities from the Naive Bayes classifier. They describe "
             "fit to the symptom pattern, not certainty of a diagnosis.",
             F(8), GRAY[400], w, 11)
    return y


# =========================================================================
# 1. desktop chat + differential
# =========================================================================
def fig_chat():
    W, H = 1400, 880
    img, d = canvas(W / S * 0 + 700, 440)   # logical 700x440
    w, h = 700, 440
    header(img, d, w)

    cx0, cx1 = 150, 550
    y = 76
    y = user_bubble(img, d, cx1, y,
                    "I have high fever, chills, headache and body pain for three days",
                    300) + 14

    assistant_open(img, d, cx0, y)
    bx = cx0 + 40
    bw = cx1 - bx
    f = F(11)
    msg = ("Thanks — that pattern is consistent with a few things. Based on what "
           "you've told me, here is how the candidates rank.")
    lines = wrap(d, msg, f, bw - 32)
    bh = 20 + len(lines) * 17 + 134
    rr(d, (bx, y, bx + bw, y + bh), 14, fill=WHITE, outline=GRAY[100], width=1)
    yy = y + 11
    for l in lines:
        T(d, (bx + 16, yy), l, f, GRAY[800])
        yy += 17
    differential(d, bx + 16, yy + 8, bw - 32,
                 [("Malaria", 0.836), ("Common cold", 0.039),
                  ("Dengue", 0.036), ("Typhoid", 0.034)])
    y += bh + 8
    trace_line = "{}  How this answer was produced · 4 stages · 82ms  "
    T(d, (bx, y), trace_line, F(9), GRAY[400])
    T(d, (bx + d.textlength(trace_line, font=F(9)) / S, y + 1), "▾",
      SYM(8), GRAY[400])

    composer(img, d, w, 368, cx0 - 36, cx1 + 36,
             "Describe your symptoms… e.g. “fever and headache since yesterday”")
    finish(img, "screenshot_chat.png", W, H)


# =========================================================================
# 2. emergency response
# =========================================================================
def fig_emergency():
    W, H = 1050, 700
    w, h = 525, 350
    img, d = canvas(w, h)
    header(img, d, w)

    y = 74
    y = user_bubble(img, d, 500, y,
                    "severe crushing chest pain spreading to my left arm, "
                    "sweating and short of breath", 290) + 14

    bx0, bx1 = 24, 500
    bh = 188
    rr_grad(img, (bx0, y, bx1, y + bh), 16, RED600, ROSE700)
    circle(d, bx0 + 42, y + 40, 20, fill=(255, 255, 255, 40))
    d.ellipse([px(bx0 + 22), px(y + 20), px(bx0 + 62), px(y + 60)], fill="#e8515f")
    EM(img, d, (bx0 + 31, y + 29), "\U0001F6A8", 22)

    tx = bx0 + 76
    tw = bx1 - tx - 22
    T(d, (tx, y + 18), "MEDICAL EMERGENCY DETECTED", F(13, "b"), WHITE)
    yy = y + 40
    f = F(9.5)
    for ln in wrap(d, "What you're describing could be a medical emergency. Please seek "
                      "immediate in-person or emergency medical care now. Do not rely on "
                      "this tool for emergency situations.", f, tw):
        T(d, (tx, yy), ln, f, "#ffe9ec")
        yy += 14
    yy += 6

    bh2 = 54
    rr(d, (tx, yy, bx1 - 22, yy + bh2), 9, fill="#8e1322")
    T(d, (tx + 12, yy + 9), "Possible cardiac emergency — chest pain with arm pain, "
      "sweating and breathlessness.", F(8.5), "#ffd9de")
    T(d, (tx + 12, yy + 25), "In India: dial", F(9, "sb"), WHITE)
    cxx = tx + 12 + d.textlength("In India: dial ", font=F(9, "sb")) / S
    for num, tail in (("112", " for emergency services or "), ("108", " for ambulance.")):
        rr(d, (cxx, yy + 24, cxx + 24, yy + 38), 4, fill=WHITE)
        T(d, (cxx + 12, yy + 26), num, F(9, "b"), RED700, anchor="ma")
        cxx += 27
        T(d, (cxx, yy + 25), tail, F(9), WHITE)
        cxx += d.textlength(tail, font=F(9)) / S
    T(d, (tx + 12, yy + 41), "For other countries, use your local emergency number.",
      F(8), "#ffc2ca")

    yy += bh2 + 10
    rr(d, (tx, yy, tx + 150, yy + 24), 8, fill=WHITE)
    T(d, (tx + 75, yy + 6), "I understand — return to chat", F(8.5, "sb"),
      RED600, anchor="ma")

    # The observation that no differential is produced belongs in the paper's
    # caption, not stamped inside a rendering of the interface.
    finish(img, "screenshot_emergency.png", W, H)


# =========================================================================
# 3. triage + evidence + Ayurveda
# =========================================================================
def fig_result():
    W, H = 1050, 1010
    w, h = 525, 505
    img, d = canvas(w, h)
    x0, x1 = 20, 505
    y = 16

    # --- triage card (urgent) ---------------------------------------------
    th = 176
    rr(d, (x0, y, x1, y + th), 14, fill="#fef5f5", outline=RED200, width=1)
    EM(img, d, (x0 + 14, y + 14), "⚡", 15)
    T(d, (x0 + 40, y + 13), "RECOMMENDED NEXT STEP", F(8, "sb"), GRAY[400])
    T(d, (x0 + 40, y + 26), "See a doctor today", F(11.5, "b"), RED800)
    para(d, (x0 + 40, y + 43),
         "Your answers point to a presentation that should be assessed in person "
         "without delay.", F(9), GRAY[600], x1 - x0 - 60, 13)

    d.line([(px(x0 + 14), px(y + 74)), (px(x1 - 14), px(y + 74))], fill=GRAY[200], width=px(1))
    T(d, (x0 + 14, y + 80), "Based on your answers", F(8.5, "sb"), GRAY[500])
    cx = x0 + 14
    for k, v in [("duration", "1-3 days"), ("intensity", "severe"),
                 ("progression", "getting worse"), ("temperature", "above 102°F")]:
        label = f"{k}: {v}"
        cw = d.textlength(label, font=F(8.5)) / S + 16
        rr(d, (cx, y + 94, cx + cw, y + 110), 8, fill=WHITE, outline=GRAY[200], width=1)
        T(d, (cx + 8, y + 97), k + ":", F(8.5), GRAY[400])
        kw = d.textlength(k + ": ", font=F(8.5)) / S
        T(d, (cx + 8 + kw, y + 97), v, F(8.5), GRAY[600])
        cx += cw + 6

    d.line([(px(x0 + 14), px(y + 118)), (px(x1 - 14), px(y + 118))], fill=GRAY[200], width=px(1))
    T(d, (x0 + 14, y + 124), "Seek care promptly if you notice", F(8.5, "sb"), GRAY[500])
    yy = y + 138
    for item in ["Difficulty breathing or shortness of breath",
                 "Chest pain or pressure"]:
        circle(d, x0 + 18, yy + 4, 2, fill=RED500)
        T(d, (x0 + 26, yy), item, F(8.5), GRAY[600])
        yy += 12
    y += th + 12

    # --- retrieved evidence ------------------------------------------------
    eh = 66
    rr(d, (x0, y, x1, y + eh), 12, fill=WHITE, outline=GRAY[200], width=1)
    T(d, (x0 + 14, y + 11), "RETRIEVED EVIDENCE", F(8, "sb"), GRAY[400])
    rr(d, (x1 - 104, y + 9, x1 - 14, y + 23), 6, fill=GRAY[100])
    T(d, (x1 - 59, y + 11), "kb/malaria · top-1", F(8), GRAY[600], anchor="ma")
    para(d, (x0 + 14, y + 28),
         "Malaria presents with cyclical fever, chills and rigors, headache and body "
         "aches, often with sweating as the fever breaks. Prompt testing and treatment "
         "are important.", F(9), GRAY[700], x1 - x0 - 28, 13)
    y += eh + 12

    # --- Ayurveda card ------------------------------------------------------
    ah = 212
    rr_grad(img, (x0, y, x1, y + ah), 14, AMBER50, ORANGE50, diagonal=True)
    rr(d, (x0, y, x1, y + ah), 14, outline=AMBER200, width=1)
    EM(img, d, (x0 + 14, y + 13), "\U0001F33F", 14)
    T(d, (x0 + 36, y + 14), "Ayurvedic Wellness Guidance", F(10.5, "b"), AMBER900)
    rr(d, (x1 - 86, y + 12, x1 - 14, y + 28), 8, fill=AMBER100, outline=AMBER200, width=1)
    T(d, (x1 - 50, y + 15), "Vata-Kapha", F(8.5, "sb"), AMBER800, anchor="ma")

    yy = y + 38
    for icon, title, body in [
        ("\U0001F957", "Dietary guidance",
         "During fever: sip warm water, consume light soups and dal. Avoid heavy and "
         "fried foods. After fever subsides, eat nutrient-rich warm meals to rebuild "
         "strength."),
        ("\U0001F9D8", "Lifestyle suggestions",
         "Rest completely during fever episodes. Stay under mosquito nets. Keep the "
         "environment clean. Rehydrate adequately. Avoid exertion for weeks after "
         "recovery."),
        ("\U0001F33E", "General herbs note",
         "Ayurvedic traditions mention tulsi and neem for general wellness support."),
    ]:
        EM(img, d, (x0 + 16, yy), "•" if icon is None else icon, 13)
        T(d, (x0 + 40, yy), title, F(9.5, "sb"), AMBER900)
        yy = para(d, (x0 + 40, yy + 13), body, F(9), "#8a5a1a", x1 - x0 - 60, 12) + 6

    rr(d, (x0 + 14, yy, x1 - 14, yy + 30), 8, fill="#fdeec4")
    T(d, (x0 + 24, yy + 6), "⚠", SYM(8), AMBER700)
    para(d, (x0 + 38, yy + 6),
         "Malaria is a serious infection requiring prompt medical treatment. "
         "Always complete prescribed medication. This is general wellness information "
         "only.", F(8), AMBER700, x1 - x0 - 62, 11)

    finish(img, "screenshot_result.png", W, H)


# =========================================================================
# 4. expanded pipeline trace
# =========================================================================
def fig_trace_ui():
    W, H = 1050, 430
    w, h = 525, 215
    img, d = canvas(w, h)
    x0, x1 = 22, 503
    y = 20

    head = "{}  How this answer was produced · 4 stages · 82ms  "
    T(d, (x0, y), head, F(10, "sb"), GRAY[500])
    T(d, (x0 + d.textlength(head, font=F(10, "sb")) / S, y + 2), "▴",
      SYM(8), GRAY[500])
    y += 24

    rail = x0 + 10
    stages = [
        ("Symptom extraction", "ok", "27.0", EMER500,
         "6 canonical symptoms resolved; 1 negation removed"),
        ("Red-flag safety scan", "ok", "2.4", EMER500,
         "8 categories evaluated, no pattern matched"),
        ("Naive Bayes prediction", "ok", "45.3", EMER500,
         "41 conditions ranked; leading posterior 83.6%"),
        ("Follow-up selection", "ok", "7.5", EMER500,
         "asked 3 of a maximum 6; next slot: temperature"),
    ]
    # Dashed rail first, so the status dots sit on top of it rather than
    # being struck through by it.
    top_y = y
    for yy in range(px(y), px(y + len(stages) * 38 - 18), px(5)):
        d.line([(px(rail), yy), (px(rail), yy + px(2.4))], fill=GRAY[300], width=px(1))

    for name, status, ms, dot, detail in stages:
        circle(d, rail, y + 6, 3.5, fill=dot, outline=WHITE, width=1.5)
        T(d, (rail + 16, y), name, F(10, "sb"), EMER700)
        badge = f"{status} · {ms}ms"
        bw = d.textlength(badge, font=F(8.5)) / S + 14
        rr(d, (x1 - bw, y, x1, y + 14), 4, fill=EMER50, outline=EMER200, width=1)
        T(d, (x1 - bw / 2, y + 2), badge, F(8.5), EMER700, anchor="ma")
        T(d, (rail + 16, y + 15), detail, F(8.5), GRAY[500])
        y += 38
    finish(img, "screenshot_trace.png", W, H)


# =========================================================================
# 5. phone-width layout
# =========================================================================
def fig_mobile():
    W, H = 620, 910
    w, h = 310, 455
    img, d = canvas(w, h)

    d.rectangle([0, 0, px(w), px(46)], fill="#fbfdfc")
    d.line([(0, px(46)), (px(w), px(46))], fill=GRAY[200], width=px(1))
    rr_grad(img, (12, 7, 44, 39), 9, BRAND[500], INDIGO600)
    EM(img, d, (20, 15), "\U0001F33F", 16)
    T(d, (52, 11), "Arogya ", F(13, "b"), GRAY[800])
    T(d, (52 + d.textlength("Arogya ", font=F(13, "b")) / S, 11), "AI", F(13, "b"), BRAND[600])
    T(d, (52, 28), "Conversational health guidance", F(8), GRAY[500])
    circle(d, w - 20, 23, 3.5, fill=EMER500)

    y = 58
    y = user_bubble(img, d, w - 12, y,
                    "I have high fever, chills and headache for three days", 190) + 12

    assistant_open(img, d, 12, y)
    bx, bw = 50, w - 62
    f = F(10)
    lines = wrap(d, "Here is how the candidates rank.", f, bw - 28)
    bh = 18 + len(lines) * 15 + 132
    rr(d, (bx, y, bx + bw, y + bh), 12, fill=WHITE, outline=GRAY[100], width=1)
    yy = y + 10
    for l in lines:
        T(d, (bx + 14, yy), l, f, GRAY[800])
        yy += 15
    differential(d, bx + 14, yy + 6, bw - 28,
                 [("Malaria", 0.836), ("Common cold", 0.039),
                  ("Dengue", 0.036), ("Typhoid", 0.034)])
    y += bh + 10

    th = 64
    rr(d, (12, y, w - 12, y + th), 12, fill="#fef5f5", outline=RED200, width=1)
    EM(img, d, (24, y + 12), "⚡", 13)
    T(d, (46, y + 11), "RECOMMENDED NEXT STEP", F(7.5, "sb"), GRAY[400])
    T(d, (46, y + 23), "See a doctor today", F(10.5, "b"), RED800)
    para(d, (46, y + 38), "Assessed in person without delay.", F(8.5), GRAY[600], w - 80, 11)

    composer(img, d, w, h - 66, 12, w - 12,
             "Describe your symptoms…")
    finish(img, "screenshot_mobile.png", W, H)


if __name__ == "__main__":
    print("Rendering interface figures from the real component source:")
    fig_chat()
    fig_emergency()
    fig_result()
    fig_trace_ui()
    fig_mobile()
    print("Done.")
