"""Quiet Relay — social-preview.png 1280x640 для GitHub.

Тот же визуальный язык, что у welcome.png: грунт, решётка точек,
кольцо-шкала на 120 засечек, плетение из трёх дуг с посчитанным
перехлёстом и один зелёный сигнал. Отличие: формат 2:1, эмблема
сдвинута влево, вордмарк с подзаголовком — справа.

Рисуется в 2x и уменьшается (supersampling).
"""

import math
import os
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1280, 640            # рекомендованный размер GitHub Social preview
SS = 2
CW, CH = W * SS, H * SS

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FONTS = r"C:\Users\Acer\.claude\skills\canvas-design\canvas-fonts"

GROUND = (10, 12, 11)
GRAPHITE_1 = (28, 32, 31)
GRAPHITE_2 = (50, 56, 54)
GRAPHITE_3 = (94, 102, 99)
CHALK = (178, 185, 181)
BONE = (234, 238, 235)
SIGNAL = (16, 163, 127)

img = Image.new("RGB", (CW, CH), GROUND)
d = ImageDraw.Draw(img, "RGBA")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size * SS)


def px(v):
    return int(round(v * SS))


# ── Геометрия: эмблема слева, текст справа ───────────────────
CX = px(370)
CY = CH // 2
R_HERO = px(118)
R_SCALE = int(R_HERO * 1.62)
R_BARREL = int(R_HERO * 4.35)

# I. Грунт
for y in range(CH):
    t = y / CH
    shade = tuple(int(c + (1.0 - abs(t - 0.32) * 1.7) * 7) for c in GROUND)
    d.line([(0, y), (CW, y)], fill=shade)

# II. Решётка точек
STEP = px(31)
for gx in range(STEP, CW, STEP):
    for gy in range(STEP, CH, STEP):
        dist = math.hypot(gx - CX, gy - CY)
        if dist < R_HERO * 1.30:
            continue
        # затухание к правому краю, чтобы текст дышал
        fade_x = max(0.0, 1.0 - max(0, gx - px(880)) / px(520))
        fade = max(0.0, 1.0 - dist / (CW * 0.72)) * fade_x
        a = int((34 + 60 * fade) * fade_x)
        if a <= 2:
            continue
        r = px(1.5) if dist < CW * 0.28 else px(1.1)
        d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=(*GRAPHITE_2, a))

# III. Кольца и засечки
def ring(radius, color, alpha, width=1.0):
    d.ellipse([CX - radius, CY - radius, CX + radius, CY + radius],
              outline=(*color, alpha), width=px(width))


ring(R_BARREL, GRAPHITE_2, 150, 1.4)
ring(R_SCALE, GRAPHITE_2, 190, 1.3)

for i in range(120):
    ang = math.radians(i * 3)
    major = (i % 10 == 0)
    ln = px(11 if major else 5)
    ca, sa = math.cos(ang), math.sin(ang)
    d.line([(CX + ca * R_SCALE, CY + sa * R_SCALE),
            (CX + ca * (R_SCALE + ln), CY + sa * (R_SCALE + ln))],
           fill=(*(GRAPHITE_3 if major else GRAPHITE_2), 225 if major else 140),
           width=px(1.5 if major else 1.0))

# IV. Плетение
WEAVE_R = int(R_HERO * 0.575)
OFF = R_HERO * 0.288
centers = []
for k in range(3):
    a = math.radians(90 + k * 120)
    centers.append((CX + math.cos(a) * OFF, CY + math.sin(a) * OFF))


def intersections(c1, c2, r):
    (x1, y1), (x2, y2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0 or dist >= 2 * r:
        return []
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    h = math.sqrt(r * r - (dist / 2) ** 2)
    ux, uy = -dy / dist, dx / dist
    return [(mx + ux * h, my + uy * h), (mx - ux * h, my - uy * h)]


def ang_at(center, pt):
    return math.degrees(math.atan2(pt[1] - center[1], pt[0] - center[0])) % 360


GAP = math.degrees(px(7) / WEAVE_R)

for i in range(3):
    under = (i - 1) % 3
    pts = intersections(centers[i], centers[under], WEAVE_R)
    if len(pts) != 2:
        continue
    a1, a2 = sorted(ang_at(centers[i], p) for p in pts)
    cx_, cy_ = centers[i]
    box = [cx_ - WEAVE_R, cy_ - WEAVE_R, cx_ + WEAVE_R, cy_ + WEAVE_R]
    d.arc(box, a1 + GAP, a2 - GAP, fill=(*CHALK, 242), width=px(3.0))
    d.arc(box, a2 + GAP, a1 + 360 - GAP, fill=(*CHALK, 242), width=px(3.0))

ring(R_HERO, GRAPHITE_3, 205, 1.5)

# V. Сигнал
d.arc([CX - R_SCALE, CY - R_SCALE, CX + R_SCALE, CY + R_SCALE],
      212, 288, fill=(*SIGNAL, 255), width=px(3.0))

ANG_IN = math.radians(250)
ci, si = math.cos(ANG_IN), math.sin(ANG_IN)
d.line([(CX + ci * R_SCALE, CY + si * R_SCALE),
        (CX + ci * R_HERO, CY + si * R_HERO)], fill=(*SIGNAL, 215), width=px(1.8))

gx, gy = CX + ci * R_SCALE, CY + si * R_SCALE
halo = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
ImageDraw.Draw(halo).ellipse([gx - px(22), gy - px(22), gx + px(22), gy + px(22)],
                             fill=(*SIGNAL, 58))
halo = halo.filter(ImageFilter.GaussianBlur(px(10)))
img = Image.alpha_composite(img.convert("RGBA"), halo).convert("RGB")
d = ImageDraw.Draw(img, "RGBA")
d.ellipse([gx - px(5.5), gy - px(5.5), gx + px(5.5), gy + px(5.5)], fill=SIGNAL)

# VI. Типографика: вордмарк справа
f_mark = font("Outfit-Regular.ttf", 74)
f_mono = font("GeistMono-Regular.ttf", 17)
f_micro = font("GeistMono-Regular.ttf", 13)
f_tiny = font("GeistMono-Regular.ttf", 11)

TX = px(672)   # левый край текстового блока


def tracked(xy, text, fnt, fill, track, anchor_left=True):
    widths = [d.textlength(ch, font=fnt) for ch in text]
    total = sum(widths) + px(track) * (len(text) - 1)
    x, y = xy
    if not anchor_left:
        x -= total / 2
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=fnt, fill=fill)
        x += w + px(track)
    return total


title_y = CY - px(78)
tracked((TX, title_y), "NEXUS GPT", f_mark, BONE, 10)

rule_y = title_y + px(108)
d.line([(TX, rule_y), (TX + px(436), rule_y)],
       fill=(*GRAPHITE_3, 195), width=px(1.2))

tracked((TX, rule_y + px(24)), "TELEGRAM AI BOT", f_mono, CHALK, 7)
tracked((TX, rule_y + px(58)), "CHAT / VISION / VOICE / PAYMENTS",
        f_tiny, (*GRAPHITE_3,), 5)

# VII. Служебная разметка
M = px(40)
CORN = px(22)
for ax, ay, dx, dy in ((M, M, 1, 1), (CW - M, M, -1, 1),
                       (M, CH - M, 1, -1), (CW - M, CH - M, -1, -1)):
    d.line([(ax, ay), (ax + CORN * dx, ay)], fill=(*GRAPHITE_2, 195), width=px(1))
    d.line([(ax, ay), (ax, ay + CORN * dy)], fill=(*GRAPHITE_2, 195), width=px(1))

d.text((M + px(8), M + px(5)), "FIG. 02", font=f_micro, fill=(*GRAPHITE_3, 230))
d.text((CW - M - px(8), M + px(5)), "PL. VII", font=f_micro,
       fill=(*GRAPHITE_3, 230), anchor="ra")
d.text((M + px(8), CH - M - px(19)), "OPENROUTER / ECHOGATE / STARS",
       font=f_tiny, fill=(*GRAPHITE_3, 195))
d.text((CW - M - px(8), CH - M - px(19)), "RELAY", font=f_micro,
       fill=(*SIGNAL, 220), anchor="ra")
# галочка рисуется линиями: в GeistMono нет глифа ✓ (упал бы квадратиком)
rw = d.textlength("RELAY", font=f_micro)
chk_cx = CW - M - px(8) - rw - px(14)
chk_cy = CH - M - px(11)
d.line([(chk_cx - px(5), chk_cy), (chk_cx - px(1), chk_cy + px(4))],
       fill=(*SIGNAL, 220), width=px(1.4))
d.line([(chk_cx - px(1), chk_cy + px(4)), (chk_cx + px(6), chk_cy - px(5))],
       fill=(*SIGNAL, 220), width=px(1.4))

# VIII. Виньетка и зерно
vig = Image.new("L", (CW, CH), 0)
ImageDraw.Draw(vig).ellipse([-CW * 0.30, -CH * 0.45, CW * 1.30, CH * 1.45], fill=255)
vig = vig.filter(ImageFilter.GaussianBlur(px(155)))
img = Image.composite(img, Image.new("RGB", (CW, CH), (3, 4, 4)), vig)

img = img.resize((W, H), Image.LANCZOS)

random.seed(404)
gr = img.load()
for _ in range(int(W * H * 0.032)):
    x, y = random.randrange(W), random.randrange(H)
    r, g, b = gr[x, y]
    n = random.randint(-6, 6)
    gr[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)),
                max(0, min(255, b + n)))

path = os.path.join(OUT_DIR, "social-preview.png")
img.save(path, "PNG", optimize=True)
print("saved:", path, img.size, f"{os.path.getsize(path) / 1024:.0f} KB")
