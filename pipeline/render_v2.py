"""v2 - real footage (untouched) -> dive into the skin -> the AI layered anatomy image ->
needle releasing the fibrous septa + collagen biostimulator -> real "after" footage.

The before/after footage is used as shot: no retouching, no relief enhancement, pen
markings kept. Only framing (digital camera moves / tracking), one global grade and
overlay rings are added on top. The anatomical part is the AI image supplied by the
client, animated with camera moves, traction cords, needle, gel and collagen fibrils.

Usage: python3 render_v2.py <workdir> <ai_image> <out.mp4> [--preview t1,t2,...]
  workdir must contain bf/%04d.jpg (before video 0..10.5 s) and af/%04d.jpg (after video
  from 22 s), extracted with ffmpeg (see README).
"""
import os
import sys
import math
import json
import subprocess

import cv2
import numpy as np

WD, AI_PATH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PREVIEW = None
if "--preview" in sys.argv:
    PREVIEW = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]

FPS, DUR = 30, 15.0
OW, OH = 1080, 1920
rng = np.random.default_rng(11)


# ----------------------------------------------------------------------------- easing
def clamp01(x):
    return min(max(x, 0.0), 1.0)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = clamp01(x)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def seg(t, a, b):
    return clamp01((t - a) / (b - a))


def lerp(a, b, k):
    return a + (b - a) * k


def spring(x):
    """0 -> 1 with a small, damped overshoot (tissue recoil)."""
    if x <= 0:
        return 0.0
    return 1 - math.exp(-7 * x) * math.cos(9 * x)


# ----------------------------------------------------------------------------- footage
def frame_path(d, i):
    return os.path.join(WD, d, "%04d.jpg" % i)


_cache = {}


def load(d, i):
    k = (d, i)
    if k not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        _cache[k] = cv2.imread(frame_path(d, i)).astype(np.float32)
    return _cache[k]


def track(d, i0, i1, pts):
    """LK track pts from frame i0 to i1 (either direction). Returns {frame: Nx2}."""
    step = 1 if i1 >= i0 else -1
    cur = np.float32(pts).reshape(-1, 1, 2)
    prev = cv2.cvtColor(cv2.imread(frame_path(d, i0)), cv2.COLOR_BGR2GRAY)
    out = {i0: cur.reshape(-1, 2).copy()}
    for i in range(i0 + step, i1 + step, step):
        g = cv2.cvtColor(cv2.imread(frame_path(d, i)), cv2.COLOR_BGR2GRAY)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(prev, g, cur, None, winSize=(41, 41), maxLevel=4,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # keep points that failed where they were (rare on this footage)
        nxt[st[:, 0] == 0] = cur[st[:, 0] == 0]
        out[i] = nxt.reshape(-1, 2).copy()
        cur, prev = nxt, g
    return out


def smooth_track(tr, sigma=3.0):
    ks = sorted(tr)
    arr = np.stack([tr[k] for k in ks])          # F x N x 2
    sm = cv2.GaussianBlur(arr.reshape(len(ks), -1).astype(np.float32), (1, 0), sigmaX=0.001, sigmaY=sigma)
    sm = sm.reshape(arr.shape)
    return {k: sm[j] for j, k in enumerate(ks)}


def crop(img, cx, cy, w):
    """Digital camera: centre (cx, cy) and width w in source pixels -> 1080x1920."""
    h = w * OH / OW
    s = OW / w
    M = np.float32([[s, 0, -(cx - w / 2) * s], [0, s, -(cy - h / 2) * s]])
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def to_out(cx, cy, w, p):
    h = w * OH / OW
    s = OW / w
    return (np.float32(p) - [cx - w / 2, cy - h / 2]) * s


# shot A: straight back, before video 0.0-2.0 s (frames 1..60)
A_PTS = [(110, 405), (300, 385), (312, 437), (200, 410)]     # 3 depressions + cluster centre
cache_f = os.path.join(WD, "tracks_v2.json")
if os.path.exists(cache_f):
    J = json.load(open(cache_f))
    TA = {int(k): np.float32(v) for k, v in J["A"].items()}
    TB = {int(k): np.float32(v) for k, v in J["B"].items()}
else:
    TA = smooth_track(track("bf", 1, 66, A_PTS), 2.5)
    # shot B: lateral arc, before video 5.6-8.2 s (frames 169..247); dive target tracked backward
    TB = smooth_track(track("bf", 247, 160, [(360, 470), (395, 505)]), 2.5)
    json.dump({"A": {k: v.tolist() for k, v in TA.items()}, "B": {k: v.tolist() for k, v in TB.items()}},
              open(cache_f, "w"))

SHOT_A = (0.0, 2.05, 1)          # out start, out end, first source frame
SHOT_B = (2.05, 4.75, 166)
SHOT_AF = (12.05, 15.0, 46)      # after video (22 s + 1.5 s)


def ring(img, c, r, prog, alpha):
    if prog <= 0 or alpha <= 0:
        return img
    lay = img.copy()
    cc = (int(c[0] * 16), int(c[1] * 16))
    rr = int(r * 16)
    cv2.ellipse(lay, cc, (rr, rr), -90, 0, 360 * prog, (245, 246, 248), 3, cv2.LINE_AA, 4)
    glow = cv2.GaussianBlur(lay - img, (0, 0), 4)
    lay = lay + glow * 0.6
    return img * (1 - alpha) + lay * alpha


def focus_pull(img, c, strength):
    """Cinematic emphasis: periphery slightly defocused and darker around a point."""
    if strength <= 0:
        return img
    Y, X = np.mgrid[0:OH:8, 0:OW:8].astype(np.float32)
    d = np.sqrt((X - c[0]) ** 2 + ((Y - c[1]) * 0.8) ** 2) / (OW * 0.55)
    m = cv2.resize(np.clip((d - 0.35) / 0.65, 0, 1) ** 1.4, (OW, OH))[..., None] * strength
    bl = cv2.GaussianBlur(img, (0, 0), 7)
    return (img * (1 - m) + bl * m) * (1 - 0.28 * m)


def zoom_blur(img, amount, c):
    """Radial zoom blur for the dive (a few scaled copies averaged)."""
    if amount <= 0.002:
        return img
    acc = img.copy()
    n = 6
    for k in range(1, n):
        s = 1 + amount * k / n
        M = np.float32([[s, 0, c[0] * (1 - s)], [0, s, c[1] * (1 - s)]])
        acc += cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


def shot_a(t):
    a0, a1, f0 = SHOT_A
    fi = f0 + int(round((t - a0) * FPS))
    img = load("bf", fi)
    tr = TA[min(fi, max(TA))]
    k = ease_io(seg(t, a0, a1 + 0.2))
    # push in and drift onto the depression cluster (tracked -> stabilised move)
    w = lerp(576, 400, k)
    cx = lerp(288, tr[3][0] + 20, k)
    cy = lerp(512, tr[3][1] + 40, k)
    out = crop(img, cx, cy, w)
    out = focus_pull(out, to_out(cx, cy, w, tr[3]), 0.75 * smooth(seg(t, 0.7, 1.8)))
    for j in range(3):
        c = to_out(cx, cy, w, tr[j])
        pr = ease_out(seg(t, 0.55 + 0.18 * j, 1.15 + 0.18 * j))
        out = ring(out, c, 24 * OW / w, pr, 0.85 * (1 - smooth(seg(t, 1.85, 2.05))))
    return out


def shot_b(t):
    b0, b1, f0 = SHOT_B
    fi = f0 + int(round((t - b0) * FPS))
    fi = min(fi, 247)
    img = load("bf", fi)
    tr = TB[min(max(fi, min(TB)), max(TB))]
    # follow the real lateral camera arc, then dive into the depression
    k = ease_in(seg(t, b1 - 1.15, b1))
    w = lerp(560, 120, k)
    cx = lerp(300, tr[0][0], ease_io(seg(t, b1 - 1.4, b1 - 0.3)))
    cy = lerp(512, tr[0][1], ease_io(seg(t, b1 - 1.4, b1 - 0.3)))
    out = crop(img, cx, cy, w)
    c = to_out(cx, cy, w, tr[0])
    out = focus_pull(out, c, 0.6 * smooth(seg(t, b1 - 1.4, b1 - 0.8)))
    out = zoom_blur(out, 0.10 * ease_in(seg(t, b1 - 0.6, b1)), c)
    return out


def shot_after(t):
    s0, s1, f0 = SHOT_AF
    fi = f0 + int(round((t - s0) * FPS))
    img = load("af", min(fi, 180))
    k = ease_out(seg(t, s0, s0 + 2.2))
    w = lerp(170, 432, k)
    cx = lerp(250, 322, k)
    cy = lerp(270, 384, k)
    out = crop(img, cx, cy, w)
    out = zoom_blur(out, 0.08 * (1 - ease_out(seg(t, s0, s0 + 0.5))), (OW / 2, OH / 2))
    return out


# ----------------------------------------------------------------------------- AI image
AI = cv2.imread(AI_PATH)                     # 900 x 1600 layered anatomy (client image)
AS = 2                                       # master supersampling
AIH, AIW = AI.shape[0] * AS, AI.shape[1] * AS
AIM = cv2.resize(AI, (AIW, AIH), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
ay, ax = np.mgrid[0:AIH, 0:AIW].astype(np.float32)

# skin dimples on the AI skin dome (this is what the traction cords cause), organic shapes
DIMPLES = [(330, 235, 1.0), (418, 300, 0.9), (560, 228, 0.8), (648, 318, 1.0), (476, 372, 0.85),
           (285, 345, 0.7), (720, 250, 0.7), (520, 300, 0.6), (395, 190, 0.55), (610, 160, 0.5),
           (240, 270, 0.6), (700, 365, 0.65), (365, 395, 0.6)]
noise = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (AIH // 24, AIW // 24)).astype(np.float32), (0, 0), 1.2),
                   (AIW, AIH), interpolation=cv2.INTER_CUBIC)
noise /= noise.std()
noise2 = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (AIH // 8, AIW // 8)).astype(np.float32), (0, 0), 1.0),
                    (AIW, AIH), interpolation=cv2.INTER_CUBIC)
noise2 /= noise2.std()
D = np.zeros((AIH, AIW), np.float32)
for (x, y, s_) in DIMPLES:
    rx_, ry_ = rng.uniform(20, 34) * AS, rng.uniform(9, 15) * AS      # cellulite: wide, shallow, elongated
    th = rng.uniform(-0.45, 0.45)
    dx, dy = ax - x * AS, ay - y * AS
    u = dx * math.cos(th) + dy * math.sin(th)
    v = -dx * math.sin(th) + dy * math.cos(th)
    r = np.sqrt((u / rx_) ** 2 + (v / ry_) ** 2) * (1 + 0.16 * noise + 0.05 * noise2)
    D += s_ * np.exp(-r ** 2 * 1.3)
D = cv2.GaussianBlur(D, (0, 0), 3.0)
gyD, gxD = np.gradient(D)
skin_mask = np.zeros((AIH, AIW), np.float32)
cv2.fillPoly(skin_mask, [np.int32([[110, 60], [860, 60], [860, 420], [110, 420]]) * AS], 1)
skin_mask = cv2.GaussianBlur(skin_mask, (0, 0), 20)


def apply_dimples(img, amt):
    if amt <= 0:
        return img
    # pulled-in surface: shaded with the image's own light (from the upper-left)
    lam = (-0.5 * gxD - 0.75 * gyD) * AS * 6.0          # Lambert term of the pitted surface
    sh = 1 + amt * skin_mask * (lam - 0.07 * D)
    return img * np.clip(sh, 0.78, 1.15)[..., None]


# traction cords: fibrous bundles from the underside of the fat to the septal layer
CORDS = [  # x, top (fat underside), bottom (deep septa)   in AI px
    (255, 628, 800), (385, 636, 812), (566, 642, 818), (716, 650, 828)]
NEEDLE_A = np.float32([960, 760])       # entry (off-frame right)
NEEDLE_B = np.float32([150, 724])       # deepest point (left)


def needle_y(x):
    k = (x - NEEDLE_A[0]) / (NEEDLE_B[0] - NEEDLE_A[0])
    return NEEDLE_A[1] + (NEEDLE_B[1] - NEEDLE_A[1]) * k


T_IN = (7.70, 9.35)        # needle travels in (cutting)
T_OUT = (9.50, 10.45)      # needle withdraws (depositing biostimulator)


def tip_x(t):
    if t < T_IN[0]:
        return None
    if t <= T_IN[1]:
        return lerp(NEEDLE_A[0], NEEDLE_B[0], ease_io(seg(t, *T_IN)))
    if t <= T_OUT[0]:
        return NEEDLE_B[0]
    if t <= T_OUT[1]:
        return lerp(NEEDLE_B[0], NEEDLE_A[0] + 20, ease_in(seg(t, *T_OUT)) * 0.35 + ease_io(seg(t, *T_OUT)) * 0.65)
    return None


def cut_time(x):
    # time at which the tip passes x on the way in
    lo, hi = T_IN
    for _ in range(30):
        mid = (lo + hi) / 2
        if tip_x(mid) > x:
            lo = mid
        else:
            hi = mid
    return hi


FIBERS = []
for ci, (x0, top, bot) in enumerate(CORDS):
    fl = []
    n = 10
    for i in range(n):
        u = (i / (n - 1)) * 2 - 1 + rng.uniform(-0.12, 0.12)
        fl.append(dict(u=u, ph=rng.uniform(0, 6.28), sh=rng.uniform(0.82, 1.08),
                       jy=rng.uniform(-7, 7), jt=rng.uniform(0, 0.16), curl=rng.uniform(0.6, 1.4) * rng.choice([-1, 1]),
                       th=rng.choice([1, 1, 2])))
    FIBERS.append(fl)
CUT_T = [cut_time(c[0]) for c in CORDS]


def cord_state(ci, t):
    return [spring(max(t - CUT_T[ci] - f["jt"], 0) / 0.55) if t >= CUT_T[ci] + f["jt"] else 0.0
            for f in FIBERS[ci]]


def fiber_path(x0, top, bot, f, s0, s1, slack):
    s = np.linspace(s0, s1, 24)
    w = np.sin(np.pi * s) ** 0.7
    xt = x0 + f["u"] * 13                    # flare into the tissue at both ends
    xb = x0 + f["u"] * 15
    x = lerp(xt, xb, s) * (1 - w) + (x0 + f["u"] * 6.5) * w
    x = x + (2.2 + 5 * slack) * np.sin(2 * np.pi * s * 2.2 + f["ph"]) * w
    y = lerp(top, bot, s)
    return np.stack([x, y], 1)


def draw_cords(img, t, roi):
    """Draw cords into the master image (AI px * AS). Returns updated img."""
    x0r, y0r, x1r, y1r = roi
    lay = np.zeros((y1r - y0r, x1r - x0r, 3), np.float32)
    a = np.zeros((y1r - y0r, x1r - x0r), np.float32)
    shadow = np.zeros_like(a)
    for ci, (x0, top, bot) in enumerate(CORDS):
        st = cord_state(ci, t)
        lift = np.mean(st) * 9          # fat underside springs back up when released
        top_ = top - lift * 0.0
        yc = needle_y(x0)
        sc = (yc - top_) / (bot - top_)
        for f, p in zip(FIBERS[ci], st):
            col = CORD_COL * f["sh"]                              # sampled from the AI septa
            hi_ = np.minimum(CORD_COL * 1.22, 255) * f["sh"]
            if p <= 0:
                parts = [(0.0, 1.0, 0.0, 0.0)]
            else:
                scj = sc + f["jy"] / (bot - top_)
                parts = [(0.0, scj - (scj - 0.08) * 0.62 * p, p, -1),
                         (scj + (1 - scj - 0.08) * 0.62 * p, 1.0, p, 1)]
            for (s0, s1, pp, side) in parts:
                P = fiber_path(x0, top_, bot, f, s0, s1, pp)
                if side:
                    # recoiling free end curls away from the cut
                    k = np.linspace(0, 1, len(P)) if side < 0 else np.linspace(1, 0, len(P))
                    P[:, 0] += f["curl"] * 9 * pp * k ** 2
                    P[:, 1] += -side * 3 * pp * k ** 2
                q = np.round((P * AS - [x0r, y0r]) * 16).astype(np.int32)
                th = f["th"] * (1 if pp == 0 else 1)
                cv2.polylines(shadow, [q + [int(2.5 * 16), int(3 * 16)]], False, 0.7, th + 1, cv2.LINE_AA, 4)
                cv2.polylines(lay, [q], False, col.tolist(), th + 1, cv2.LINE_AA, 4)
                cv2.polylines(a, [q], False, 0.88, th + 1, cv2.LINE_AA, 4)
                if rng.random() < 0.5:
                    cv2.polylines(lay, [q - [8, 8]], False, hi_.tolist(), 1, cv2.LINE_AA, 4)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 3)[..., None]
    a = cv2.GaussianBlur(a, (0, 0), 0.7)[..., None]
    lay = cv2.GaussianBlur(lay, (0, 0), 0.6)
    reg = img[y0r:y1r, x0r:x1r]
    reg = reg * (1 - 0.35 * shadow)
    reg = reg * (1 - a) + lay * a
    img[y0r:y1r, x0r:x1r] = reg
    return img


def fat_pull(img, t):
    """The cords drag the fat underside down; released cords let it spring back."""
    y0r, y1r = 560 * AS, 720 * AS
    x0r, x1r = 160 * AS, 800 * AS
    yy, xx = ay[y0r:y1r, x0r:x1r], ax[y0r:y1r, x0r:x1r]
    dy = np.zeros_like(yy)
    for ci, (x0, top, bot) in enumerate(CORDS):
        rel = np.mean(cord_state(ci, t))
        A = 8 * AS * (1 - rel)
        dy += A * np.exp(-((xx - x0 * AS) / (26 * AS)) ** 2) * np.exp(-((yy - top * AS) / (30 * AS)) ** 2)
    if np.max(dy) < 0.2:
        return img
    reg = cv2.remap(img, xx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = img.copy()
    img[y0r:y1r, x0r:x1r] = reg
    return img


# septal fibres in the AI image itself (to embed the needle *inside* the web)
_g = cv2.cvtColor(AIM.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
fib = np.clip((_g - cv2.GaussianBlur(_g, (0, 0), 4)) / 18, 0, 1)
band = np.zeros_like(fib)
band[600 * AS:840 * AS] = 1
FIBMASK = cv2.GaussianBlur(fib * cv2.GaussianBlur(band, (0, 0), 10), (0, 0), 0.8)
CORD_COL = np.float32(AIM[(FIBMASK > 0.5)].mean(0)) * 1.05     # tone of the septal fibres


def draw_needle(img, t):
    tx = tip_x(t)
    if tx is None:
        return img
    tip = np.float32([tx, needle_y(tx)]) * AS
    tail = np.float32([NEEDLE_A[0] + 80, needle_y(NEEDLE_A[0] + 80)]) * AS
    d = tip - tail
    L = np.linalg.norm(d)
    dn = d / L
    nrm = np.float32([-dn[1], dn[0]])
    R = 5.5 * AS                                       # cannula radius
    lay = np.zeros_like(img)
    m = np.zeros(img.shape[:2], np.float32)
    sh = np.zeros(img.shape[:2], np.float32)
    # metallic cylinder: stack of parallel lines across the diameter
    for k in np.linspace(-1, 1, 23):
        lum = 0.30 + 1.15 * math.exp(-((k + 0.42) / 0.16) ** 2) + 0.25 * math.exp(-((k - 0.6) / 0.2) ** 2)
        base = np.float32([170, 174, 180]) * (0.35 + 0.65 * math.sqrt(max(0, 1 - k * k)))
        col = np.clip(base * lum + np.float32([40, 30, 20]) * max(0, k) * 0.6, 0, 255)
        o = nrm * k * R
        p0 = tuple(np.int32((tail + o) * 16))
        p1 = tuple(np.int32((tip - dn * R * 2.2 * (abs(k) ** 0.8) + o) * 16))   # bevelled point
        cv2.line(lay, p0, p1, col.tolist(), 2, cv2.LINE_AA, 4)
        cv2.line(m, p0, p1, 1.0, 2, cv2.LINE_AA, 4)
    cv2.line(sh, tuple(np.int32((tail + nrm * 9 * AS) * 16)), tuple(np.int32((tip + nrm * 9 * AS) * 16)),
             1.0, int(2 * R), cv2.LINE_AA, 4)
    sh = cv2.GaussianBlur(sh, (0, 0), 7 * AS)[..., None]
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.5)
    # inside the tissue: septal fibres pass in front of the shaft here and there
    m = m * (1 - 0.55 * FIBMASK)
    img = img * (1 - 0.35 * sh)
    return img * (1 - m[..., None]) + lay * m[..., None]


# biostimulator gel deposited along the retreating needle + collagen fibrils
DROPS = []
x = NEEDLE_B[0] + 25
while x < 870:
    DROPS.append(dict(x=x + rng.uniform(-6, 6), y=needle_y(x) + rng.uniform(-6, 6), r=rng.uniform(6, 11),
                      e=rng.uniform(0.7, 1.0), a=rng.uniform(0, 3.14)))
    x += rng.uniform(20, 30)
FIBRILS = []
for dp in DROPS:
    for _ in range(7):
        ang = rng.uniform(0, 2 * math.pi)
        L_ = rng.uniform(18, 46)
        steps = 14
        pts = [(dp["x"], dp["y"])]
        a_ = ang
        for _s in range(steps):
            a_ += rng.normal(0, 0.28)
            px, py = pts[-1]
            pts.append((px + math.cos(a_) * L_ / steps, py + math.sin(a_) * L_ / steps))
        FIBRILS.append((np.float32(pts), rng.uniform(0, 0.5), dp["x"]))


def drop_time(x):
    lo, hi = T_OUT
    for _ in range(30):
        mid = (lo + hi) / 2
        if tip_x(mid) < x:
            lo = mid
        else:
            hi = mid
    return hi


for dp in DROPS:
    dp["t"] = drop_time(dp["x"])


def draw_bio(img, t, roi):
    if t < T_OUT[0]:
        return img
    x0r, y0r, x1r, y1r = roi
    reg = img[y0r:y1r, x0r:x1r]
    gel = np.zeros(reg.shape[:2], np.float32)
    for dp in DROPS:
        g = ease_out(seg(t, dp["t"], dp["t"] + 0.45))
        if g <= 0:
            continue
        c = (int((dp["x"] * AS - x0r) * 16), int((dp["y"] * AS - y0r) * 16))
        ax_ = (int(dp["r"] * AS * g * 16), int(dp["r"] * AS * g * dp["e"] * 16))
        cv2.ellipse(gel, c, ax_, math.degrees(dp["a"]), 0, 360, 0.8, -1, cv2.LINE_AA, 4)
        # the bolus spreads along the tunnel left by the cannula
        cl = (int((dp["x"] * AS - x0r - 14 * AS * g) * 16), c[1])
        cr = (int((dp["x"] * AS - x0r + 14 * AS * g) * 16), c[1])
        cv2.line(gel, cl, cr, 0.6, int(5 * AS * g), cv2.LINE_AA, 4)
    if gel.max() > 0:
        soft = cv2.GaussianBlur(gel, (0, 0), 3.5 * AS)
        core = cv2.GaussianBlur(gel, (0, 0), 1.2 * AS)
        gy_, gx_ = np.gradient(soft)
        spec = np.clip((-gx_ * 0.5 - gy_ * 0.8) * 40, 0, 1) * core
        core = np.clip(core, 0, 1)
        a = (0.40 * core + 0.20 * soft)[..., None]
        milky = np.float32([214, 226, 236])
        reg = reg * (1 - a) + milky * a                       # translucent pearl gel
        reg = reg + np.float32([250, 252, 255]) * (0.35 * spec)[..., None]
        reg = reg * (1 - 0.10 * (soft - core).clip(0, 1))[..., None]   # faint refraction rim
    # collagen fibrils growing from the deposits (slow, delicate)
    lay = np.zeros(reg.shape[:2], np.float32)
    for pts, dl, xs in FIBRILS:
        t0 = drop_time(xs) + 0.35 + dl
        gr = ease_out(seg(t, t0, t0 + 1.2))
        if gr <= 0:
            continue
        n = max(2, int(len(pts) * gr))
        q = np.round((pts[:n] * AS - [x0r, y0r]) * 16).astype(np.int32)
        cv2.polylines(lay, [q], False, 1.0, 2, cv2.LINE_AA, 4)
    if lay.max() > 0:
        lay = cv2.GaussianBlur(lay, (0, 0), 0.8)[..., None]
        glow = cv2.GaussianBlur(lay[..., 0], (0, 0), 6)[..., None]
        col = np.float32([196, 226, 246])                 # warm ivory collagen
        reg = reg * (1 - 0.7 * lay) + col * 0.7 * lay + col * 0.14 * glow
    img[y0r:y1r, x0r:x1r] = reg
    return img


# AI camera keys: time, centre x, centre y, width  (AI px)
AI_CAM = [
    (4.75, 470, 300, 150),
    (5.25, 470, 330, 330),
    (6.40, 452, 720, 880),
    (6.95, 452, 730, 860),
    (7.55, 500, 728, 430),
    (11.2, 505, 728, 388),
    (11.7, 470, 285, 340),
    (12.15, 468, 290, 170),
]


def ai_cam(t):
    for (t0, x0, y0, w0), (t1, x1, y1, w1) in zip(AI_CAM[:-1], AI_CAM[1:]):
        if t0 <= t <= t1:
            k = ease_io((t - t0) / (t1 - t0))
            return lerp(x0, x1, k), lerp(y0, y1, k), math.exp(lerp(math.log(w0), math.log(w1), k))
    k = AI_CAM[0] if t < AI_CAM[0][0] else AI_CAM[-1]
    return k[1], k[2], k[3]


def light_sweep(img, t):
    """A soft specular pass travelling across the stack (cinematic light move)."""
    k = seg(t, 5.6, 7.0)
    if k <= 0 or k >= 1:
        return img
    pos = lerp(-0.3, 1.3, ease_io(k))
    X = (ax / AIW + ay / AIH * 0.35) - pos * 1.35
    band_ = np.exp(-(X / 0.08) ** 2) * math.sin(math.pi * k)
    return img * (1 + 0.16 * band_[..., None])


def shot_ai(t):
    img = AIM.copy()
    dim = 1 - smooth(seg(t, 11.2, 11.95))                # skin dimples relax after release
    rel_all = np.mean([np.mean(cord_state(ci, t)) for ci in range(len(CORDS))])
    img = apply_dimples(img, dim * (1 - 0.35 * rel_all))
    img = fat_pull(img, t)
    roi = (100 * AS, 560 * AS, 880 * AS, 900 * AS)
    img = draw_cords(img, t, roi)
    img = draw_bio(img, t, roi)
    img = draw_needle(img, t)
    img = light_sweep(img, t)
    cx, cy, w = ai_cam(t)
    out = crop(img * 1.0, cx * AS, cy * AS, w * AS)
    if t < 5.25:
        out = zoom_blur(out, 0.10 * (1 - ease_out(seg(t, 4.75, 5.25))), (OW / 2, OH / 2))
    if t > 11.75:
        out = zoom_blur(out, 0.10 * ease_in(seg(t, 11.75, 12.15)), (OW / 2, OH / 2))
    return out


# ----------------------------------------------------------------------------- grade / main
_vig = None


def grade(x, t):
    global _vig
    x = np.clip(x / 255.0, 0, 1)
    x = np.clip((x - 0.45) * 1.06 + 0.45, 0, 1)
    x = 0.014 + x * 0.98                                  # soft blacks
    # halation: warm bloom around highlights
    hl = cv2.GaussianBlur(np.clip(x - 0.72, 0, 1), (0, 0), 18)
    x = x + hl * np.float32([0.05, 0.10, 0.22])
    if _vig is None:
        Y, X = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = np.sqrt(((X - OW / 2) / (OW * 0.72)) ** 2 + ((Y - OH / 2) / (OH * 0.70)) ** 2)
        _vig = (1 - 0.26 * np.clip(r, 0, 1) ** 2.3)[..., None]
    x = x * _vig
    bl = cv2.GaussianBlur(x, (0, 0), 1.4)
    x = x + 0.30 * (x - bl)
    g = np.random.default_rng(int(t * 1000)).normal(0, 0.011, (OH // 2, OW // 2)).astype(np.float32)
    x = x + cv2.resize(g, (OW, OH))[..., None] * (0.55 + 0.45 * x)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def frame(t):
    if t < SHOT_A[1]:
        out = shot_a(t)
        xf = seg(t, SHOT_A[1] - 0.12, SHOT_A[1])        # short dissolve into the lateral arc
        if xf > 0:
            out = out * (1 - xf) + shot_b(SHOT_A[1]) * xf
    elif t < SHOT_B[1] - 0.3:
        out = shot_b(t)
    elif t < SHOT_B[1]:
        k = smooth(seg(t, SHOT_B[1] - 0.3, SHOT_B[1]))
        out = shot_b(t) * (1 - k) + shot_ai(AI_CAM[0][0]) * k
    elif t < SHOT_AF[0] - 0.3:
        out = shot_ai(t)
    elif t < SHOT_AF[0]:
        k = smooth(seg(t, SHOT_AF[0] - 0.3, SHOT_AF[0]))
        out = shot_ai(t) * (1 - k) + shot_after(SHOT_AF[0]) * k
    else:
        out = shot_after(t)
    fade = smooth(seg(t, 0, 0.25)) * (1 - smooth(seg(t, DUR - 0.35, DUR)))
    return grade(out * fade, t)


if PREVIEW:
    for t in PREVIEW:
        cv2.imwrite(OUT.replace(".mp4", "_%05.2f.jpg" % t), frame(t), [cv2.IMWRITE_JPEG_QUALITY, 92])
    sys.exit(0)

N = int(DUR * FPS)
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", "%dx%d" % (OW, OH), "-r", str(FPS), "-i", "-",
                         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for f in range(N):
    proc.stdin.write(frame(f / FPS).tobytes())
    if f % 30 == 0:
        print("frame", f, "/", N, flush=True)
proc.stdin.close()
proc.wait()
