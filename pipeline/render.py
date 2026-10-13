"""Gluteal cellulite remake - photo -> through-the-skin anatomy -> treatment -> result.

The patient's photographs drive everything:
  * geometry never comes from a 3D model: every layer is rendered in the before
    photo's own pixel space, shaded with the photo's own low-frequency light;
  * the depressions that are enhanced, tethered and released are the real ones
    detected on the photo (prep.py / landmarks.TETHERS);
  * the before -> after transition is a landmark thin-plate-spline morph between
    the two real photos (no generated body).

Usage: python3 render.py <workdir> <out.mp4> [--preview N]
"""
import os
import sys
import json
import math
import subprocess

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from landmarks import PAIRS, TETHERS, ZONES  # noqa: E402

WD = sys.argv[1]
OUT = sys.argv[2]
PREVIEW = None
if "--preview" in sys.argv:
    PREVIEW = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]

FPS = 30
DUR = 15.0
OW, OH = 1080, 1920
SS = 2                      # master supersampling over the before photo (576x1024 -> 1152x2048)
rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------- helpers


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_io(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def seg(t, a, b):
    return min(max((t - a) / (b - a), 0.0), 1.0)


def smooth_arr(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


# ----------------------------------------------------------------------------- inputs
before1 = cv2.imread(os.path.join(WD, "before_clean.png"))
after1 = cv2.imread(os.path.join(WD, "after.png"))
band1 = np.load(os.path.join(WD, "band.npy")).astype(np.float32)
skin1 = cv2.imread(os.path.join(WD, "skin_mask.png"), 0)
H1, W1 = before1.shape[:2]
MH, MW = H1 * SS, W1 * SS


def up(img, interp=cv2.INTER_LANCZOS4):
    return cv2.resize(img, (MW, MH), interpolation=interp)


before = up(before1).astype(np.float32)
band = cv2.GaussianBlur(up(band1, cv2.INTER_CUBIC), (0, 0), 1.0)
skin = cv2.GaussianBlur(up(skin1, cv2.INTER_LINEAR).astype(np.float32) / 255.0, (0, 0), 2)
skin_in = cv2.GaussianBlur((cv2.distanceTransform((skin > 0.5).astype(np.uint8), cv2.DIST_L2, 5) > 10)
                           .astype(np.float32), (0, 0), 6)

lumB = cv2.cvtColor(before.astype(np.uint8), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
shade_low = cv2.GaussianBlur(lumB, (0, 0), 28)            # the photo's own large-scale light / volume
shade_n = shade_low / (np.percentile(shade_low[skin > 0.5], 70) + 1e-3)
skin_hp = before - cv2.GaussianBlur(before, (0, 0), 2.0)   # pores / fine grain of the real skin

zone = np.zeros((MH, MW), np.uint8)
for poly in ZONES:
    cv2.fillPoly(zone, [np.array(poly, np.int32) * SS], 1)
zone = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 14 * SS)
zone = np.clip(zone * 1.6, 0, 1)
# gluteal convexities only (the see-through window never crosses the infragluteal fold)
zone_g = np.zeros((MH, MW), np.uint8)
for poly in ZONES[:2]:
    cv2.fillPoly(zone_g, [np.array(poly, np.int32) * SS], 1)
zone_g = cv2.dilate(zone_g, np.ones((14 * SS, 14 * SS), np.uint8))
zone_g = cv2.GaussianBlur(zone_g.astype(np.float32), (0, 0), 9 * SS)

TETH = [(x * SS, y * SS, s, hero) for (x, y, s, hero) in TETHERS]
HERO = [t for t in TETH if t[3]]
# treatment order: left glute (lateral -> medial), then right glute
ORDER = [2, 0, 1, 4, 6]          # indices in TETH (hero ones)
ORDER = [i for i in ORDER if TETH[i][3]]

# ----------------------------------------------------------------------------- after photo placement
bpts = np.float32([p[1] for p in PAIRS])
apts = np.float32([p[2] for p in PAIRS])
wts = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, .5, .5, .5, .5, .3, .3, .3, .3])
X, Y = [], []
for (ax, ay), (bx, by), wi in zip(apts, bpts, wts):
    X += [[ax * wi, wi, 0], [ay * wi, 0, wi]]
    Y += [bx * wi, by * wi]
sc, tx, ty = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0]
A_M = np.float32([[sc * SS, 0, tx * SS], [0, sc * SS, ty * SS]])
after = cv2.warpAffine(after1, A_M, (MW, MH), flags=cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_REFLECT).astype(np.float32)
B_pts = bpts * SS
A_pts = apts * sc * SS + np.float32([tx * SS, ty * SS])
# the morph is driven by trunk / gluteal / thigh landmarks only: the legs stand wider apart
# in the after photo, and warping knees/calves across that gap makes them look like they melt.
# Below mid-thigh the two photos simply dissolve.
_mi = [k for k, p in enumerate(PAIRS) if p[1][1] < 760]
B_m, A_m = B_pts[_mi], A_pts[_mi]

# gentle global white-balance harmonisation of the after photo (lighting differs between
# sessions). Mean shift only, 55 %, measured on skin. No local / shape change.
labA = cv2.cvtColor(after.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
labB = cv2.cvtColor(before.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
chA = np.hypot(labA[..., 1] - 128, labA[..., 2] - 128)
mA = (chA > 12) & (labA[..., 0] > 60)
mB = skin > 0.5
shift = (labB[mB].mean(0) - labA[mA].mean(0)) * 0.55
shift[0] *= 0.6
labA += shift
after = cv2.cvtColor(np.clip(labA, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

# ----------------------------------------------------------------------------- thin plate spline


def tps_fit(src, dst, reg=1e-3):
    n = len(src)
    d = np.linalg.norm(src[:, None] - src[None], axis=2)
    K = np.where(d > 0, d * d * np.log(d + 1e-9), 0) + reg * np.eye(n)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    Yv = np.vstack([dst, np.zeros((3, 2))])
    return src, np.linalg.solve(L, Yv)


def tps_eval(model, pts):
    src, coef = model
    d = np.linalg.norm(pts[:, None] - src[None], axis=2)
    U = np.where(d > 0, d * d * np.log(d + 1e-9), 0)
    n = len(src)
    return U @ coef[:n] + coef[n] + pts @ coef[n + 1:]


edge_anchor = np.float32([[x, y] for x in np.linspace(0, MW, 5) for y in np.linspace(0, MH, 7)
                          if x in (0, MW) or y in (0, MH)])
GS = 8
gy, gx = np.mgrid[0:MH:GS, 0:MW:GS].astype(np.float32)
grid = np.stack([gx.ravel(), gy.ravel()], 1)


def warp_to(img, dst_pts, src_pts):
    """Backward-warp img so that src_pts land on dst_pts."""
    m = tps_fit(np.vstack([dst_pts, edge_anchor]), np.vstack([src_pts, edge_anchor]), reg=120.0)
    mp = tps_eval(m, grid).reshape(gx.shape + (2,)).astype(np.float32)
    mx = cv2.resize(mp[..., 0], (MW, MH), interpolation=cv2.INTER_CUBIC)
    my = cv2.resize(mp[..., 1], (MW, MH), interpolation=cv2.INTER_CUBIC)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


fwd = tps_fit(np.vstack([B_pts, edge_anchor]), np.vstack([A_pts, edge_anchor]), reg=40.0)
TETH_AFTER = tps_eval(fwd, np.float32([[t[0], t[1]] for t in TETH]))

# ----------------------------------------------------------------------------- anatomy assets
yy, xx = np.mgrid[0:MH, 0:MW].astype(np.float32)

# region of interest for the anatomical layer (glutes + upper thighs)
RX0, RY0, RX1, RY1 = 40 * SS, 300 * SS, 560 * SS, 800 * SS
ry, rx = yy[RY0:RY1, RX0:RX1], xx[RY0:RY1, RX0:RX1]
_bd = np.minimum.reduce([rx - RX0, RX1 - 1 - rx, ry - RY0, RY1 - 1 - ry])
roi_fade = smooth_arr(_bd / (40.0 * SS))

# lobule seeds: jittered grid (~11 px at photo scale ~ 8-10 mm lobules)
SP = 15 * SS
sy, sx = np.mgrid[RY0 - SP:RY1 + SP:SP, RX0 - SP:RX1 + SP:SP].astype(np.float32)
sx = sx + (np.arange(sx.shape[0])[:, None] % 2) * SP * 0.5
seeds0 = np.stack([sx.ravel(), sy.ravel()], 1) + rng.uniform(-0.38, 0.38, (sx.size, 2)) * SP
NS = len(seeds0)
seed_tone = rng.uniform(0, 1, NS).astype(np.float32)
# micro (adipocyte) texture: bubble-like cells (~6 mm clusters), static
mseed = np.full((RY1 - RY0, RX1 - RX0), 255, np.uint8)
NM = 17000
pts_m = np.column_stack([rng.integers(0, RX1 - RX0, NM), rng.integers(0, RY1 - RY0, NM)])
mseed[pts_m[:, 1], pts_m[:, 0]] = 0
mdist, mlab = cv2.distanceTransformWithLabels(mseed, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
medge = np.zeros_like(mdist)
medge[:, 1:] += (mlab[:, 1:] != mlab[:, :-1])
medge[1:, :] += (mlab[1:, :] != mlab[:-1, :])
mrad = cv2.dilate(mdist, np.ones((9, 9), np.uint8))          # ~ local cell radius
micro = np.clip(1 - (mdist / (mrad + 1.5)) ** 2, 0, 1) ** 0.6   # one dome per adipocyte
micro = cv2.GaussianBlur(micro, (0, 0), 0.8)
mcrev = cv2.GaussianBlur(np.clip(medge, 0, 1).astype(np.float32), (0, 0), 0.9)

# vessel-free, low-frequency organic noise used for the reveal front
noise_lo = cv2.GaussianBlur(rng.normal(0, 1, (MH // 16, MW // 16)).astype(np.float32), (0, 0), 1.5)
noise_lo = cv2.resize(noise_lo, (MW, MH), interpolation=cv2.INTER_CUBIC)
noise_lo /= noise_lo.std() + 1e-6

# fine collagen fibrils for the bio-stimulation beat (very discreet)
FIBRILS = []
for (tx_, ty_, s_, hero) in TETH:
    if not hero:
        continue
    for _ in range(26):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(4, 30) * SS
        cx, cy = tx_ + r * math.cos(a), ty_ + r * math.sin(a)
        ang = rng.uniform(0, math.pi)
        L_ = rng.uniform(6, 16) * SS
        bend = rng.uniform(-0.5, 0.5)
        ts = np.linspace(-0.5, 0.5, 9)
        px = cx + ts * L_ * math.cos(ang) + bend * (ts ** 2 - 0.25) * L_ * -math.sin(ang)
        py = cy + ts * L_ * math.sin(ang) + bend * (ts ** 2 - 0.25) * L_ * math.cos(ang)
        FIBRILS.append((np.stack([px, py], 1), rng.uniform(0, 1)))


# cannula choreography: enter laterally, fan under each tether (subcision), glide to
# the next one without leaving the tissue, exit; then the contralateral side.
ENTRY_L = np.float32([112, 548]) * SS
ENTRY_R = np.float32([452, 552]) * SS
D_ENTER, D_FAN, D_GLIDE, D_EXIT = 0.25, 0.42, 0.18, 0.18


def _build_schedule(t0):
    ev, t = [], t0
    for entry, ids in ((ENTRY_L, [2, 0, 1]), (ENTRY_R, [4, 6])):
        pts = [np.float32(TETH[i][:2]) for i in ids]
        ev.append(("move", t, t + D_ENTER, entry, pts[0], entry, None)); t += D_ENTER
        for k, i in enumerate(ids):
            ev.append(("fan", t, t + D_FAN, pts[k], pts[k], entry, i)); t += D_FAN
            if k + 1 < len(ids):
                ev.append(("move", t, t + D_GLIDE, pts[k], pts[k + 1], entry, None)); t += D_GLIDE
        ev.append(("move", t, t + D_EXIT, pts[-1], entry, entry, None)); t += D_EXIT
    return ev, t


SCHED, _t_end = _build_schedule(6.8)
PROC = (6.8, _t_end)


def release_state(t):
    """per-tether release 0..1, cannula tip / entry and visibility."""
    rel = np.zeros(len(TETH), np.float32)
    tip, entry, inside = None, None, 0.0
    for kind, a, b, p0, p1, ent, i in SCHED:
        if kind == "fan":
            rel[i] = smooth(seg(t, a + 0.1 * (b - a), b))
        if a <= t <= b:
            u = (t - a) / (b - a)
            entry = ent
            if kind == "move":
                tip = p0 + (p1 - p0) * ease_io(u)
            else:
                d = p0 - ent
                dn = d / (np.linalg.norm(d) + 1e-6)
                perp = np.float32([-dn[1], dn[0]])
                # short fan sweeps right under the depression
                tip = p0 + perp * math.sin(u * 2 * math.pi * 2) * 7 * SS * math.sin(math.pi * u) \
                    + dn * 3 * SS * math.sin(math.pi * u)
            inside = 1.0
    # non-hero tethers soften with their neighbours (same session, same tissue)
    for j, tj in enumerate(TETH):
        if not tj[3]:
            rel[j] = 0.45 * smooth(seg(t, PROC[0] + 0.5 * (PROC[1] - PROC[0]), PROC[1] + 0.6))
    return rel, tip, entry, inside


# radial fibrous strands (septa diving into each depression), static shape per tether
STRANDS = []
for (tx_, ty_, st, hero) in TETH:
    ss = []
    nk = 6
    for k in range(nk):
        ang = 2 * math.pi * k / nk + rng.uniform(-0.45, 0.45)
        ss.append((ang, rng.uniform(13, 23) * SS * (0.7 + 0.4 * st), rng.uniform(-0.8, 0.8)))
    STRANDS.append(ss)


def strands_layer(rel, pull, shape):
    lay = np.zeros(shape + (3,), np.float32)
    msk = np.zeros(shape, np.float32)
    for i, (tx_, ty_, st, hero) in enumerate(TETH):
        if pull <= 0:
            break
        r = rel[i]
        for ang, L_, bend in STRANDS[i]:
            # on release the strand lets go of the depression and retracts / relaxes
            r_in = 2 * SS + r * 0.65 * L_
            ts = np.linspace(0, 1, 10)
            rr = r_in + (L_ - r_in) * ts
            aa = ang + bend * (1 + 1.5 * r) * (1 - ts) * ts * 2
            px = tx_ + rr * np.cos(aa) - RX0
            py = ty_ + rr * np.sin(aa) - RY0
            q = np.round(np.stack([px, py], 1) * 16).astype(np.int32)
            for j in range(9):
                th = 2 if ts[j] < 0.6 and r < 0.5 else 1
                # fibres fade toward the tip (they dive toward the fascia) and toward the outer end
                w_ = (0.55 + 0.45 * math.sin(math.pi * min(ts[j] * 1.3, 1.0))) * 1.05
                cv2.line(lay, tuple(q[j]), tuple(q[j + 1]), (198, 222, 240), th, cv2.LINE_AA, 4)
                cv2.line(msk, tuple(q[j]), tuple(q[j + 1]), float(st * pull * (1 - 0.7 * r) * w_), th, cv2.LINE_AA, 4)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.3)
    msk = cv2.GaussianBlur(msk, (0, 0), 1.4)
    return lay, msk


def anatomy(rel, pull_gain, glow):
    """Render the subcutaneous layer as seen through the skin, in master space ROI."""
    # tethered septa pull the lobule mosaic toward each depression point
    s = seeds0.copy()
    for i, (tx_, ty_, st, hero) in enumerate(TETH):
        d = np.float32([tx_, ty_]) - s
        r2 = (d ** 2).sum(1)
        sig = 22 * SS * (0.8 + 0.4 * st)
        c = 0.40 * st * pull_gain * (1 - rel[i])
        s += d * (c * np.exp(-r2 / (2 * sig * sig)))[:, None]
    img = np.full((RY1 - RY0, RX1 - RX0), 255, np.uint8)
    loc = np.round(s - [RX0, RY0]).astype(int)
    ok = (loc[:, 0] >= 0) & (loc[:, 0] < RX1 - RX0) & (loc[:, 1] >= 0) & (loc[:, 1] < RY1 - RY0)
    img[loc[ok, 1], loc[ok, 0]] = 0
    dist, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # labels index the zero pixels in raster order -> map back to each seed's tone
    zy, zx = np.nonzero(img == 0)
    lut = np.full(len(zy) + 1, 0.5, np.float32)
    tone_img = np.zeros(img.shape, np.float32)
    tone_img[loc[ok, 1], loc[ok, 0]] = seed_tone[ok]
    lut[1:] = tone_img[zy, zx]
    tone = lut[lab]
    # septa = boundaries between lobules
    edge = np.zeros_like(dist)
    edge[:, 1:] += (lab[:, 1:] != lab[:, :-1])
    edge[1:, :] += (lab[1:, :] != lab[:-1, :])
    edge = cv2.GaussianBlur(np.clip(edge, 0, 1).astype(np.float32), (0, 0), 1.2)
    # lobule = cluster of adipocyte domes, lit from upper-left like the room light
    cell_r = SP * 0.70
    h = np.clip(1 - (dist / cell_r) ** 2.4, 0, 1)
    h = cv2.GaussianBlur(h, (0, 0), 2.5)
    hm = 0.5 * h + 0.5 * micro * (0.35 + 0.65 * h)
    gy_, gx_ = np.gradient(cv2.GaussianBlur(hm, (0, 0), 0.7))
    nx, ny = -gx_ * 6.0, -gy_ * 6.0
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1)
    ndl = np.clip((nx * -0.45 + ny * -0.6 + 0.66) * inv, 0, 1)
    spec = np.clip((nx * -0.25 + ny * -0.35 + 0.9) * inv, 0, 1) ** 28
    c_lo = np.float32([30, 112, 182])     # BGR deep amber (crevices / depth)
    c_hi = np.float32([105, 205, 248])    # BGR golden butter
    k = np.clip(0.1 + 0.85 * hm + 0.12 * (tone - 0.5), 0, 1)[..., None]
    fat = (c_lo * (1 - k) + c_hi * k) * (0.55 + 0.6 * ndl)[..., None]
    fat *= (1 - 0.35 * mcrev * h)[..., None]
    # small, soft wet highlights (moist tissue, not plastic gloss)
    fat += np.float32([225, 240, 250]) * (0.35 * spec * h)[..., None]
    # septa: pale translucent fibrous tissue, thicker/brighter where under traction
    tens = np.zeros_like(dist)
    for i, (tx_, ty_, st, hero) in enumerate(TETH):
        r2 = (rx - tx_) ** 2 + (ry - ty_) ** 2
        tens += st * (1 - rel[i]) * np.exp(-r2 / (2 * (18 * SS) ** 2))
    tens = np.clip(tens, 0, 1.2)
    sept_col = np.float32([186, 214, 236])
    sept_a = np.clip(edge * (0.55 + 0.8 * tens), 0, 0.95)[..., None]
    out = fat * (1 - sept_a) + sept_col * sept_a
    # deep anchor of each tether: the septum dives toward the fascia
    core = np.zeros_like(dist)
    gl = np.zeros_like(dist)
    for i, (tx_, ty_, st, hero) in enumerate(TETH):
        r2 = (rx - tx_) ** 2 + (ry - ty_) ** 2
        core += st * (1 - rel[i]) * np.exp(-r2 / (2 * (5.0 * SS) ** 2))
        if hero:
            gl += st * np.exp(-r2 / (2 * (15 * SS) ** 2))
    core = np.clip(core, 0, 1)[..., None]
    out = out * (1 - 0.3 * core) + np.float32([150, 175, 205]) * 0.25 * core
    # localised soft warm light on the traction points
    out += np.float32([150, 205, 255])[None, None] * (0.28 * glow * np.clip(gl, 0, 1))[..., None]
    return out, tens


def draw_cannula(canvas, entry, tip, alpha):
    """Thin steel cannula seen through tissue (drawn in master space)."""
    if tip is None or np.max(alpha) <= 0:
        return canvas
    lay = np.zeros_like(canvas)
    m = np.zeros(canvas.shape[:2], np.float32)
    e = tuple(int(v * 16) for v in entry)
    p = tuple(int(v * 16) for v in tip)
    cv2.line(lay, e, p, (150, 155, 160), 7, cv2.LINE_AA, 4)
    cv2.line(lay, e, p, (225, 230, 235), 2, cv2.LINE_AA, 4)
    cv2.line(m, e, p, 1.0, 7, cv2.LINE_AA, 4)
    cv2.circle(m, p, 3 * 16, 1.0, -1, cv2.LINE_AA, 4)
    cv2.circle(lay, p, 3 * 16, (175, 180, 186), -1, cv2.LINE_AA, 4)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.8)
    m = (cv2.GaussianBlur(m, (0, 0), 0.9) * alpha)[..., None]
    return canvas * (1 - m) + lay * m


# ----------------------------------------------------------------------------- timeline
# beats mirror the reference cut (15 s): photo -> scan -> close-up -> reveal -> treatment ->
# back to skin -> result
T_SCAN = (1.0, 2.7)
T_ZOOM = (2.4, 4.3)
T_REVEAL = (4.3, 6.5)
T_BACK = (PROC[1], PROC[1] + 1.1)
T_MORPH = (T_BACK[1], T_BACK[1] + 1.5)

CAM = [  # time, centre x, centre y, width   (photo pixel units)
    (0.0, 288, 512, 576),
    (1.0, 288, 515, 556),
    (2.4, 285, 520, 532),
    (4.3, 250, 462, 336),
    (6.8, 252, 464, 318),
    (10.3, 256, 468, 308),
    (11.5, 283, 580, 500),
    (13.0, 283, 580, 488),
    (15.0, 283, 576, 470),
]


def camera(t):
    for (t0, x0, y0, w0), (t1, x1, y1, w1) in zip(CAM[:-1], CAM[1:]):
        if t0 <= t <= t1:
            k = ease_io((t - t0) / (t1 - t0))
            # zoom in log space so speed feels constant
            return lerp(x0, x1, k), lerp(y0, y1, k), math.exp(lerp(math.log(w0), math.log(w1), k))
    return CAM[-1][1:]


def to_out(cx, cy, w, pts):
    h = w * OH / OW
    s = OW / w
    return (np.asarray(pts, np.float32) / SS - [cx - w / 2, cy - h / 2]) * s


def render_camera(master, cx, cy, w):
    h = w * OH / OW
    s = OW / (w * SS)
    M = np.float32([[s, 0, -(cx - w / 2) * SS * s], [0, s, -(cy - h / 2) * SS * s]])
    flag = cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA
    return cv2.warpAffine(master, M, (OW, OH), flags=flag, borderMode=cv2.BORDER_REFLECT)


# silhouette hairlines (from the skin mask of the real photo)
SIL_L, SIL_R = [], []
for y in range(215, 1000, 4):
    xs = np.nonzero(skin1[y] > 0)[0]
    if len(xs) > 20:
        SIL_L.append((xs[0], y))
        SIL_R.append((xs[-1], y))


def smooth_poly(p):
    p = np.float32(p)
    k = np.ones(9) / 9
    return np.stack([np.convolve(np.pad(p[:, i], 4, mode="edge"), k, "valid") for i in range(2)], 1)


SIL_L, SIL_R = smooth_poly(SIL_L) * SS, smooth_poly(SIL_R) * SS


def hairline(img, pts, prog, alpha, col=(235, 240, 245), th=2):
    if prog <= 0 or alpha <= 0:
        return img
    n = max(2, int(len(pts) * prog))
    lay = img.copy()
    p = np.round(pts[:n] * 16).astype(np.int32)
    cv2.polylines(lay, [p], False, col, th, cv2.LINE_AA, 4)
    if prog < 1:  # travelling light at the head of the line
        hx, hy = pts[n - 1]
        cv2.circle(lay, (int(hx * 16), int(hy * 16)), 5 * 16, (245, 248, 250), -1, cv2.LINE_AA, 4)
    return cv2.addWeighted(lay, alpha, img, 1 - alpha, 0)


def ring(img, c, r, prog, alpha, th=2):
    if prog <= 0 or alpha <= 0:
        return img
    lay = img.copy()
    cc = (int(c[0] * 16), int(c[1] * 16))
    cv2.ellipse(lay, cc, (int(r * 16), int(r * 16)), -90, 0, 360 * prog, (238, 242, 246), th, cv2.LINE_AA, 4)
    return cv2.addWeighted(lay, alpha, img, 1 - alpha, 0)


# grade: soft filmic contrast, gentle warm-neutral, vignette, grain (same on every beat)
vig = None


def grade(img, t):
    global vig
    x = img.astype(np.float32) / 255.0
    x = np.clip(x, 0, 1)
    x = np.clip((x - 0.45) * 1.07 + 0.45, 0, 1)                      # gentle contrast
    x = 0.012 + x * 0.985                                            # lifted, soft blacks
    x[..., 0] *= 0.99
    x[..., 2] *= 1.008
    if vig is None:
        Y, X = np.mgrid[0:OH, 0:OW].astype(np.float32)
        r = np.sqrt(((X - OW / 2) / (OW * 0.75)) ** 2 + ((Y - OH / 2) / (OH * 0.72)) ** 2)
        vig = (1 - 0.22 * np.clip(r, 0, 1) ** 2.4)[..., None]
    x *= vig
    # unsharp (counteracts the up-sampling softness without halos)
    bl = cv2.GaussianBlur(x, (0, 0), 1.6)
    x = x + 0.35 * (x - bl)
    g = np.random.default_rng(int(t * 1000)).normal(0, 0.010, (OH // 2, OW // 2)).astype(np.float32)
    g = cv2.resize(g, (OW, OH), interpolation=cv2.INTER_LINEAR)[..., None]
    x = x + g * (0.6 + 0.4 * x)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- per-frame
def frame(t):
    rel, tip, entry, inside = release_state(t)

    # 1) relief: amplify the patient's own depressions (contact shadows + rim light)
    enh = 0.95 * smooth(seg(t, 1.2, 4.0))
    relm = np.zeros((MH, MW), np.float32)
    for i, (tx_, ty_, st, hero) in enumerate(TETH):
        relm = np.maximum(relm, rel[i] * np.exp(-((xx - tx_) ** 2 + (yy - ty_) ** 2) / (2 * (22 * SS) ** 2)))
    glob_soft = 0.55 * smooth(seg(t, T_BACK[0], T_MORPH[0] + 0.3))
    neg = -9.0 * np.tanh(-np.minimum(band, 0) / 9.0)   # soft-capped: no craters
    pos = np.maximum(band, 0)
    k_enh = enh * (1 - np.clip(relm + glob_soft, 0, 1))
    fill = np.clip(relm * 0.85 + glob_soft, 0, 0.9)
    dL = zone * (k_enh * (0.9 * neg + 0.35 * pos)) - fill * np.minimum(band, 0) - fill * 0.4 * pos
    base = before
    Lc = np.clip(lumB, 1, None)
    gain = np.clip((Lc + dL * 1.1) / Lc, 0.6, 1.4)[..., None]
    skin_img = base * gain
    # contact shadows a touch warmer/redder like real skin folds
    sh = np.clip(-dL / 20.0, 0, 1)[..., None]
    skin_img = skin_img * (1 - 0.10 * sh * np.float32([1.0, 0.6, 0.0]))

    master = skin_img

    # 2) through-the-skin reveal
    rv = smooth(seg(t, T_REVEAL[0], T_REVEAL[1])) * (1 - smooth(seg(t, T_BACK[0], T_BACK[1])))
    if rv > 0:
        prog = smooth(seg(t, T_REVEAL[0], T_REVEAL[1] + 0.3))
        back = smooth(seg(t, T_BACK[0], T_BACK[1]))
        sl = (slice(RY0, RY1), slice(RX0, RX1))
        # organic window growing out of the real depressions
        dmin = np.full(rx.shape, 1e9, np.float32)
        for (tx_, ty_, st, hero) in HERO:
            dmin = np.minimum(dmin, np.sqrt((rx - tx_) ** 2 + (ry - ty_) ** 2))
        R = (6 + 70 * prog) * SS * (1 - 0.5 * back)
        front = (R - dmin + noise_lo[sl] * 7 * SS) / (48 * SS)
        win = smooth_arr(front) * skin_in[sl] * roi_fade * zone_g[sl]
        pull = smooth(seg(t, T_REVEAL[0] + 0.5, T_REVEAL[1] + 0.3))
        glow = smooth(seg(t, T_REVEAL[1] - 0.3, T_REVEAL[1] + 0.5)) * (1 - 0.6 * smooth(seg(t, PROC[0], PROC[1])))
        anat, tens = anatomy(rel, pull, glow)
        slay, smsk = strands_layer(rel, pull, anat.shape[:2])
        anat = anat * (1 - smsk[..., None] * 0.85) + slay * (smsk[..., None] * 0.85)
        roi = master[sl]
        # lit by the photo's own light, slightly desaturated: seen through living skin
        anat = anat * np.clip(shade_n[sl], 0.6, 1.2)[..., None]
        lum = anat.mean(2, keepdims=True)
        anat = lum + (anat - lum) * 0.8
        tint = roi / (cv2.GaussianBlur(roi, (0, 0), 30) + 1)
        # the real surface relief (incl. the depressions) stays readable on top
        anat = anat * (0.7 + 0.3 * tint) * gain[sl] ** 0.6
        film = 0.26 + 0.3 * (1 - rv)
        anat = anat * (1 - film) + roi * film + skin_hp[sl] * 0.5
        # stage 1: the skin first turns luminous / translucent, structures resolve after
        s2 = smooth(seg(t, T_REVEAL[0] + 0.5, T_REVEAL[1]))
        early = roi * np.float32([1.02, 1.07, 1.12]) + 6
        anat = early * (1 - s2) + anat * s2
        a2 = win * rv * 0.82
        a = a2[..., None]
        # subsurface: a warm translucent rim where skin turns to tissue
        rim = (np.clip(a2 * 4, 0, 1) * (1 - np.clip(a2 * 1.4, 0, 1)))[..., None]
        roi_w = roi * (1 + 0.10 * rim * np.float32([-0.4, 0.0, 0.5]))
        roi_new = roi_w * (1 - a) + anat * a
        if tip is not None:
            roi_new = draw_cannula(roi_new, entry - [RX0, RY0], tip - [RX0, RY0],
                                   inside * np.clip(a2 * 1.25, 0, 0.95))
        bio = smooth(seg(t, PROC[0] + 0.8, PROC[1] + 0.4))
        if bio > 0:
            lay = np.zeros_like(roi_new)
            for pts, ph in FIBRILS:
                q = np.round((pts - [RX0, RY0]) * 16).astype(np.int32)
                cv2.polylines(lay, [q], False, (210, 226, 240), 1, cv2.LINE_AA, 4)
            lay = cv2.GaussianBlur(lay, (0, 0), 0.6)
            roi_new = roi_new + lay * 0.16 * bio * a
        master = master.copy()
        master[sl] = roi_new

    # 3) morph to the result photo
    mk = ease_io(seg(t, *T_MORPH))
    if mk > 0:
        P = B_m * (1 - mk) + A_m * mk
        wb = warp_to(master, P, B_m) if mk < 1 else None
        wa = warp_to(after, P, A_m) if mk > 0 else None
        if mk >= 1:
            master = after
        else:
            bk = smooth((mk - 0.25) / 0.5)
            master = wb * (1 - bk) + wa * bk

    cx, cy, w = camera(t)
    out = render_camera(np.clip(master, 0, 255).astype(np.float32), cx, cy, w)

    # 4) hairline overlays in output space (thin, editorial)
    a_sil = smooth(seg(t, T_SCAN[0], T_SCAN[0] + 0.2)) * (1 - smooth(seg(t, T_ZOOM[0] + 0.3, T_ZOOM[0] + 1.0)))
    if a_sil > 0:
        p = smooth(seg(t, T_SCAN[0], T_SCAN[1] - 0.2))
        out = hairline(out, to_out(cx, cy, w, SIL_L), p, 0.75 * a_sil)
        out = hairline(out, to_out(cx, cy, w, SIL_R), p, 0.75 * a_sil)
    a_ring = smooth(seg(t, 1.7, 2.1)) * (1 - smooth(seg(t, T_REVEAL[0] + 0.6, T_REVEAL[0] + 1.4)))
    if a_ring > 0:
        for k, i in enumerate(ORDER):
            tx_, ty_ = TETH[i][0], TETH[i][1]
            c = to_out(cx, cy, w, [[tx_, ty_]])[0]
            pr = smooth(seg(t, 1.8 + 0.18 * k, 2.4 + 0.18 * k))
            r = 15 * SS * OW / (w * SS)
            out = ring(out, c, r, pr, 0.8 * a_ring)
    a_end = smooth(seg(t, 13.4, 13.9))
    if a_end > 0:
        for k, i in enumerate(ORDER):
            c = to_out(cx, cy, w, [TETH_AFTER[i]])[0]
            pr = smooth(seg(t, 13.4 + 0.12 * k, 14.0 + 0.12 * k))
            r = 15 * OW / w
            out = ring(out, c, r, pr, 0.7 * a_end)

    return grade(out, t)


# ----------------------------------------------------------------------------- main
if PREVIEW:
    for t in PREVIEW:
        cv2.imwrite(OUT.replace(".mp4", "_%05.2f.jpg" % t), frame(t), [cv2.IMWRITE_JPEG_QUALITY, 92])
    sys.exit(0)

N = int(DUR * FPS)
proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                         "-s", "%dx%d" % (OW, OH), "-r", str(FPS), "-i", "-",
                         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                         "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
for f in range(N):
    t = f / FPS
    proc.stdin.write(frame(t).tobytes())
    if f % 30 == 0:
        print("frame", f, "/", N, flush=True)
proc.stdin.close()
proc.wait()
