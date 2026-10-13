"""Pre-processing of the new case: skin mask, pen-mark cleanup, cellulite depression map
and tether points. Everything is derived from the patient's own photo; nothing is invented.
"""
import cv2
import numpy as np
import sys, os, json

W = sys.argv[1] if len(sys.argv) > 1 else "work"
before = cv2.imread(os.path.join(W, "before.png"))
H, Wd = before.shape[:2]
lab = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
L, A, B = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128

# --- skin mask: chroma-based (background and underwear are neutral) ---
chroma = np.sqrt(A * A + B * B)
skin = ((chroma > 13) & (L > 60)).astype(np.uint8)
skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
n, lbl, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
skin = (lbl == big).astype(np.uint8)
skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
cv2.imwrite(os.path.join(W, "skin_mask.png"), skin * 255)

# interior: away from silhouette, underwear edge, intergluteal cleft and gluteal folds
dist = cv2.distanceTransform(skin, cv2.DIST_L2, 5)
interior = (dist > 9).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:Wd]
# intergluteal cleft + inner-thigh contact line (x ~ 281 all the way down)
cleft = np.exp(-((xx - 281) / 14.0) ** 2)
# infragluteal folds: natural creases, not cellulite -> excluded from the map
folds = np.zeros((H, Wd), np.uint8)
for poly in ([(60, 470), (95, 500), (150, 535), (210, 542), (270, 532)],
             [(292, 540), (350, 562), (425, 572), (500, 556), (565, 515)]):
    cv2.polylines(folds, [np.array(poly, np.int32)], False, 1, 22)
folds = cv2.GaussianBlur(folds.astype(np.float32), (0, 0), 6)
interior *= (1 - cleft) * (1 - np.clip(folds * 1.6, 0, 1))
interior = cv2.GaussianBlur(interior, (0, 0), 4)
np.save(os.path.join(W, "interior.npy"), interior)

# --- pen marks: thin dark strokes inside the skin only ---
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k)
pen = ((blackhat > 8) & (dist > 12) & (cleft < 0.3)).astype(np.uint8)
n, lbl, stats, _ = cv2.connectedComponentsWithStats(pen, 8)
keep = np.zeros_like(pen)
for i in range(1, n):
    x, y, w, h, a = stats[i]
    if a >= 6 and max(w, h) >= 10 and a / float(w * h) < 0.4:
        keep[lbl == i] = 1
keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
cv2.imwrite(os.path.join(W, "pen_mask.png"), keep * 255)
clean = cv2.inpaint(before, keep * 255, 4, cv2.INPAINT_TELEA)
# soft edge between inpainted strokes and original pixels
m = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8)[..., None]
clean = (clean * m + before * (1 - m)).astype(np.uint8)
cv2.imwrite(os.path.join(W, "before_clean.png"), clean)

# --- relief from luminance band-pass of the clean photo ---
Lc = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
mid = cv2.GaussianBlur(Lc, (0, 0), 2.5)
low = cv2.GaussianBlur(Lc, (0, 0), 18.0)
band = (mid - low) * interior          # <0 = local shadow (depression)
dep = cv2.GaussianBlur(np.clip(-band, 0, None), (0, 0), 1.5)
np.save(os.path.join(W, "band.npy"), band)
np.save(os.path.join(W, "dep.npy"), dep)

# --- tether points: strongest real depressions inside the gluteal / thigh surfaces ---
zones = np.zeros((H, Wd), np.uint8)
for poly in ([(80, 380), (140, 345), (230, 350), (252, 420), (250, 500), (200, 520), (110, 495), (75, 450)],
             [(310, 385), (380, 350), (480, 360), (535, 430), (530, 510), (440, 538), (330, 520), (305, 460)],
             [(115, 575), (250, 560), (255, 760), (150, 770)],
             [(310, 595), (500, 585), (490, 760), (320, 770)]):
    cv2.fillPoly(zones, [np.array(poly, np.int32)], 1)
d2 = cv2.GaussianBlur(dep, (0, 0), 3)
pts = []
work = d2 * zones
for _ in range(40):
    y, x = np.unravel_index(np.argmax(work), work.shape)
    v = work[y, x]
    if v < 3.0 or len(pts) >= 22:
        break
    pts.append((int(x), int(y), float(v)))
    cv2.circle(work, (int(x), int(y)), 18, 0, -1)
json.dump(pts, open(os.path.join(W, "tethers.json"), "w"))
vis = cv2.applyColorMap(np.clip(dep * 14, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
for x, y, v in pts:
    cv2.circle(vis, (x, y), 6, (0, 255, 0), 1)
    cv2.putText(vis, "%.0f" % v, (x + 7, y), 0, 0.35, (0, 255, 0), 1)
cv2.imwrite(os.path.join(W, "dep_vis.png"), vis)
ov = clean.copy()
for i, (x, y, v) in enumerate(pts):
    cv2.circle(ov, (x, y), 9, (0, 255, 0), 1)
    cv2.putText(ov, str(i), (x + 9, y + 4), 0, 0.4, (0, 255, 0), 1)
cv2.imwrite(os.path.join(W, "tethers_vis.png"), ov)
print("pen px", int(keep.sum()), "tethers", len(pts))
