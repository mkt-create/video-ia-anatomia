"""Hand-placed anatomical landmarks (pixel coords in the source frames).

before = d0aeb91f video @ ~3.0s (576x1024) ; after = cb3b5a4f video @ ~26.0s (576x768)
Pairs follow the same anatomical point on both photos.
"""
PAIRS = [
    # name                         before        after
    ("waistband L",               (127, 147),   (120, 30)),
    ("waistband top centre",      (300, 122),   (300, 27)),
    ("leg opening L mid",         (220, 240),   (250, 115)),
    ("leg opening R mid",         (350, 235),   (390, 92)),
    ("leg opening L corner",      (120, 207),   (102, 107)),
    ("leg opening R corner",      (500, 212),   (440, 97)),
    ("underwear lowest point",    (281, 345),   (320, 177)),
    ("glute L outer max",         (47, 450),    (77, 280)),
    ("glute R outer max",         (572, 470),   (502, 262)),
    ("gluteal fold L lowest",     (165, 540),   (185, 365)),
    ("gluteal fold R lowest",     (425, 570),   (425, 347)),
    ("cleft lower end",           (280, 530),   (300, 320)),
    ("thigh L outer mid",         (120, 700),   (142, 495)),
    ("thigh R outer mid",         (490, 700),   (470, 495)),
    ("thigh L inner mid",         (275, 700),   (277, 495)),
    ("thigh R inner mid",         (288, 700),   (325, 495)),
    ("knee L outer",              (135, 825),   (152, 630)),
    ("knee L inner",              (275, 825),   (272, 630)),
    ("knee R inner",              (277, 850),   (360, 620)),
    ("knee R outer",              (420, 850),   (475, 610)),
    ("calf L outer",              (132, 1000),  (165, 750)),
    ("calf L inner",              (265, 1000),  (265, 750)),
    ("calf R inner",              (272, 1000),  (385, 750)),
    ("calf R outer",              (410, 980),   (482, 735)),
]

# Real cellulite depressions detected on the before photo (prep.py), curated:
# (x, y, strength, hero)  -- hero = shown in the close-up and treated on screen
TETHERS = [
    (177, 460, 1.00, True),
    (194, 509, 0.90, True),
    (124, 404, 0.80, True),
    (251, 453, 0.60, False),
    (357, 421, 0.85, True),
    (337, 425, 0.50, False),
    (348, 484, 0.90, True),
    (498, 513, 0.50, False),
    (472, 609, 0.80, False),
    (430, 622, 0.50, False),
    (185, 725, 0.50, False),
    (340, 757, 0.40, False),
]

# Surfaces where cellulite relief is emphasised (gluteal convexities and posterior thighs).
# Infragluteal folds, the cleft and the silhouette are left untouched.
ZONES = [
    [(80, 380), (140, 345), (230, 350), (252, 420), (250, 500), (200, 520), (110, 495), (75, 450)],
    [(310, 385), (380, 350), (480, 360), (535, 430), (530, 510), (440, 538), (330, 520), (305, 460)],
    [(115, 575), (250, 560), (255, 760), (150, 770)],
    [(310, 595), (500, 585), (490, 760), (320, 770)],
]
