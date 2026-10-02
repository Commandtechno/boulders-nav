#!/usr/bin/env python3
"""Detect walkable paths in the Scottsdale aerials and build distance fields for registration.

Paths at the resort are either asphalt (dark grey, wide) or decomposed-granite / concrete walkways
(light, smooth, 1.5-4 m wide). The walkways are found with a multi-scale ridge filter (Sato) on a
"light and smooth, not vegetation, not building" image at full resolution (0.124 m/px), tiled.

Outputs data/aerial_paths.npz at 0.248 m/px (the down=2 mosaic grid):
  path   uint8  walkway or asphalt mask
  road   uint8  asphalt mask
  bld    uint8  OSM building footprints
  dpath, droad, dbld  float32 distance (m) to the nearest pixel of each mask, capped at 60 m
  geo    [tx0, ty0, ts, z] mosaic geometry for px<->metres (see tools/aerial.py)
"""
import sys, json, pathlib, time
import numpy as np, cv2
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, Mosaic

t0 = time.time()
M = Mosaic(20, down=1); img = M.img; H, W = img.shape[:2]
print(f"mosaic {W}x{H} at {M.res:.3f} m/px ({time.time() - t0:.0f}s)")

# building footprints (OSM), rasterised at full res then dilated ~2 m so roof edges do not count as paths
osm = json.load(open(root / "data/osm.json"))["elements"]; bld = np.zeros((H, W), np.uint8)
for e in osm:
    t = e.get("tags", {}); g = e.get("geometry")
    if g and "building" in t and len(g) > 3:
        cv2.fillPoly(bld, [M.ll2px([q["lat"] for q in g], [q["lon"] for q in g]).astype(np.int32)], 1)
bld_dil = cv2.dilate(bld, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33)))

# asphalt (whole image at once: cheap)
f = img.astype(np.float32); r, g, b = f[..., 0], f[..., 1], f[..., 2]
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); S, V = hsv[..., 1], hsv[..., 2]
asph = ((S < 40) & (V > 60) & (V < 150) & (np.abs(r - b) < 22) & (np.abs(g - b) < 22)).astype(np.uint8)
asph = cv2.morphologyEx(cv2.morphologyEx(asph, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
n, lab, stats, _ = cv2.connectedComponentsWithStats(asph, 8); keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] * M.res ** 2 > 150
road = keep[lab].astype(np.uint8); del lab
print(f"asphalt {road.mean():.2%} ({time.time() - t0:.0f}s)")

# walkways: tiled ridge detection
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
veg = (g > r - 4) | (b > r + 5)
R = np.zeros((H, W), np.float32); T = 2048; O = 96
for y0 in range(0, H, T):
    for x0 in range(0, W, T):
        ya, yb = max(0, y0 - O), min(H, y0 + T + O); xa, xb = max(0, x0 - O), min(W, x0 + T + O)
        gsub = gray[ya:yb, xa:xb]
        mu = cv2.blur(gsub, (9, 9)); sd = np.sqrt(np.maximum(cv2.blur(gsub * gsub, (9, 9)) - mu * mu, 0))
        bg = cv2.GaussianBlur(gsub, (0, 0), 30); rel = gsub - bg
        inp = np.clip(1 - sd / 18, 0, 1) * np.clip((rel + 10) / 30, 0, 1) * np.clip((gsub - 100) / 60, 0, 1)
        inp[veg[ya:yb, xa:xb]] = 0; inp[bld_dil[ya:yb, xa:xb] > 0] = 0; inp[road[ya:yb, xa:xb] > 0] = 0
        if inp.max() == 0: continue
        rr = sato(inp, sigmas=[3, 4.5, 6.5, 9], black_ridges=False)
        R[y0:min(H, y0 + T), x0:min(W, x0 + T)] = rr[y0 - ya:y0 - ya + min(T, H - y0), x0 - xa:x0 - xa + min(T, W - x0)]
    print(f"  ridge rows {y0}-{min(H, y0 + T)} ({time.time() - t0:.0f}s)")
walk = apply_hysteresis_threshold(R, 0.08, 0.18); walk = remove_small_objects(walk, 400)
n, lab, stats, _ = cv2.connectedComponentsWithStats(walk.astype(np.uint8), 8); keep = np.zeros(n, bool); keep[1:] = np.maximum(stats[1:, 2], stats[1:, 3]) > 120
walk = keep[lab].astype(np.uint8); del lab, R
print(f"walkways {walk.mean():.2%} ({time.time() - t0:.0f}s)")

# downsample masks to 0.248 m/px (any full-res pixel set -> set) and build distance fields
def down2(m): return (cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0.25).astype(np.uint8)
path2, road2, bld2 = down2(walk | road), down2(road), down2(bld)
res2 = M.res * 2
def dist(m): return np.minimum(cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) * res2, 60.0).astype(np.float32)
np.savez_compressed(root / "data/aerial_paths.npz", path=path2, road=road2, bld=bld2, dpath=dist(path2), droad=dist(road2), dbld=dist(bld2),
                    geo=np.array([M.tx0, M.ty0, M.ts // 2, M.z], np.int64), res=res2)
print(f"wrote data/aerial_paths.npz ({time.time() - t0:.0f}s)")

# visual check
from PIL import Image
small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
o = small.copy(); o[path2 > 0] = (o[path2 > 0] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8); o[bld2 > 0] = (o[bld2 > 0] * 0.6 + np.array([255, 255, 0]) * 0.4).astype(np.uint8)
def crop(name, x0, x1, y0, y1):
    c = M.m2px(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)) / 2; a = c.min(0).astype(int); bb = c.max(0).astype(int)
    Image.fromarray(o[a[1]:bb[1], a[0]:bb[0]]).save(f"/tmp/paths_{name}.png")
crop("north", -560, -130, 150, 420); crop("south", -340, -60, -300, -50); crop("lodge", -480, -180, -130, 130); crop("all", -850, 600, -350, 650)
