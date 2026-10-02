#!/usr/bin/env python3
"""Extract registration features from the cached Scottsdale aerials (verify/tiles):
  - asphalt road centrelines (skeleton of a dark, grey, low-saturation mask)
  - white casita roof blobs (centroid + area)
Writes data/aerial_features.npz with everything in the app's local metre frame.
"""
import sys, pathlib, json
import numpy as np, cv2
from skimage.morphology import skeletonize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import Mosaic, root

M = Mosaic(20, down=2)                       # ~0.25 m/px
img = M.img
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); S, V = hsv[..., 1], hsv[..., 2]
r, g, b = [img[..., i].astype(int) for i in range(3)]
px_m2 = M.res ** 2

# --- asphalt ---
asphalt = ((S < 40) & (V > 60) & (V < 150) & (np.abs(r - b) < 22) & (np.abs(g - b) < 22)).astype(np.uint8)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
m = cv2.morphologyEx(cv2.morphologyEx(asphalt, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k9)
n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] * px_m2 > 90
road = keep[lab]
# distance-to-edge gives half-width; the skeleton of wide blobs (parking lots) is not a road centreline
dist = cv2.distanceTransform(road.astype(np.uint8), cv2.DIST_L2, 5) * M.res
sk = skeletonize(road)
ys, xs = np.nonzero(sk)
halfw = dist[ys, xs]
ok = halfw < 6.0                                  # drop skeleton pixels inside areas wider than ~12 m
skel_m = M.px2m(np.c_[xs[ok] + 0.5, ys[ok] + 0.5])
print(f"asphalt: {road.mean():.3%} of area, skeleton {ok.sum()} px ({ok.sum() * M.res / 1000:.1f} km of centreline)")

# --- white roofs ---
roof = ((V > 205) & (S < 40)).astype(np.uint8)
roof = cv2.morphologyEx(cv2.morphologyEx(roof, cv2.MORPH_CLOSE, k9), cv2.MORPH_OPEN, k9)
n, lab, stats, cent = cv2.connectedComponentsWithStats(roof, 8)
area = stats[:, cv2.CC_STAT_AREA] * px_m2
keep = (area > 40) & (area < 3000); keep[0] = False
roof_m = M.px2m(cent[keep]); roof_area = area[keep]
print(f"roofs: {keep.sum()} blobs, median {np.median(roof_area):.0f} m2")

np.savez_compressed(root / "data/aerial_features.npz", skel_m=skel_m.astype(np.float32), roof_m=roof_m.astype(np.float32), roof_area=roof_area.astype(np.float32),
                    res=M.res)
print("wrote data/aerial_features.npz")
