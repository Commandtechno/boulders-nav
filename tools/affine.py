#!/usr/bin/env python3
"""Georeference the PDF map with ONE global affine transform (no warp): the drawing is shown exactly as drawn,
just rotated, scaled and placed. Fitted by least squares to fixed landmarks (pools, lakes, lodge, clubhouse,
gate, road junctions). Writes data/appdata.json with linear fwd/inv grids so the app needs no changes.

  python3 tools/affine.py [--write] [--similarity]
"""
import sys, json, pathlib, argparse
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--similarity", action="store_true", help="rotation + uniform scale only"); ap.add_argument("--out", default="data/appdata.affine.json"); args = ap.parse_args()
D = json.load(open(root / "data/appdata.v1.json")); rooms = D["rooms"]; landmarks = D["landmarks"]

anchors = [  # PDF point (pt), real position (local metres), name
    ((96, 74), (-705.7, 415.5), "gate / Tom Darlington"), ((157, 117), (-587.5, 305.3), "spa pool"), ((273, 447), (-348.6, 25.9), "lodge"),
    ((240, 426), (-390.6, 40.9), "lodge pool"), ((370, 450), (-266.2, 12.4), "duck pond"), ((728, 580), (85.7, -164.0), "villa pool"),
    ((831, 303), (223.7, 213.9), "clubhouse pool"), ((854, 297), (259.2, 235.9), "clubhouse"), ((795, 330), (165.0, 180.0), "tennis"),
    ((875, 365), (317.6, 111.8), "club lake"), ((105, 711), (-608.1, -162.8), "El Pedregal"), ((640, 355), (24.6, 137.4), "Pkwy x Clubhouse Rd"),
    ((692, 547), (26.6, -146.2), "Pkwy x Whitethorn"), ((688, 585), (0.0, -191.2), "Pkwy x Sunflower"), ((680, 615), (-5.0, -221.5), "Pkwy x Clubhouse Ct"),
    ((555, 360), (-73.9, 97.9), "E Clubhouse Rd x Clubhouse Rd"), ((458, 368), (-191.8, 125.2), "Haciendas loop x Clubhouse Rd"),
    ((249, 297), (-369.7, 172.8), "#5 green"), ((470, 563), (-169.2, -114.7), "#6 green"), ((203, 632), (-394.2, -139.4), "#7 green"),
]
P = np.array([a[0] for a in anchors], float); Q = np.array([a[1] for a in anchors], float)
if args.similarity:
    # Umeyama with reflection allowed (PDF y points down)
    mp, mq = P.mean(0), Q.mean(0); H = (P - mp).T @ (Q - mq) / len(P); U, S, Vt = np.linalg.svd(H)
    Sg = np.eye(2); Sg[1, 1] = np.sign(np.linalg.det(U) * np.linalg.det(Vt)) if False else 1
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) > 0: Vt2 = Vt.copy(); Vt2[-1] *= -1; Rm = Vt2.T @ U.T; S = S * np.array([1, -1])   # reflection needed
    s = S.sum() / ((P - mp) ** 2).sum() * len(P); A = s * Rm; t = mq - A @ mp
    f = lambda X: X @ A.T + t
else:
    B = np.hstack([P, np.ones((len(P), 1))]); Th = np.linalg.lstsq(B, Q, rcond=None)[0]
    f = lambda X: np.hstack([X, np.ones((len(X), 1))]) @ Th
res = np.linalg.norm(f(P) - Q, axis=1)
print(("similarity" if args.similarity else "affine") + f" fit: residual median {np.median(res):.1f} m, max {res.max():.1f} m")
for a, r in sorted(zip(anchors, res), key=lambda x: -x[1])[:8]: print(f"  {a[2]:32s} {r:5.1f} m")
J = (f(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])) - f(np.array([[0.0, 0.0]] * 3)))[:2]
print(f"scale: {np.linalg.norm(J[0]):.3f} m/pt along x, {np.linalg.norm(J[1]):.3f} m/pt along y; rotation of x axis {np.degrees(np.arctan2(J[0][1], J[0][0])):.1f} deg")

# score against OSM building footprints
osm = json.load(open(root / "data/osm.json"))["elements"]; polys = []
for e in osm:
    tg = e.get("tags", {}); g = e.get("geometry")
    if g and "building" in tg and len(g) > 3: polys.append(ll2m([q["lat"] for q in g], [q["lon"] for q in g]).astype(np.float32))
ctree = cKDTree(np.array([p.mean(0) for p in polys]))
def dist_to_bld(X):
    out = np.zeros(len(X))
    for i, p in enumerate(X):
        best = 1e9
        for j in ctree.query_ball_point(p, 60):
            d = cv2.pointPolygonTest(polys[j].reshape(-1, 1, 2), (float(p[0]), float(p[1])), True); best = min(best, 0.0 if d >= 0 else -d)
        out[i] = best
    return out
names = list(rooms.keys()); R = np.array([[v[2], v[3]] for v in rooms.values()])
for grp in "2356":
    sel = np.array([k[0] == grp for k in names]); d = dist_to_bld(f(R[sel]))
    print(f"  {grp}xx: inside {(d == 0).mean():4.0%}, within 8 m {(d < 8).mean():4.0%}, within 15 m {(d < 15).mean():4.0%}, median {np.median(d):5.1f} m")
if not args.write: sys.exit(0)

step_f = 12.0; fgx = np.arange(0, 1009, step_f); fgy = np.arange(0, 793, step_f)
G = np.array([(x, y) for y in fgy for x in fgx]); GM = f(G)
# inverse: exact for an affine map
Bm = np.hstack([GM, np.ones((len(GM), 1))]); Ti = np.linalg.lstsq(Bm, G, rcond=None)[0]
mn = GM.min(0) - 40; mx = GM.max(0) + 40; step = 15.0
ix = np.arange(mn[0], mx[0] + step, step); iy = np.arange(mn[1], mx[1] + step, step); IG = np.array([(x, y) for y in iy for x in ix])
IGP = np.hstack([IG, np.ones((len(IG), 1))]) @ Ti
RM = f(R)
for (k, v), ll in zip(rooms.items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
for k, v in landmarks.items():
    ll = m2ll(f(np.array([[v[2], v[3]]])))[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
D["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
D["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(D, open(root / args.out, "w"), separators=(",", ":"))
print(f"wrote {args.out} (affine only)")
