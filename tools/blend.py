#!/usr/bin/env python3
"""Blend two georeferencings: v1 (data/appdata.v1.json, the original OSM-road fit, best around the Lodge and
casitas) inside the casita areas, v2 (data/appdata.v2.json, the named-road fit, best for roads, villas and
the clubhouse) everywhere else, with a smooth transition in PDF space. Writes data/appdata.json.

  python3 tools/blend.py [--write]
"""
import sys, json, pathlib, argparse
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--ramp", type=float, default=220.0); args = ap.parse_args()
V1 = json.load(open(root / "data/appdata.v1.json")); V2 = json.load(open(root / "data/appdata.v2.json"))
def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
f1 = lambda X: grid_lookup(V1["fwd"], X); f2 = lambda X: grid_lookup(V2["fwd"], X)

# casita areas in PDF points (north casitas 2xx, south casitas 3xx), from the room boxes with a margin
rooms = V1["rooms"]
def box(prefix, pad):
    P = np.array([[v[2], v[3]] for k, v in rooms.items() if k[0] == prefix]); return P.min(0) - pad, P.max(0) + pad
boxes = [box("2", 25), box("3", 25)]
def weight(X):
    """1 inside a casita box, 0 beyond `ramp` points from any box, smoothstep between."""
    d = np.full(len(X), np.inf)
    for lo, hi in boxes:
        dx = np.maximum(np.maximum(lo[0] - X[:, 0], X[:, 0] - hi[0]), 0); dy = np.maximum(np.maximum(lo[1] - X[:, 1], X[:, 1] - hi[1]), 0); d = np.minimum(d, np.hypot(dx, dy))
    t = np.clip(d / args.ramp, 0, 1); return 1 - (3 * t * t - 2 * t * t * t)
def f(X):
    w = weight(X)[:, None]; return w * f1(X) + (1 - w) * f2(X)

R = np.array([[v[2], v[3]] for v in rooms.values()])
def jac_dets(fn, X, h=0.5):
    fx = (fn(X + [h, 0]) - fn(X - [h, 0])) / (2 * h); fy = (fn(X + [0, h]) - fn(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
G24 = np.array([(x, y) for y in np.arange(0, 793, 12) for x in np.arange(0, 1009, 12)]); dg = jac_dets(f, G24)
print(f"blend ramp {args.ramp} pt: jacobian det on grid {dg.min():.2f}..{dg.max():.2f}, folds {(dg >= 0).sum()}; at rooms {jac_dets(f, R).min():.2f}..{jac_dets(f, R).max():.2f}")

# score against OSM building footprints
osm = json.load(open(root / "data/osm.json"))["elements"]; polys = []
for e in osm:
    t = e.get("tags", {}); g = e.get("geometry")
    if g and "building" in t and len(g) > 3: polys.append(ll2m([q["lat"] for q in g], [q["lon"] for q in g]).astype(np.float32))
ctree = cKDTree(np.array([p.mean(0) for p in polys]))
def dist_to_bld(P):
    out = np.zeros(len(P))
    for i, p in enumerate(P):
        best = 1e9
        for j in ctree.query_ball_point(p, 60):
            d = cv2.pointPolygonTest(polys[j].reshape(-1, 1, 2), (float(p[0]), float(p[1])), True); best = min(best, 0.0 if d >= 0 else -d)
        out[i] = best
    return out
names = list(rooms.keys())
for grp in "2356":
    sel = np.array([k[0] == grp for k in names])
    for label, fn in (("v1", f1), ("v2", f2), ("blend", f)):
        d = dist_to_bld(fn(R[sel])); print(f"  {grp}xx {label:5s}: inside {(d == 0).mean():4.0%}, within 8 m {(d < 8).mean():4.0%}, median {np.median(d):5.1f} m")
if not args.write: sys.exit(0)

step_f = 12.0; fgx = np.arange(0, 1009, step_f); fgy = np.arange(0, 793, step_f)
G = np.array([(x, y) for y in fgy for x in fgx]); GM = f(G)
mn = GM.min(0) - 40; mx = GM.max(0) + 40; step = 10.0
ix = np.arange(mn[0], mx[0] + step, step); iy = np.arange(mn[1], mx[1] + step, step); IG = np.array([(x, y) for y in iy for x in ix])
gm_tree = cKDTree(GM)
def inv_local(Mq, k=12):
    d, ii = gm_tree.query(Mq, k); out = np.zeros((len(Mq), 2))
    for r in range(len(Mq)):
        A = np.hstack([np.ones((k, 1)), GM[ii[r]]]); w = 1 / (d[r] + 5.0)
        th = np.linalg.lstsq(A * w[:, None], G[ii[r]] * w[:, None], rcond=None)[0]; out[r] = np.array([1, Mq[r, 0], Mq[r, 1]]) @ th
    return out
IGP = inv_local(IG)
D = json.loads(json.dumps(V2)); RM = f(R)
for (k, v), ll in zip(D["rooms"].items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
exact = {"Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9), "Club Lake": (317.6, 111.8), "Duck Pond": (-266.2, 12.4),
         "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Lodge (Main Lobby)": (-330.0, 60.0)}
for k, v in D["landmarks"].items():
    mm = np.array([exact[k]], float) if k in exact else f(np.array([[v[2], v[3]]])); ll = m2ll(mm)[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
D["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
D["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(D, open(root / "data/appdata.json", "w"), separators=(",", ":"))
err = np.linalg.norm(grid_lookup(D["inv"], RM) - R, axis=1); print(f"wrote data/appdata.json; inverse roundtrip at rooms median {np.median(err):.2f} pt max {err.max():.2f} pt")
