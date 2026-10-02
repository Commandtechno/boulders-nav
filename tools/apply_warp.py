#!/usr/bin/env python3
"""Apply a puppet warp exported from the verify page to a georeferencing.

The page exports {"base": <version id>, "pins": [{"o": [mx, my], "d": [dx, dy], "r": radius_m}, ...], "rooms": {id: [lat, lon]}}
where o is the pin's origin and d its drag, both in the app's local metre frame. The displacement at any
point is the Gaussian-weighted mean of the pins' drags (identical formula to the page), applied on top of
the base version's forward grid; rooms dragged individually override their position.

  python3 tools/apply_warp.py warp.json [--base data/appdata.v1.json] [--out data/appdata.json]
"""
import sys, json, pathlib, argparse
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m

ap = argparse.ArgumentParser(); ap.add_argument("warp"); ap.add_argument("--base"); ap.add_argument("--out", default="data/appdata.json"); args = ap.parse_args()
Wj = json.load(open(args.warp)); pins = Wj.get("pins", []); fixes = Wj.get("rooms", {})
base_path = args.base or f"data/appdata.{Wj.get('base') or 'v1'}.json"
BASE = json.load(open(root / base_path)); print(f"base {base_path}: {len(pins)} pins, {len(fixes)} dragged rooms")
def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
O = np.array([p["o"] for p in pins], float).reshape(-1, 2); Dd = np.array([p["d"] for p in pins], float).reshape(-1, 2); Rr = np.array([p["r"] for p in pins], float)
def disp(M):
    if not len(pins): return np.zeros_like(M)
    d2 = ((M[:, None, :] - O[None]) ** 2).sum(2); w = np.exp(-d2 / (2 * Rr[None] ** 2)); sw = w.sum(1, keepdims=True)
    return np.where(sw > 0, (w @ Dd) / np.maximum(sw, 1e-12) * (sw / (sw + 0.05)), 0.0)
f = lambda X: (lambda M: M + disp(M))(grid_lookup(BASE["fwd"], X))

rooms = BASE["rooms"]; R = np.array([[v[2], v[3]] for v in rooms.values()])
step_f = 12.0; fgx = np.arange(0, 1009, step_f); fgy = np.arange(0, 793, step_f)
G = np.array([(x, y) for y in fgy for x in fgx]); GM = f(G)
def jac(X, h=0.5):
    fx = (f(X + [h, 0]) - f(X - [h, 0])) / (2 * h); fy = (f(X + [0, h]) - f(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
dg = jac(G); print(f"jacobian det on grid {dg.min():.2f}..{dg.max():.2f}, folds {(dg >= 0).sum()}; max shift {np.linalg.norm(GM - grid_lookup(BASE['fwd'], G), axis=1).max():.1f} m")
mn = GM.min(0) - 40; mx = GM.max(0) + 40; step = 10.0
ix = np.arange(mn[0], mx[0] + step, step); iy = np.arange(mn[1], mx[1] + step, step); IG = np.array([(x, y) for y in iy for x in ix])
tree = cKDTree(GM)
def inv_local(Mq, k=12):
    d, ii = tree.query(Mq, k); out = np.zeros((len(Mq), 2))
    for r in range(len(Mq)):
        A = np.hstack([np.ones((k, 1)), GM[ii[r]]]); w = 1 / (d[r] + 5.0)
        th = np.linalg.lstsq(A * w[:, None], G[ii[r]] * w[:, None], rcond=None)[0]; out[r] = np.array([1, Mq[r, 0], Mq[r, 1]]) @ th
    return out
IGP = inv_local(IG)
D = json.loads(json.dumps(BASE)); RM = f(R)
for (k, v), ll in zip(D["rooms"].items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
for k, ll in fixes.items():
    if k in D["rooms"]: D["rooms"][k][0] = round(float(ll[0]), 6); D["rooms"][k][1] = round(float(ll[1]), 6)
for k, v in D["landmarks"].items():
    ll = m2ll(f(np.array([[v[2], v[3]]])))[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
D["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
D["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(D, open(root / args.out, "w"), separators=(",", ":"))
err = np.linalg.norm(grid_lookup(D["inv"], RM) - R, axis=1); print(f"wrote {args.out}; inverse roundtrip at rooms median {np.median(err):.2f} pt max {err.max():.2f} pt")
