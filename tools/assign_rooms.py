#!/usr/bin/env python3
"""Assign room numbers to real buildings by walking each lane in the drawing's order.

The drawing is schematic but its ORDER is reliable: along any lane the room numbers appear in the
sequence a guest meets them. Real building footprints (OpenStreetMap, which match the 10 cm aerial)
are exact but unlabelled. So, per lane:
  1. rooms on the lane, ordered by their position along it (from the drawing)
  2. candidate buildings: footprints within reach of the lane as placed by the geometric warp
     (data/appdata.<base>.json, normally the aerial-aligned one), ordered along the same lane
  3. monotone alignment (dynamic programming): rooms map to buildings in order, several units may
     share one building, buildings may be skipped, cost = distance warped-room -> footprint centroid
Rooms end up on footprint centroids (spread slightly when a building holds several units).
Villas and haciendas are left where the warp puts them (already >80% on footprints).

  python3 tools/assign_rooms.py [--base ffd] [--out data/appdata.json]
"""
import sys, json, pathlib, argparse
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m, Mosaic

ap = argparse.ArgumentParser(); ap.add_argument("--base", default="ffd"); ap.add_argument("--out", default="data/appdata.json"); ap.add_argument("--reach", type=float, default=45.0); args = ap.parse_args()
D = json.load(open(root / f"data/appdata.{args.base}.json")); rooms = D["rooms"]; roads = json.load(open(root / "data/roads_pdf.json"))
def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
f = lambda X: grid_lookup(D["fwd"], np.asarray(X, float))

# ---------- real buildings ----------
osm = json.load(open(root / "data/osm.json"))["elements"]; polys = []
for e in osm:
    t = e.get("tags", {}); g = e.get("geometry")
    if g and "building" in t and len(g) > 3:
        P = ll2m([q["lat"] for q in g], [q["lon"] for q in g]); A = abs(cv2.contourArea(P.astype(np.float32)))
        if 40 < A < 2600: polys.append((P, A, P.mean(0)))
B = np.array([p[2] for p in polys]); BA = np.array([p[1] for p in polys]); btree = cKDTree(B)
print(f"{len(B)} candidate buildings (40-2600 m2)")

# ---------- groups: runs of consecutive numbers that are spatially contiguous in the drawing ----------
cas = dict(rooms); names = sorted(cas.keys(), key=int); RP = np.array([[cas[k][2], cas[k][3]] for k in names])
groups = [[0]]
for i in range(1, len(names)):
    if int(names[i]) - int(names[i - 1]) <= 2 and np.linalg.norm(RP[i] - RP[i - 1]) < 45: groups[-1].append(i)
    else: groups.append([i])
# tiny runs (1-2 rooms) join the numerically adjacent run when they are drawn within 120 pt of it
merged = []
for g in groups:
    if len(g) <= 2 and merged and int(names[g[0]]) - int(names[merged[-1][-1]]) <= 2 and np.linalg.norm(RP[g[0]] - RP[merged[-1][-1]]) < 120: merged[-1].extend(g)
    else: merged.append(g)
groups = merged
print(f"{len(groups)} number runs:", [f"{names[g[0]]}-{names[g[-1]]}" for g in groups])
cap_of = lambda j: int(np.clip(round(BA[j] / 90.0), 1, 5))        # casita units are ~90 m2 of footprint each

# ---------- per run: monotone alignment with building capacities ----------
assigned = {}; report = []
for g in groups:
    idx = np.array(g); RM = f(RP[idx]); n = len(idx)
    cand = sorted({j for m in RM for j in btree.query_ball_point(m, args.reach)})
    if not cand: print("  no buildings near", names[idx[0]]); continue
    if n == 1:
        assigned[names[idx[0]]] = cand[int(np.argmin(np.linalg.norm(B[cand] - RM[0], axis=1)))]; continue
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(RM, axis=0), axis=1))]
    dense = np.vstack([np.c_[np.linspace(RM[k, 0], RM[k + 1, 0], 20), np.linspace(RM[k, 1], RM[k + 1, 1], 20)] for k in range(n - 1)])
    dense_s = np.concatenate([np.linspace(d[k], d[k + 1], 20) for k in range(n - 1)])
    lt = cKDTree(dense); dq, iq = lt.query(B[cand]); proj = dense_s[iq]; order = np.argsort(proj); cand = [cand[k] for k in order]; near_lane = dq[order] < 22
    m = len(cand); soft = [cap_of(j) for j in cand]; C = 8; caps = [C] * m        # soft capacity from area, hard cap 8
    dist = np.linalg.norm(RM[:, None, :] - B[cand][None, :, :], axis=2)
    SKIP = 8.0      # metres-equivalent for leaving a building next to the lane without a number
    INF = 1e18
    # state: (room i, building j, count c of rooms already placed in j including room i) -> cost
    cost = np.full((n, m, C + 1), INF); back = {}
    for j in range(m):
        cost[0, j, 1] = dist[0, j] + SKIP * near_lane[:j].sum()
    for i in range(1, n):
        for j in range(m):
            # stay in the same building (count+1)
            for c in range(2, caps[j] + 1):
                v = cost[i - 1, j, c - 1]
                if v < INF: cost[i, j, c] = v + dist[i, j] + 3.0 * (c - 1) + 20.0 * max(0, c - soft[j]); back[(i, j, c)] = (j, c - 1)
            # come from an earlier building k < j, paying skips for empty buildings between k and j
            bestv, bestk = INF, None
            for k in range(j):
                v = cost[i - 1, k].min()
                if v >= INF: continue
                v = v + SKIP * near_lane[k + 1:j].sum()
                if v < bestv: bestv, bestk = v, (k, int(cost[i - 1, k].argmin()))
            if bestk is not None: cost[i, j, 1] = bestv + dist[i, j]; back[(i, j, 1)] = bestk
    # end: pay for near-lane buildings after the last one used
    final = np.array([[cost[n - 1, j, c] + SKIP * near_lane[j + 1:].sum() for c in range(C + 1)] for j in range(m)])
    j, c = np.unravel_index(int(final.argmin()), final.shape); seq = []
    for i in range(n - 1, -1, -1):
        seq.append(j)
        if i > 0: j, c = back[(i, j, c)]
    seq = seq[::-1]
    for k, i in enumerate(idx): assigned[names[i]] = cand[seq[k]]
    report.append((names[idx[0]], names[idx[-1]], n, len(set(seq)), np.median(dist[np.arange(n), seq])))
for a, b, n, nb, med in report: print(f"  {a}-{b}: {n} rooms -> {nb} buildings, median move {med:.0f} m")

# ---------- place rooms: footprint centroid, spread units of one building along its long axis ----------
by_b = {}
for k, j in assigned.items(): by_b.setdefault(j, []).append(k)
out = {}
for j, ks in by_b.items():
    P, A, c = polys[j]; ks = sorted(ks, key=int)
    if len(ks) == 1: out[ks[0]] = c; continue
    Q = P - c; _, _, vt = np.linalg.svd(Q, full_matrices=False); ax = vt[0]; ext = (Q @ ax); lo, hi = np.percentile(ext, [15, 85])
    for t, k in enumerate(ks): out[k] = c + ax * (lo + (hi - lo) * (t + 0.5) / len(ks))
moved = []
for k, mpos in out.items():
    old = ll2m(rooms[k][0], rooms[k][1])[0]; moved.append(np.linalg.norm(mpos - old))
    ll = m2ll(np.array([mpos]))[0]; rooms[k][0] = round(float(ll[0]), 6); rooms[k][1] = round(float(ll[1]), 6)
print(f"placed {len(out)} rooms on {len(by_b)} buildings; move from warp median {np.median(moved):.1f} m, max {max(moved):.1f} m; unassigned: {[k for k in names if k not in out]}")
json.dump(D, open(root / args.out, "w"), separators=(",", ":")); print(f"wrote {args.out}")

# ---------- picture ----------
from PIL import Image, ImageDraw, ImageFont
M = Mosaic(20, down=2); im = Image.fromarray(M.img); dr = ImageDraw.Draw(im); font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
G = D["graph"]; N = f(np.array(G["nodes"], float)); PN = M.m2px(N)
for a, b in G["edges"]: dr.line([tuple(PN[a]), tuple(PN[b])], fill=(0, 255, 255), width=2)
for P, A, c in polys: pts = [tuple(q) for q in M.m2px(P)]; dr.line(pts + [pts[0]], fill=(255, 80, 80), width=1)
for k, v in rooms.items():
    q = M.m2px(ll2m(v[0], v[1]))[0]; dr.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=(255, 255, 0)); dr.text((q[0] + 5, q[1] - 7), k, fill=(255, 255, 0), font=font)
for name, (x0, x1, y0, y1) in {"north": (-600, -180, 150, 430), "south": (-340, -60, -300, -50)}.items():
    c = M.m2px(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)); im.crop((*c.min(0).astype(int), *c.max(0).astype(int))).save(f"/tmp/assign_{name}.png")
