#!/usr/bin/env python3
"""Re-register the resort PDF map to reality.

The illustration is schematic: parts of it sit 50-120 m from where they belong, so blind nearest-neighbour
matching cannot recover the warp. Instead every labelled road on the map is matched to the SAME road in
OpenStreetMap by name (OSM lines up with the City of Scottsdale 10 cm aerials to within a few metres), and
features that are unambiguous in the aerial (pools, lakes, the lodge, the clubhouse, road junctions) anchor
the fit. A thin-plate spline (PDF points -> local metres) is fitted by ICP on those named correspondences,
then the forward/inverse lookup grids in data/appdata.json are rebuilt and all room/landmark coordinates
recomputed.

  python3 tools/register.py            # fit, print diagnostics, write overlays to /tmp/reg_*.png
  python3 tools/register.py --write    # also update data/appdata.json (backup in data/appdata.prev.json)
"""
import sys, json, pathlib, shutil, argparse
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, DATA, ll2m, m2ll, Mosaic

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--lam", type=float, default=3000.0)
ap.add_argument("--centers", type=float, default=50.0, help="TPS centre spacing in PDF points"); args = ap.parse_args()

OSM = root / "data/osm.json"
osm = json.load(open(OSM))["elements"]
roads = json.load(open(root / "data/roads_pdf.json"))
rooms = DATA["rooms"]; landmarks = DATA["landmarks"]
F = np.load(root / "data/aerial_features.npz"); skel = F["skel_m"].astype(float); skel_tree = cKDTree(skel)

# ---------- helpers ----------
def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
def resample(pl, step):
    pl = np.asarray(pl, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
    if d[-1] < step: return pl
    s = np.arange(0, d[-1], step); return np.c_[np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])]
def U(r): out = np.zeros_like(r); m = r > 0; out[m] = r[m] ** 2 * np.log(r[m]); return out
class TPS:
    def __init__(s, C): s.C = C; s.n = len(C); s.K = U(np.linalg.norm(C[:, None] - C[None], axis=2))
    def basis(s, X): return np.hstack([np.ones((len(X), 1)), X, U(np.linalg.norm(X[:, None] - s.C[None], axis=2))])
    def fit(s, X, Y, w, lam):
        B = s.basis(X) * np.sqrt(w)[:, None]; Yw = Y * np.sqrt(w)[:, None]
        Rg = np.zeros((3 + s.n, 3 + s.n)); Rg[3:, 3:] = s.K * lam
        Pc = np.hstack([np.zeros((3, 3)), np.vstack([np.ones(s.n), s.C.T])]) * 1e4
        A = B.T @ B + Pc.T @ Pc + Rg; A[np.diag_indices_from(A)] += 1e-6
        s.theta = np.linalg.solve(A, B.T @ Yw); return s
    def __call__(s, X): return s.basis(X) @ s.theta
def jac_dets(f, X, h=0.5):
    fx = (f(X + [h, 0]) - f(X - [h, 0])) / (2 * h); fy = (f(X + [0, h]) - f(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]

# ---------- real-world targets from OSM ----------
ways = {}
for e in osm:
    t = e.get("tags", {}); g = e.get("geometry")
    if g and "highway" in t: ways[e["id"]] = (t.get("name", ""), ll2m([q["lat"] for q in g], [q["lon"] for q in g]))
def named(*names, ids=None, pred=lambda M: True):
    out = [M for wid, (n, M) in ways.items() if (n in names or (ids and wid in ids)) and pred(M)]
    return np.vstack([resample(M, 2.0) for M in out])
CLUB_N = {961180982, 961180983, 961180984, 961180985}                     # clubhouse -> Boulder Pass (north branch)
CLUB_S = {961180978, 961180979, 961180980, 961180981}                     # lodge -> Boulders Pkwy -> clubhouse (south branch)
CLUB_SW = {5638838, 529875753, 757507240, 961180963, 961180964, 961180965, 961180969, 961180971}   # lodge -> Carefree Hwy
targets = {
    "north":      named("North Boulder Pass", ids=CLUB_N, pred=lambda M: M[:, 1].mean() < 500),
    "south":      named(ids=CLUB_S),
    "sw":         named(ids=CLUB_SW),
    "pkwy":       named("East Clubhouse Road", "Boulders Parkway", "North Boulders Parkway"),
    "clubct":     named("East Clubhouse Court"),
    "sunflower":  named("East Sunflower Court"),
    "whitethorn": named("East Whitethorn Circle"),
    "haciendas":  named("East Clubhouse Drive"),
}
targets["pkwy"] = targets["pkwy"][targets["pkwy"][:, 1] > -520]        # the map stops well short of the Parkway's south end
trees = {k: cKDTree(v) for k, v in targets.items()}

# ---------- PDF sources ----------
g5 = [np.asarray(pl, float) for pl in roads["gray5"]]; t5 = [np.asarray(pl, float) for pl in roads["tan5"]]
loop = g5[8]; split = int(np.argmax(loop[:, 0]))          # the gate -> clubhouse -> lodge loop, split at the clubhouse end
sources = [  # (PDF polyline, target name)
    (loop[:split + 1], "north"), (loop[split:], "south"),
    (g5[0], "pkwy"), (g5[1], "clubct"), (g5[2], "sunflower"), (g5[3], "whitethorn"), (g5[4], "whitethorn"),
    (g5[5], "haciendas"), (t5[10], "sw"),
]
samples = [(resample(pl, 3.0), k) for pl, k in sources]

# ---------- anchors: PDF point (pt) -> real point (m), weight ----------
def wnode(name, near):          # OSM node of a named road closest to a guess
    M = np.vstack([M for n, M in ways.values() if n == name]); return M[np.argmin(np.linalg.norm(M - near, axis=1))]
anchors = [
    ((96, 74),    (-705.7, 415.5), 10, "gate / Tom Darlington"),
    ((157, 117),  (-587.5, 305.3), 10, "spa pool"),
    ((130, 98),   (-625.0, 345.0), 4,  "spa building"),
    ((273, 447),  (-348.6, 25.9),  10, "lodge"),
    ((240, 426),  (-390.6, 40.9),  10, "lodge pool"),
    ((370, 450),  (-266.2, 12.4),  5,  "duck pond"),
    ((249, 297),  (-369.7, 172.8), 3,  "#5 green"),
    ((470, 563),  (-169.2, -114.7), 3, "#6 green"),
    ((203, 632),  (-394.2, -139.4), 2, "#7 green"),
    ((728, 580),  (85.7, -164.0),  10, "villa pool"),
    ((831, 303),  (223.7, 213.9),  8,  "clubhouse pool"),
    ((854, 297),  (259.2, 235.9),  8,  "clubhouse"),
    ((795, 330),  (165.0, 180.0),  5,  "tennis"),
    ((875, 365),  (317.6, 111.8),  8,  "club lake"),
    ((105, 711),  (-608.1, -162.8), 5, "El Pedregal"),
    ((640, 355),  (24.6, 137.4),   8,  "Pkwy x Clubhouse Rd"),
    ((692, 547),  (26.6, -146.2),  8,  "Pkwy x Whitethorn"),
    ((688, 585),  (0.0, -191.2),   8,  "Pkwy x Sunflower"),
    ((680, 615),  (-5.0, -221.5),  8,  "Pkwy x Clubhouse Ct"),
    ((555, 360),  (-73.9, 97.9),   6,  "E Clubhouse Rd x Clubhouse Rd"),
    ((458, 368),  (-191.8, 125.2), 6,  "Haciendas loop x Clubhouse Rd"),
    ((346, 346),  (-318.0, 84.0),  4,  "Clubhouse Rd at lodge"),
    ((773, 257),  tuple(wnode("North Clubhouse Road", np.array([148, 285]))), 4, "Clubhouse Rd at clubhouse parking"),
]
AP = np.array([a[0] for a in anchors], float); AM = np.array([a[1] for a in anchors], float); AW = np.array([a[2] for a in anchors], float)

# ---------- fit ----------
gx = np.arange(0, 1008.1, args.centers); gy = np.arange(0, 792.1, args.centers)
tps = TPS(np.array([(x, y) for y in gy for x in gx]))
f_prev = lambda X: grid_lookup(DATA["fwd"], X)
f = TPS(tps.C).fit(AP, AM, AW, args.lam * 3)                 # anchors alone give the coarse layout
for it in range(16):
    gate = max(6.0, 40.0 - 2.5 * it)
    X, Y, W = [AP], [AM], [AW]
    for P, k in samples:
        d, j = trees[k].query(f(P)); ok = d < gate
        X.append(P[ok]); Y.append(targets[k][j[ok]]); W.append(np.full(ok.sum(), 1.0))
    # the spa driveway (gate -> spa) has no OSM way: match it to the asphalt skeleton once roughly in place
    if it >= 6:
        P = resample(t5[0], 3.0); d, j = skel_tree.query(f(P)); ok = d < min(gate, 12)
        X.append(P[ok]); Y.append(skel[j[ok]]); W.append(np.full(ok.sum(), 0.7))
    tps.fit(np.vstack(X), np.vstack(Y), np.concatenate(W), args.lam); f = tps

# ---------- diagnostics ----------
def report(name, f):
    print(f"--- {name}")
    for P, k in samples:
        d, _ = trees[k].query(f(P)); print(f"  {k:11s} n={len(P):4d}  med {np.median(d):5.1f} m  p90 {np.percentile(d, 90):5.1f}  <5 m {(d < 5).mean():4.0%}")
    r = np.linalg.norm(f(AP) - AM, axis=1)
    print("  anchors: " + ", ".join(f"{a[3]} {e:.0f}" for a, e in zip(anchors, r) if e > 6) + f"  | median {np.median(r):.1f} max {r.max():.1f}")
    R = np.array([[v[2], v[3]] for v in rooms.values()]); dets = jac_dets(f, R); print(f"  jacobian det at rooms {dets.min():.2f}..{dets.max():.2f} (all same sign: {(np.sign(dets) == np.sign(dets[0])).all()})")
report("before (current appdata)", f_prev); report("after", f)

# ---------- overlays ----------
from PIL import Image, ImageDraw, ImageFont
M = Mosaic(20, down=2); im = Image.fromarray(M.img); dr = ImageDraw.Draw(im); font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
for k, T in targets.items():
    for p in M.m2px(T): dr.point((p[0], p[1]), fill=(255, 0, 255))
for P, k in samples:
    for fn, col in ((f_prev, (255, 60, 60)), (f, (80, 255, 80))): dr.line([tuple(q) for q in M.m2px(fn(P))], fill=col, width=2)
R = np.array([[v[2], v[3]] for v in rooms.values()])
for q, name in zip(M.m2px(f(R)), rooms.keys()): dr.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=(80, 255, 80), width=2); dr.text((q[0] + 5, q[1] - 6), name, fill=(255, 255, 0), font=font)
for q in M.m2px(AM): dr.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=(0, 255, 255), width=2)
crops = {"north": (-520, 150, 150, 450), "lodge": (-460, -120, -200, 110), "south": (-300, 60, -420, -160), "villas": (-160, 200, -400, -100), "club": (-100, 400, 30, 320), "all": None}
for name, box in crops.items():
    if box is None: im.resize((im.width // 4, im.height // 4)).save("/tmp/reg_all.png"); continue
    x0, x1, y0, y1 = box; c = M.m2px(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)); im.crop((*c.min(0).astype(int), *c.max(0).astype(int))).save(f"/tmp/reg_{name}.png")
print("overlays written to /tmp/reg_*.png")

# fold check: the PDF has y down and metres have y up, so a healthy warp has det < 0 everywhere
names = list(rooms.keys()); dets = jac_dets(f, R)
bad = [(n, round(d, 2), R[i].round(0).tolist()) for i, (n, d) in enumerate(zip(names, dets)) if d > -0.3]
if bad: print("WARNING rooms with det > -0.3 (folded or squashed):", bad)
G24 = np.array([(x, y) for y in np.arange(0, 793, 24) for x in np.arange(0, 1009, 24)]); dg = jac_dets(f, G24)
if (dg >= 0).any(): print("WARNING grid cells with det >= 0:", [(int(x), int(y), round(d, 2)) for (x, y), d in zip(G24, dg) if d >= 0][:40])
for P, k in samples:
    d, _ = trees[k].query(f(P))
    if (d > 20).any(): print(f"  {k}: {(d > 20).sum()} samples >20 m off, PDF pts e.g. {P[d > 20][::6].round(0).tolist()}")
allP = np.vstack([P for P, k in samples]); alld = np.concatenate([trees[k].query(f(P))[0] for P, k in samples])
print(f"SUMMARY lam={args.lam} centers={args.centers}: roads med {np.median(alld):.1f} p90 {np.percentile(alld, 90):.1f} >10m {(alld > 10).mean():.0%} | anchors max {np.linalg.norm(f(AP) - AM, axis=1).max():.1f} | folded rooms {sum(1 for d in dets if d > -0.3)} folded cells {(dg >= 0).sum()} det range {dets.min():.2f}..{dets.max():.2f}")
if not args.write: sys.exit(0)
# ---------- write grids + coordinates ----------
shutil.copy(root / "data/appdata.json", root / "data/appdata.prev.json")
step_f = 12.0; fgx = np.arange(0, 1009, step_f); fgy = np.arange(0, 793, step_f)
G = np.array([(x, y) for y in fgy for x in fgx]); GM = f(G)
mn = GM.min(0) - 40; mx = GM.max(0) + 40; step = 15.0
ix = np.arange(mn[0], mx[0] + step, step); iy = np.arange(mn[1], mx[1] + step, step); IG = np.array([(x, y) for y in iy for x in ix])
gm_tree = cKDTree(GM)
def inv_local(Mq, k=12):
    d, ii = gm_tree.query(Mq, k); out = np.zeros((len(Mq), 2))
    for r in range(len(Mq)):
        A = np.hstack([np.ones((k, 1)), GM[ii[r]]]); w = 1 / (d[r] + 5.0)
        th = np.linalg.lstsq(A * w[:, None], G[ii[r]] * w[:, None], rcond=None)[0]; out[r] = np.array([1, Mq[r, 0], Mq[r, 1]]) @ th
    return out
IGP = inv_local(IG)
RM = f(R); LL = m2ll(RM)
for (k, v), ll in zip(rooms.items(), LL): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
for k, v in landmarks.items():
    ll = m2ll(f(np.array([[v[2], v[3]]])))[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
# features whose true position is known exactly from OSM / the aerial override the warp
exact = {"Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9), "Club Lake": (317.6, 111.8), "Duck Pond": (-266.2, 12.4),
         "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Lodge (Main Lobby)": (-330.0, 60.0)}
for k, m in exact.items():
    if k in landmarks: ll = m2ll(np.array([m], float))[0]; landmarks[k][0] = round(float(ll[0]), 6); landmarks[k][1] = round(float(ll[1]), 6)
d = DATA
d["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
d["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(d, open(root / "data/appdata.json", "w"), separators=(",", ":"))
rt = grid_lookup(d["inv"], RM); err = np.linalg.norm(rt - R, axis=1)
print(f"wrote data/appdata.json; inverse-grid roundtrip at rooms median {np.median(err):.2f} pt, max {err.max():.2f} pt")
