#!/usr/bin/env python3
EXPERIMENTAL, not applied to the shipped data: see README "Accuracy".
"""Stage 2 of the registration: pull the casita clusters onto the buildings seen in the aerial.

tools/register.py fixes the roads. Between the roads the illustration is still schematic, so here the
casita room boxes (rooms 2xx/3xx) are grouped into buildings, matched to white-roof blobs from the aerial
(tools/detect_features.py), and a smooth residual field (Gaussian-weighted, ~35 m scale) is added on top
of the road warp. Road strokes and known anchors are pinned with zero residual so they do not move.

  python3 tools/refine_casitas.py           # diagnostics + overlays (/tmp/ref_*.png)
  python3 tools/refine_casitas.py --write   # update data/appdata.json
"""
import sys, json, pathlib, shutil, argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, DATA, ll2m, m2ll, Mosaic

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--sigma", type=float, default=35.0); args = ap.parse_args()
rooms = DATA["rooms"]; landmarks = DATA["landmarks"]; roads = json.load(open(root / "data/roads_pdf.json"))
OB = np.load(root / "data/osm_buildings.npz"); osm_b = OB["m"].astype(float); osm_a = OB["area"].astype(float)      # OSM building centroids (tools: see register.py)

def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
def resample(pl, step):
    pl = np.asarray(pl, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
    if d[-1] < step: return pl
    s = np.arange(0, d[-1], step); return np.c_[np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])]
f0 = lambda X: grid_lookup(DATA["fwd"], X)

# ---------- buildings on both sides ----------
cas_ids = [k for k in rooms if k[0] in "23"]; RP = np.array([[rooms[k][2], rooms[k][3]] for k in cas_ids])
cl = fcluster(linkage(RP, "single"), 12.0, "distance")
CP = np.array([RP[cl == k].mean(0) for k in np.unique(cl)]); CN = np.array([(cl == k).sum() for k in np.unique(cl)])
# candidate buildings: casita-sized OSM footprints within reach of where stage 1 put the casitas
near = cKDTree(f0(CP)).query(osm_b)[0] < 110
bld = osm_b[near & (osm_a > 40) & (osm_a < 1000)]; bld_tree = cKDTree(bld)
print(f"{len(CP)} casita buildings from {len(RP)} rooms (sizes {np.bincount(CN)[1:].tolist()}), {len(bld)} candidate OSM buildings")

# ---------- pins: roads and anchors keep their stage-1 positions ----------
pins = np.vstack([resample(np.asarray(pl, float), 6.0) for pl in roads["gray5"] if len(pl) > 1] + [resample(np.asarray(roads["tan5"][i], float), 6.0) for i in (0, 10)])
pinsM = f0(pins)

# ---------- residual field ----------
def make_field(Xc, res, w, sigma):
    """corr(m) = weighted Gaussian average of residual vectors around the metre position m (pins have zero residual)."""
    allX = np.vstack([Xc, pinsM]); allR = np.vstack([res, np.zeros_like(pinsM)]); allW = np.concatenate([w, np.full(len(pinsM), 0.6)])
    tree = cKDTree(allX)
    def corr(Mq):
        out = np.zeros_like(Mq)
        for s in range(0, len(Mq), 500):
            q = Mq[s:s + 500]; nb = tree.query_ball_point(q, 3 * sigma)
            for r, ids in enumerate(nb):
                if not ids: continue
                ids = np.array(ids); d2 = ((allX[ids] - q[r]) ** 2).sum(1); ww = np.exp(-d2 / (2 * sigma ** 2)) * allW[ids]
                sw = ww.sum(); out[s + r] = (ww @ allR[ids]) / max(sw, 1e-9) * (sw / (sw + 0.5))     # shrink where support is thin
        return out
    return corr
corr = lambda M: np.zeros_like(M)
f = lambda X: (lambda M: M + corr(M))(f0(X))

# One-to-one assignment (Hungarian) between PDF casita buildings and OSM buildings, coarse to fine.
# Nearest-neighbour matching lets several buildings collapse onto one footprint; assignment keeps the
# sequence of buildings along each lane intact, which is what the illustration preserves.
from scipy.optimize import linear_sum_assignment
def assign(Pm, gate):
    D = np.linalg.norm(Pm[:, None] - bld[None], axis=2); r, c = linear_sum_assignment(np.minimum(D, gate * 1.5))
    ok = D[r, c] < gate; return r[ok], c[ok]
for it in range(18):
    gate = max(12.0, 90.0 - 5.0 * it); sigma = max(args.sigma, 80.0 - 3.5 * it)
    Pm0 = f0(CP); Pm = Pm0 + corr(Pm0); r, c = assign(Pm, gate)
    corr = make_field(Pm0[r], bld[c] - Pm0[r], np.sqrt(CN[r]), sigma)
r, c = assign(f(CP), 15.0); print(f"assigned {len(r)} of {len(CP)} casita buildings to distinct OSM buildings (within 15 m)")
def report(name, fn):
    d, _ = bld_tree.query(fn(RP)); cov = cKDTree(fn(RP)).query(bld)[0]
    print(f"{name}: rooms to nearest building: med {np.median(d):.1f} m, p90 {np.percentile(d, 90):.1f}, <8 m {(d < 8).mean():.0%}, >20 m {(d > 20).sum()} of {len(d)} | buildings with a room within 12 m: {(cov < 12).sum()} of {len(bld)}")
report("before", f0); report("after ", f)
pd = np.linalg.norm(f(pins) - pinsM, axis=1); print(f"road pins moved: max {pd.max():.2f} m")
R = np.array([[v[2], v[3]] for v in rooms.values()])
def jac_dets(fn, X, h=0.5):
    fx = (fn(X + [h, 0]) - fn(X - [h, 0])) / (2 * h); fy = (fn(X + [0, h]) - fn(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
dets = jac_dets(f, R); print(f"jacobian det at rooms {dets.min():.2f}..{dets.max():.2f} (folds: {(dets >= 0).sum()})")

# ---------- overlays ----------
from PIL import Image, ImageDraw, ImageFont
M = Mosaic(20, down=2); im = Image.fromarray(M.img); dr = ImageDraw.Draw(im); font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
for q in M.m2px(bld): dr.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=(255, 0, 255))
for cls, col in (("tan5", (255, 140, 0)), ("gray1", (0, 200, 255))):
    for pl in roads[cls]:
        if len(pl) > 1: dr.line([tuple(q) for q in M.m2px(f(resample(np.asarray(pl, float), 2.0)))], fill=col, width=2)
for q0, q1, name in zip(M.m2px(f0(R)), M.m2px(f(R)), rooms.keys()):
    dr.line([tuple(q0), tuple(q1)], fill=(255, 60, 60), width=1)
    dr.ellipse([q1[0] - 4, q1[1] - 4, q1[0] + 4, q1[1] + 4], outline=(80, 255, 80), width=2); dr.text((q1[0] + 5, q1[1] - 6), name, fill=(255, 255, 0), font=font)
for name, (x0, x1, y0, y1) in {"north": (-560, -150, 150, 420), "south": (-330, 0, -300, -60), "farsouth": (-330, -80, -420, -220)}.items():
    c = M.m2px(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)); im.crop((*c.min(0).astype(int), *c.max(0).astype(int))).save(f"/tmp/ref_{name}.png")
print("overlays: /tmp/ref_*.png")
if not args.write: sys.exit(0)

# ---------- write ----------
shutil.copy(root / "data/appdata.json", root / "data/appdata.stage1.json")
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
RM = f(R)
for (k, v), ll in zip(rooms.items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
exact = {"Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9), "Club Lake": (317.6, 111.8), "Duck Pond": (-266.2, 12.4),
         "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Lodge (Main Lobby)": (-330.0, 60.0)}
for k, v in landmarks.items():
    m = np.array([exact[k]], float) if k in exact else f(np.array([[v[2], v[3]]])); ll = m2ll(m)[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
d = DATA
d["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
d["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(d, open(root / "data/appdata.json", "w"), separators=(",", ":"))
err = np.linalg.norm(grid_lookup(d["inv"], RM) - R, axis=1); print(f"wrote data/appdata.json; inverse roundtrip at rooms median {np.median(err):.2f} pt max {err.max():.2f} pt")
