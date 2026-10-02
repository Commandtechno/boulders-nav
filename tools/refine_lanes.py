#!/usr/bin/env python3
EXPERIMENTAL, not applied to the shipped data: see README "Accuracy".
"""Stage 2 of the registration: align the casita lanes and footpaths to the paths visible in the aerial.

tools/register.py fixes the roads (asphalt). The pedestrian lanes between the casitas are drawn
schematically, so here every lane stroke (tan5) and footpath (gray1) is ICP-matched to a skeleton of
"path" pixels from the aerial (asphalt plus the light, smooth, thin paved walkways), and the result is
applied as a smooth residual field on top of the road warp. Road strokes are pinned so they do not move.

  python3 tools/refine_lanes.py           # diagnostics + overlays (/tmp/lane_*.png)
  python3 tools/refine_lanes.py --write   # update data/appdata.json
"""
import sys, json, pathlib, shutil, argparse
import numpy as np, cv2
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, DATA, ll2m, m2ll, Mosaic

ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); ap.add_argument("--sigma", type=float, default=30.0); args = ap.parse_args()
rooms = DATA["rooms"]; landmarks = DATA["landmarks"]; roads = json.load(open(root / "data/roads_pdf.json"))

def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
def resample(pl, step):
    pl = np.asarray(pl, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
    if d[-1] < step: return pl
    s = np.arange(0, d[-1], step); return np.c_[np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])]
def unit(v): return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
f0 = lambda X: grid_lookup(DATA["fwd"], X)

# ---------- path skeleton from the aerial ----------
M = Mosaic(20, down=2); img = M.img
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); S, V = hsv[..., 1], hsv[..., 2]; r, g, b = [img[..., i].astype(int) for i in range(3)]
asphalt = ((S < 40) & (V > 60) & (V < 150) & (np.abs(r - b) < 22) & (np.abs(g - b) < 22)).astype(np.uint8)
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
asphalt = cv2.morphologyEx(cv2.morphologyEx(asphalt, cv2.MORPH_OPEN, k5), cv2.MORPH_CLOSE, k9)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
mu = cv2.blur(gray, (7, 7)); sd = np.sqrt(np.maximum(cv2.blur(gray * gray, (7, 7)) - mu * mu, 0))
smooth = ((sd < 9) & (V > 120) & (V < 215) & (S < 90) & (g < r)).astype(np.uint8)              # flat, light, not vegetation
smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, k5)
wide = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))   # sand flats, roofs, parking
lane = cv2.morphologyEx(cv2.subtract(smooth, wide), cv2.MORPH_OPEN, k5)
n, lab, stats, _ = cv2.connectedComponentsWithStats(lane, 8)
keep = np.zeros(n, bool); keep[1:] = (stats[1:, cv2.CC_STAT_AREA] > 400) & (np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) > 80)
paths = (keep[lab] | (asphalt > 0))
sk = skeletonize(paths); ys, xs = np.nonzero(sk); skel = M.px2m(np.c_[xs + 0.5, ys + 0.5]); skel_tree = cKDTree(skel)
T = np.zeros_like(skel)
for i, nb in enumerate(skel_tree.query_ball_point(skel, 2.0)):
    if len(nb) < 3: T[i] = [1, 0]; continue
    Q = skel[nb] - skel[nb].mean(0); T[i] = np.linalg.svd(Q, full_matrices=False)[2][0]
print(f"path skeleton: {len(skel)} px ({len(skel) * M.res / 1000:.1f} km)")

# ---------- strokes ----------
def plen(pl): pl = np.asarray(pl, float); return np.linalg.norm(np.diff(pl, axis=0), axis=1).sum()
lanes = [np.asarray(pl, float) for pl in roads["tan5"] if len(pl) > 1 and plen(pl) > 15]
foot = [np.asarray(pl, float) for pl in roads["gray1"] if len(pl) > 1 and 6 < plen(pl) < 140]
nature = [np.asarray(pl, float) for pl in roads.get("brown05", []) if len(pl) > 1]
pins = np.vstack([resample(np.asarray(pl, float), 6.0) for pl in roads["gray5"] if len(pl) > 1]); pinsM = f0(pins)
def strokes(pls, step): return [(resample(pl, step), unit(np.gradient(resample(pl, step), axis=0))) for pl in pls if len(resample(pl, step)) > 1]
lane_s = strokes(lanes, 3.0); foot_s = strokes(foot, 3.0)
print(f"{len(lane_s)} lanes, {len(foot_s)} footpaths, {len(pins)} road pins")

# ---------- residual field ----------
def make_field(Xm, res, w, sigma):
    allX = np.vstack([Xm, pinsM]); allR = np.vstack([res, np.zeros_like(pinsM)]); allW = np.concatenate([w, np.full(len(pinsM), 1.0)]); tree = cKDTree(allX)
    def corr(Mq):
        out = np.zeros_like(Mq)
        for s in range(0, len(Mq), 500):
            q = Mq[s:s + 500]
            for rr, ids in enumerate(tree.query_ball_point(q, 3 * sigma)):
                if not ids: continue
                ids = np.array(ids); d2 = ((allX[ids] - q[rr]) ** 2).sum(1); ww = np.exp(-d2 / (2 * sigma ** 2)) * allW[ids]; sw = ww.sum()
                out[s + rr] = (ww @ allR[ids]) / max(sw, 1e-9) * (sw / (sw + 0.3))
        return out
    return corr
corr = lambda Mq: np.zeros_like(Mq)
f = lambda X: (lambda Mm: Mm + corr(Mm))(f0(X))
def match(strokes_, fn, gate, ang, w):
    X, Y, W = [], [], []
    for P, Tp in strokes_:
        Pm = fn(P); d, j = skel_tree.query(Pm); Tm = unit(fn(P + Tp * 0.5) - fn(P - Tp * 0.5))
        ok = (d < gate) & (np.abs((Tm * T[j]).sum(1)) > np.cos(np.radians(ang)))
        # a stroke only counts when most of it finds a path: avoids dragging a lane onto a random nearby path fragment
        if ok.mean() < 0.4: continue
        X.append(P[ok]); Y.append(skel[j[ok]]); W.append(np.full(ok.sum(), w))
    return (np.vstack(X), np.vstack(Y), np.concatenate(W)) if X else (np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
for it in range(14):
    gate = max(5.0, 26.0 - 1.5 * it); sigma = max(args.sigma, 60.0 - 3.0 * it)
    Xl, Yl, Wl = match(lane_s, f, gate, 35, 1.0); Xf, Yf, Wf = match(foot_s, f, gate * 0.8, 40, 0.5)
    X = np.vstack([Xl, Xf]); Y = np.vstack([Yl, Yf]); W = np.concatenate([Wl, Wf]); X0 = f0(X)
    corr = make_field(X0, Y - X0, W, sigma)
def report(name, fn):
    for label, ss in (("lanes", lane_s), ("footpaths", foot_s)):
        P = np.vstack([s[0] for s in ss]); d, _ = skel_tree.query(fn(P))
        print(f"{name} {label:9s}: to nearest path med {np.median(d):.1f} m, p75 {np.percentile(d, 75):.1f}, p90 {np.percentile(d, 90):.1f}, <4 m {(d < 4).mean():.0%}")
report("before", f0); report("after ", f)
print(f"road pins moved: max {np.linalg.norm(f(pins) - pinsM, axis=1).max():.2f} m")
R = np.array([[v[2], v[3]] for v in rooms.values()])
def jac_dets(fn, X, h=0.5):
    fx = (fn(X + [h, 0]) - fn(X - [h, 0])) / (2 * h); fy = (fn(X + [0, h]) - fn(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
G24 = np.array([(x, y) for y in np.arange(0, 793, 24) for x in np.arange(0, 1009, 24)]); dg = jac_dets(f, G24); dets = jac_dets(f, R)
print(f"jacobian det: rooms {dets.min():.2f}..{dets.max():.2f}, grid folds {(dg >= 0).sum()}")
shift = np.linalg.norm(f(R) - f0(R), axis=1); print(f"room moves: med {np.median(shift):.1f} m, max {shift.max():.1f} m")

# ---------- overlays ----------
from PIL import Image, ImageDraw, ImageFont
im = Image.fromarray(img); dr = ImageDraw.Draw(im); font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
for p in M.m2px(skel)[::2]: dr.point((p[0], p[1]), fill=(255, 0, 255))
for pls, col in ((lanes, (255, 140, 0)), (foot, (0, 200, 255)), (nature, (160, 90, 20))):
    for pl in pls: dr.line([tuple(q) for q in M.m2px(f(resample(pl, 2.0)))], fill=col, width=2)
for q0, q1, name in zip(M.m2px(f0(R)), M.m2px(f(R)), rooms.keys()):
    dr.line([tuple(q0), tuple(q1)], fill=(255, 60, 60), width=1); dr.ellipse([q1[0] - 3, q1[1] - 3, q1[0] + 3, q1[1] + 3], outline=(80, 255, 80), width=2); dr.text((q1[0] + 4, q1[1] - 6), name, fill=(255, 255, 0), font=font)
for name, (x0, x1, y0, y1) in {"north": (-560, -130, 150, 420), "south": (-340, -80, -300, -50), "farsouth": (-330, -100, -440, -230), "lodge": (-480, -200, -120, 120)}.items():
    c = M.m2px(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)); im.crop((*c.min(0).astype(int), *c.max(0).astype(int))).save(f"/tmp/lane_{name}.png")
print("overlays: /tmp/lane_*.png")
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
    for rr in range(len(Mq)):
        A = np.hstack([np.ones((k, 1)), GM[ii[rr]]]); w = 1 / (d[rr] + 5.0)
        th = np.linalg.lstsq(A * w[:, None], G[ii[rr]] * w[:, None], rcond=None)[0]; out[rr] = np.array([1, Mq[rr, 0], Mq[rr, 1]]) @ th
    return out
IGP = inv_local(IG); RM = f(R)
for (k, v), ll in zip(rooms.items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
exact = {"Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9), "Club Lake": (317.6, 111.8), "Duck Pond": (-266.2, 12.4),
         "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Lodge (Main Lobby)": (-330.0, 60.0)}
for k, v in landmarks.items():
    mm = np.array([exact[k]], float) if k in exact else f(np.array([[v[2], v[3]]])); ll = m2ll(mm)[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
d = DATA
d["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
d["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(d, open(root / "data/appdata.json", "w"), separators=(",", ":"))
err = np.linalg.norm(grid_lookup(d["inv"], RM) - R, axis=1); print(f"wrote data/appdata.json; inverse roundtrip at rooms median {np.median(err):.2f} pt max {err.max():.2f} pt")
