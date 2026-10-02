#!/usr/bin/env python3
"""Free-form registration of the resort drawing to paths and buildings seen in the aerials.

One smooth displacement field over PDF space (B-spline-like bilinear grid, optimised with PyTorch) moves
every stroke and room box together, so the drawing's topology is preserved while its geometry is pulled
onto reality:
  gray5 (main roads)          -> asphalt            (distance field droad)
  tan5 / gray1 / brown05      -> any detected path  (dpath)   lanes, footpaths, nature path
  room boxes                  -> building footprint (dbld)
  surveyed anchors            -> exact points
Starts from the blended georeferencing (data/appdata.blend.json) and writes data/appdata.ffd.json.

  python3 tools/register_ffd.py [--iters 600] [--spacing 24]
"""
import sys, json, pathlib, argparse, math, time
import numpy as np, torch, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aerial import root, m2ll, ll2m, LAT0, LON0, K_LAT, K_LON

ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=600); ap.add_argument("--spacing", type=float, default=24.0)
ap.add_argument("--bend", type=float, default=0.1); ap.add_argument("--elastic", type=float, default=0.01); ap.add_argument("--base", default="data/appdata.blend.json"); ap.add_argument("--room-targets", default=None, help="appdata json whose room lat/lon are treated as exact targets"); ap.add_argument("--out", default="ffd"); ap.add_argument("--anchor-w", type=float, default=0.002); args = ap.parse_args()
t0 = time.time()
BASE = json.load(open(root / args.base)); rooms = BASE["rooms"]; landmarks = BASE["landmarks"]; roads = json.load(open(root / "data/roads_pdf.json"))
A = np.load(root / "data/aerial_paths.npz"); dpath, droad, dbld = A["dpath"], A["droad"], A["dbld"]; tx0, ty0, ts, z = [int(v) for v in A["geo"]]; res = float(A["res"])
H, W = dpath.shape; n_tiles = 2 ** z

def grid_lookup(g, X):
    v = np.asarray(g["v"]).reshape(g["ny"], g["nx"], 2); fx = (X[:, 0] - g["x0"]) / g["step"]; fy = (X[:, 1] - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2); tx = (fx - i)[:, None]; ty = (fy - j)[:, None]
    return (1 - tx) * (1 - ty) * v[j, i] + tx * (1 - ty) * v[j, i + 1] + (1 - tx) * ty * v[j + 1, i] + tx * ty * v[j + 1, i + 1]
base = lambda X: grid_lookup(BASE["fwd"], X)
def resample(pl, step):
    pl = np.asarray(pl, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pl, axis=0), axis=1))]
    if d[-1] < step: return pl
    s = np.arange(0, d[-1] + 1e-9, step); return np.c_[np.interp(s, d, pl[:, 0]), np.interp(s, d, pl[:, 1])]
plen = lambda pl: np.linalg.norm(np.diff(np.asarray(pl, float), axis=0), axis=1).sum()

# ---------- sample points: (pdf xy, field id, weight) ----------
FIELDS = {"road": 0, "path": 1, "bld": 2}
pts, fid, wts = [], [], []
def add(P, f, w): pts.append(P); fid.append(np.full(len(P), FIELDS[f])); wts.append(np.full(len(P), w))
for pl in roads["gray5"]:
    if len(pl) > 1: add(resample(pl, 3.0), "road", 1.0)
for pl in roads["tan5"]:
    if len(pl) > 1 and plen(pl) > 10: add(resample(pl, 3.0), "path", 1.0)
for pl in roads["gray1"]:
    if len(pl) > 1 and 6 < plen(pl) < 140: add(resample(pl, 3.0), "path", 0.5)
for pl in roads.get("brown05", []):
    if len(pl) > 1 and plen(pl) > 10: add(resample(pl, 3.0), "path", 0.7)
R = np.array([[v[2], v[3]] for v in rooms.values()]); rnames = list(rooms.keys())
add(R, "bld", 2.0)
P = np.vstack(pts); FID = np.concatenate(fid); WT = np.concatenate(wts)
anchors = [((157, 117), (-587.5, 305.3)), ((240, 426), (-390.6, 40.9)), ((370, 450), (-266.2, 12.4)), ((728, 580), (85.7, -164.0)), ((831, 303), (223.7, 213.9)),
           ((854, 297), (259.2, 235.9)), ((875, 365), (317.6, 111.8)), ((96, 74), (-705.7, 415.5)), ((273, 447), (-348.6, 25.9)), ((795, 330), (165.0, 180.0))]
AP = np.array([a[0] for a in anchors], float); AM = np.array([a[1] for a in anchors], float)
if args.room_targets:
    RT = json.load(open(root / args.room_targets))["rooms"]
    ids = [k for k in rnames if k in RT]
    AP = np.vstack([AP, R[[rnames.index(k) for k in ids]]]); AM = np.vstack([AM, ll2m([RT[k][0] for k in ids], [RT[k][1] for k in ids])])
    print(f"{len(ids)} room targets added as anchors")
print(f"{len(P)} sample points: road {np.sum(FID == 0)}, path {np.sum(FID == 1)}, rooms {np.sum(FID == 2)}; {len(AP)} anchors")

# ---------- torch model ----------
dev = torch.device("cpu")
fields_raw = torch.tensor(np.stack([droad, dpath, dbld]), dtype=torch.float32)[None]      # 1 x 3 x H x W
fields_blur = torch.tensor(np.stack([cv2.GaussianBlur(a, (0, 0), 24) for a in (droad, dpath, dbld)]), dtype=torch.float32)[None]   # ~6 m blur for the coarse stage
fields = fields_raw
def m2px_t(m):
    """local metres -> mosaic pixel coords (differentiable)."""
    lat = LAT0 + m[:, 1] / K_LAT; lon = LON0 + m[:, 0] / K_LON
    x = (lon + 180) / 360 * n_tiles; latr = lat * math.pi / 180
    y = (1 - torch.log(torch.tan(latr) + 1 / torch.cos(latr)) / math.pi) / 2 * n_tiles
    return torch.stack([(x - tx0) * ts, (y - ty0) * ts], 1)
def sample_field(px, which):
    """bilinear sample of distance field `which` at pixel coords px (N x 2)."""
    gx = px[:, 0] / (W - 1) * 2 - 1; gy = px[:, 1] / (H - 1) * 2 - 1
    g = torch.stack([gx, gy], 1)[None, :, None, :]                                        # 1 x N x 1 x 2
    out = torch.nn.functional.grid_sample(fields, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0]   # 3 x N
    return out[which, torch.arange(len(px))]
class FFD(torch.nn.Module):
    def __init__(s, spacing):
        super().__init__(); s.sp = spacing; s.nx = int(math.ceil(1008 / spacing)) + 3; s.ny = int(math.ceil(792 / spacing)) + 3
        s.D = torch.nn.Parameter(torch.zeros(1, 2, s.ny, s.nx))                            # displacement (metres) on a PDF-space grid, 1 cell margin
    def disp(s, X):   # X: N x 2 pdf points -> N x 2 metres displacement
        gx = (X[:, 0] / s.sp + 1) / (s.nx - 1) * 2 - 1; gy = (X[:, 1] / s.sp + 1) / (s.ny - 1) * 2 - 1
        g = torch.stack([gx, gy], 1)[None, :, None, :]
        return torch.nn.functional.grid_sample(s.D, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0].T
    def fold(s, base_fn):
        """penalty on cells of a fine (6 pt) PDF grid whose total-map Jacobian determinant rises above -0.35 (det is ~-1.1 for a healthy map)."""
        if not hasattr(s, "_fg"):
            g = np.array([(x, y) for y in np.arange(0, 793, 6.0) for x in np.arange(0, 1009, 6.0)]); s._fg = torch.tensor(g, dtype=torch.float32)
            s._fgb = torch.tensor(base_fn(g), dtype=torch.float32); s._fny, s._fnx = len(np.arange(0, 793, 6.0)), len(np.arange(0, 1009, 6.0))
        T = (s._fgb + s.disp(s._fg)).reshape(s._fny, s._fnx, 2)
        dx = (T[:, 1:, :] - T[:, :-1, :])[:-1] / 6.0; dy = (T[1:, :, :] - T[:-1, :, :])[:, :-1] / 6.0
        det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        return (torch.relu(det + 0.35) ** 2).sum()
    def elastic(s):
        D = s.D; return ((D[:, :, :, 1:] - D[:, :, :, :-1]) ** 2).mean() + ((D[:, :, 1:, :] - D[:, :, :-1, :]) ** 2).mean()
    def bending(s):
        D = s.D; dxx = D[:, :, :, 2:] - 2 * D[:, :, :, 1:-1] + D[:, :, :, :-2]; dyy = D[:, :, 2:, :] - 2 * D[:, :, 1:-1, :] + D[:, :, :-2, :]
        dxy = D[:, :, 1:, 1:] - D[:, :, 1:, :-1] - D[:, :, :-1, 1:] + D[:, :, :-1, :-1]
        return (dxx ** 2).mean() + (dyy ** 2).mean() + 2 * (dxy ** 2).mean()
Pt = torch.tensor(P, dtype=torch.float32); BM = torch.tensor(base(P), dtype=torch.float32); FIDt = torch.tensor(FID); WTt = torch.tensor(WT, dtype=torch.float32)
APt = torch.tensor(AP, dtype=torch.float32); ABM = torch.tensor(base(AP), dtype=torch.float32); AMt = torch.tensor(AM, dtype=torch.float32)
# points that start more than 45 m from any target cannot be helped and only add a constant: drop them from the data term
d0 = sample_field(m2px_t(BM), FIDt).detach(); use = d0 < 45; print(f"using {use.sum().item()} of {len(P)} points (others start >45 m from any target)")
def run(model, iters, lr, bend, blur):
    global fields
    fields = fields_blur if blur else fields_raw
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        opt.zero_grad()
        M = BM + model.disp(Pt); d = sample_field(m2px_t(M), FIDt); d = d[use]
        data = (WTt[use] * torch.nn.functional.huber_loss(d, torch.zeros_like(d), delta=3.0, reduction="none")).sum() / WTt[use].sum()
        anc = ((ABM + model.disp(APt) - AMt) ** 2).sum(1).mean()
        loss = data + args.anchor_w * anc + bend * model.bending() + args.elastic * model.elastic() + 50.0 * model.fold(base)
        loss.backward(); opt.step()
        if it % 100 == 0 or it == iters - 1: print(f"  it {it:4d} data {data.item():6.2f} med dist {d.median().item():4.1f} m anchors rms {anc.sqrt().item():5.1f} m bend {model.bending().item():.3f} fold {model.fold(base).item():.2f} ({time.time() - t0:.0f}s)")
    return model
# coarse to fine
m1 = run(FFD(args.spacing * 3), args.iters // 2, 1.0, args.bend, blur=True)
m2 = FFD(args.spacing)
with torch.no_grad():   # initialise fine grid from coarse
    cx = torch.arange(m2.nx) * args.spacing - args.spacing; cy = torch.arange(m2.ny) * args.spacing - args.spacing
    gX, gY = torch.meshgrid(cx, cy, indexing="xy"); G2 = torch.stack([gX.flatten(), gY.flatten()], 1).float()
    m2.D.copy_(m1.disp(G2).T.reshape(1, 2, m2.ny, m2.nx))
model = run(m2, args.iters, 0.3, args.bend, blur=False)

# ---------- evaluate ----------
with torch.no_grad():
    def f(X):
        Xt = torch.tensor(np.asarray(X, float), dtype=torch.float32); return (torch.tensor(base(np.asarray(X, float)), dtype=torch.float32) + model.disp(Xt)).detach().numpy()
def px_of(Mm): ll = m2ll(Mm); x = (ll[:, 1] + 180) / 360 * n_tiles; lat = np.radians(ll[:, 0]); y = (1 - np.log(np.tan(lat) + 1 / np.cos(lat)) / math.pi) / 2 * n_tiles; return np.c_[(x - tx0) * ts, (y - ty0) * ts]
def sample_np(field, Mm):
    px = px_of(Mm); xi = np.clip(px[:, 0].round().astype(int), 0, W - 1); yi = np.clip(px[:, 1].round().astype(int), 0, H - 1); return field[yi, xi]
for name, fn in (("before", base), ("after ", f)):
    out = []
    for fname, k in FIELDS.items():
        sel = FID == k; d = sample_np([droad, dpath, dbld][k], fn(P[sel])); out.append(f"{fname}: med {np.median(d):4.1f} p90 {np.percentile(d, 90):4.1f} <4m {(d < 4).mean():3.0%}")
    print(name, " | ".join(out), f"| anchors rms {np.sqrt(((fn(AP) - AM) ** 2).sum(1).mean()):.1f} m")
for grp in "2356":
    sel = np.array([k[0] == grp for k in rnames]); d0 = sample_np(dbld, base(R[sel])); d1 = sample_np(dbld, f(R[sel]))
    print(f"  rooms {grp}xx: within 8 m of a building {np.mean(d0 < 8):3.0%} -> {np.mean(d1 < 8):3.0%}, median {np.median(d0):4.1f} -> {np.median(d1):4.1f} m")
def jac_dets(fn, X, h=0.5):
    fx = (fn(X + [h, 0]) - fn(X - [h, 0])) / (2 * h); fy = (fn(X + [0, h]) - fn(X - [0, h])) / (2 * h); return fx[:, 0] * fy[:, 1] - fx[:, 1] * fy[:, 0]
G12 = np.array([(x, y) for y in np.arange(0, 793, 12) for x in np.arange(0, 1009, 12)]); dg = jac_dets(f, G12)
print(f"jacobian det on 12 pt grid {dg.min():.2f}..{dg.max():.2f}, folds {(dg >= 0).sum()}; max displacement {np.linalg.norm(f(G12) - base(G12), axis=1).max():.1f} m")

# ---------- write ----------
step_f = 12.0; fgx = np.arange(0, 1009, step_f); fgy = np.arange(0, 793, step_f)
G = np.array([(x, y) for y in fgy for x in fgx]); GM = f(G)
mn = GM.min(0) - 40; mx = GM.max(0) + 40; step = 10.0
ix = np.arange(mn[0], mx[0] + step, step); iy = np.arange(mn[1], mx[1] + step, step); IG = np.array([(x, y) for y in iy for x in ix])
gm_tree = cKDTree(GM)
def inv_local(Mq, k=12):
    d, ii = gm_tree.query(Mq, k); out = np.zeros((len(Mq), 2))
    for r in range(len(Mq)):
        Am = np.hstack([np.ones((k, 1)), GM[ii[r]]]); w = 1 / (d[r] + 5.0)
        th = np.linalg.lstsq(Am * w[:, None], G[ii[r]] * w[:, None], rcond=None)[0]; out[r] = np.array([1, Mq[r, 0], Mq[r, 1]]) @ th
    return out
IGP = inv_local(IG)
D = json.loads(json.dumps(BASE)); RM = f(R)
for (k, v), ll in zip(D["rooms"].items(), m2ll(RM)): v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
exact = {"Lodge Pool": (-390.6, 40.9), "Spa Pool": (-587.5, 305.3), "Golf Clubhouse": (259.2, 235.9), "Club Lake": (317.6, 111.8), "Duck Pond": (-266.2, 12.4),
         "Tennis Center": (165.0, 180.0), "Main Gate": (-705.7, 415.5), "Lodge (Main Lobby)": (-330.0, 60.0)}
for k, v in D["landmarks"].items():
    mm = np.array([exact[k]], float) if k in exact else f(np.array([[v[2], v[3]]])); ll = m2ll(mm)[0]; v[0] = round(float(ll[0]), 6); v[1] = round(float(ll[1]), 6)
D["fwd"] = {"x0": 0, "y0": 0, "step": step_f, "nx": len(fgx), "ny": len(fgy), "v": GM.round(2).tolist()}
D["inv"] = {"x0": float(ix[0]), "y0": float(iy[0]), "step": step, "nx": len(ix), "ny": len(iy), "v": IGP.round(2).tolist()}
json.dump(D, open(root / f"data/appdata.{args.out}.json", "w"), separators=(",", ":"))
err = np.linalg.norm(grid_lookup(D["inv"], RM) - R, axis=1); print(f"wrote data/appdata.{args.out}.json; inverse roundtrip at rooms median {np.median(err):.2f} pt max {err.max():.2f} pt ({time.time() - t0:.0f}s)")
