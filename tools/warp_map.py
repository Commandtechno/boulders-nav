#!/usr/bin/env python3
"""Warp the resort PDF map onto Web Mercator using the app's own inverse lookup grid, for the
verification overlay (verify/map.webp + verify/map_bounds.json).

Run with the pipeline venv (needs PyMuPDF, numpy, Pillow):
  python3 tools/warp_map.py [--res 0.25]
"""
import argparse, json, math, pathlib, sys
import numpy as np
import fitz
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--res", type=float, default=0.25, help="output ground resolution in metres/pixel (at the resort's latitude)")
ap.add_argument("--scale", type=float, default=4.0, help="PDF rasterisation scale (points -> pixels)")
args = ap.parse_args()

data = json.load(open(root / "data/appdata.json"))
lat0, lon0, k_lat, k_lon = data["lat0"], data["lon0"], data["k_lat"], data["k_lon"]
R = 6378137.0

def ll2merc(lat, lon):
    return lon * math.pi / 180 * R, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R

def merc2ll(x, y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - math.pi / 2), np.degrees(x / R)

def grid_lookup(g, x, y):
    """Vectorised copy of the app's bilinear gridLookup (same edge clamping)."""
    v = np.asarray(g["v"], dtype=np.float64).reshape(g["ny"], g["nx"], 2)
    fx = (x - g["x0"]) / g["step"]; fy = (y - g["y0"]) / g["step"]
    i = np.clip(np.floor(fx).astype(int), 0, g["nx"] - 2); j = np.clip(np.floor(fy).astype(int), 0, g["ny"] - 2)
    tx = fx - i; ty = fy - j
    a = v[j, i]; b = v[j, i + 1]; c = v[j + 1, i]; d = v[j + 1, i + 1]
    w = lambda k: (1 - tx) * (1 - ty) * a[..., k] + tx * (1 - ty) * b[..., k] + (1 - tx) * ty * c[..., k] + tx * ty * d[..., k]
    return w(0), w(1)

# 1. rasterise the PDF page
pdf = next(root.glob("*.pdf"))
page = fitz.open(pdf)[0]
pix = page.get_pixmap(matrix=fitz.Matrix(args.scale, args.scale), alpha=False)
src = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
print(f"rendered {pdf.name}: {pix.width}x{pix.height}")

# 2. output extent: the PDF's four edges pushed through the forward grid, in metres, then to lat/lon
fwd = data["fwd"]
edge = np.concatenate([np.stack([np.linspace(0, data["pdf_w"], 200), np.full(200, y)], 1) for y in (0, data["pdf_h"])] +
                      [np.stack([np.full(200, x), np.linspace(0, data["pdf_h"], 200)], 1) for x in (0, data["pdf_w"])])
mx, my = grid_lookup(fwd, edge[:, 0], edge[:, 1])
lats = lat0 + my / k_lat; lons = lon0 + mx / k_lon
x0, y0 = ll2merc(lats.min(), lons.min()); x1, y1 = ll2merc(lats.max(), lons.max())
merc_per_m = 1 / math.cos(math.radians(lat0))              # Mercator metres per ground metre here
W = int(math.ceil((x1 - x0) / (args.res * merc_per_m))); H = int(math.ceil((y1 - y0) / (args.res * merc_per_m)))
print(f"output {W}x{H} px, {args.res} m/px, bounds lat {lats.min():.6f}..{lats.max():.6f} lon {lons.min():.6f}..{lons.max():.6f}")

# 3. inverse-map every output pixel to PDF coordinates (chunked by rows to bound memory)
out = np.zeros((H, W, 4), dtype=np.uint8)
xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
for r0 in range(0, H, 256):
    r1 = min(H, r0 + 256)
    ys = y1 - (np.arange(r0, r1) + 0.5) * (y1 - y0) / H          # row 0 is north
    X, Y = np.meshgrid(xs, ys)
    lat, lon = merc2ll(X, Y)
    pmx = (lon - lon0) * k_lon; pmy = (lat - lat0) * k_lat
    px, py = grid_lookup(data["inv"], pmx, pmy)
    sx = px * args.scale; sy = py * args.scale
    # mask where the lookup leaves the inverse grid's trusted area: the forward warp of the found PDF point
    # must land back near the queried position (the grids extrapolate linearly from their edge cells otherwise)
    bx, by = grid_lookup(fwd, px, py)
    consistent = np.hypot(bx - pmx, by - pmy) < 3.0
    inside = (sx >= 0) & (sx < src.shape[1] - 1) & (sy >= 0) & (sy < src.shape[0] - 1) & consistent
    sx = np.clip(sx, 0, src.shape[1] - 1.001); sy = np.clip(sy, 0, src.shape[0] - 1.001)
    ix = np.floor(sx).astype(int); iy = np.floor(sy).astype(int); tx = (sx - ix)[..., None]; ty = (sy - iy)[..., None]
    s = (src[iy, ix] * (1 - tx) * (1 - ty) + src[iy, ix + 1] * tx * (1 - ty) + src[iy + 1, ix] * (1 - tx) * ty + src[iy + 1, ix + 1] * tx * ty)
    out[r0:r1, :, :3] = s.astype(np.uint8)
    out[r0:r1, :, 3] = np.where(inside, 255, 0)

outdir = root / "verify"; outdir.mkdir(exist_ok=True)
Image.fromarray(out, "RGBA").save(outdir / "map.webp", quality=82, method=6)
json.dump({"south": float(lats.min()), "north": float(lats.max()), "west": float(lons.min()), "east": float(lons.max()),
           "width": W, "height": H, "res_m": args.res}, open(outdir / "map_bounds.json", "w"), indent=1)
print(f"wrote verify/map.webp ({(outdir / 'map.webp').stat().st_size / 1e6:.2f} MB) and verify/map_bounds.json")
