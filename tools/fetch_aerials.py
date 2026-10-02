#!/usr/bin/env python3
"""Cache the City of Scottsdale aerial imagery over the resort as standard XYZ tiles (verify/tiles/z/x/y.jpg).

The city's MapServer renders each export on the fly from MrSID mosaics, which takes 10+ seconds per request,
so the verification page cannot stream it live. Instead this pulls a handful of 4096 px exports once, cuts
them into 256 px Web Mercator tiles at the native zoom and builds the lower zooms by downsampling.

  python3 tools/fetch_aerials.py [--year 2026] [--zmax 20] [--zmin 15]
"""
import argparse, io, json, math, pathlib, sys, time, urllib.request
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, default=2026)
ap.add_argument("--zmax", type=int, default=20)
ap.add_argument("--zmin", type=int, default=15)
ap.add_argument("--pad", type=float, default=0.0008, help="degrees of padding around the warped map bounds")
args = ap.parse_args()

R = 6378137.0; W = 2 * math.pi * R
mb = json.load(open(root / "verify/map_bounds.json"))
south, north, west, east = mb["south"] - args.pad, mb["north"] + args.pad, mb["west"] - args.pad, mb["east"] + args.pad

def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
def merc(tx, ty, z):
    n = 2 ** z
    return -W / 2 + tx / n * W, W / 2 - ty / n * W

z = args.zmax
x0, y0 = tile_xy(north, west, z); x1, y1 = tile_xy(south, east, z)
tx0, ty0, tx1, ty1 = int(x0), int(y0), int(math.ceil(x1)), int(math.ceil(y1))
print(f"z{z}: tiles x {tx0}..{tx1 - 1}, y {ty0}..{ty1 - 1} = {(tx1 - tx0) * (ty1 - ty0)} tiles")

out = root / "verify/tiles"
BLOCK = 16   # 16 x 16 tiles = 4096 px, the server's maximum export size
for bx in range(tx0, tx1, BLOCK):
    for by in range(ty0, ty1, BLOCK):
        ex, ey = min(bx + BLOCK, tx1), min(by + BLOCK, ty1)
        if all((out / str(z) / str(x) / f"{y}.jpg").exists() for x in range(bx, ex) for y in range(by, ey)):
            continue
        ax, ay = merc(bx, by, z); cx, cy = merc(ex, ey, z)
        w, h = (ex - bx) * 256, (ey - by) * 256
        url = (f"https://maps.scottsdaleaz.gov/arcgis/rest/services/Aerials/Aerials_{args.year}/MapServer/export"
               f"?bbox={ax},{cy},{cx},{ay}&bboxSR=3857&imageSR=3857&size={w},{h}&format=jpg&f=image")
        for attempt in range(4):
            try:
                t = time.time()
                data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (boulders-nav verify tiles)"}), timeout=300).read()
                img = Image.open(io.BytesIO(data)).convert("RGB"); break
            except Exception as e:
                print(f"  retry {attempt + 1}: {e}", file=sys.stderr); time.sleep(5)
        else:
            sys.exit("export failed")
        print(f"block x{bx}-{ex} y{by}-{ey}: {w}x{h} px in {time.time() - t:.0f}s")
        for x in range(bx, ex):
            for y in range(by, ey):
                d = out / str(z) / str(x); d.mkdir(parents=True, exist_ok=True)
                img.crop(((x - bx) * 256, (y - by) * 256, (x - bx + 1) * 256, (y - by + 1) * 256)).save(d / f"{y}.jpg", quality=82, optimize=True)

# lower zooms: each tile is the 2x2 block of children shrunk by half
for zz in range(z - 1, args.zmin - 1, -1):
    kids = {}
    for p in (out / str(zz + 1)).glob("*/*.jpg"):
        kids.setdefault((int(p.parent.name) // 2, int(p.stem) // 2), []).append(p)
    for (x, y), ps in kids.items():
        tile = Image.new("RGB", (512, 512), (20, 20, 20))
        for p in ps:
            cx, cy = int(p.parent.name), int(p.stem)
            tile.paste(Image.open(p), ((cx - 2 * x) * 256, (cy - 2 * y) * 256))
        d = out / str(zz) / str(x); d.mkdir(parents=True, exist_ok=True)
        tile.resize((256, 256), Image.LANCZOS).save(d / f"{y}.jpg", quality=82, optimize=True)
    print(f"z{zz}: {len(kids)} tiles")

total = sum(p.stat().st_size for p in out.rglob("*.jpg")); n = sum(1 for _ in out.rglob("*.jpg"))
json.dump({"year": args.year, "zmin": args.zmin, "zmax": args.zmax, "south": south, "north": north, "west": west, "east": east},
          open(root / "verify/tiles.json", "w"), indent=1)
print(f"{n} tiles, {total / 1e6:.1f} MB")
