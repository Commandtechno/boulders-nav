"""Shared helpers for working with the cached Scottsdale aerial tiles (verify/tiles) in the app's local metre frame.

Mosaic pixel space: Web Mercator at zoom Z (default 20, ~0.124 m/px here), origin at the tile range's top-left.
"""
import json, math, pathlib
import numpy as np
from PIL import Image

root = pathlib.Path(__file__).resolve().parent.parent
DATA = json.load(open(root / "data/appdata.json"))
LAT0, LON0, K_LAT, K_LON = DATA["lat0"], DATA["lon0"], DATA["k_lat"], DATA["k_lon"]
R = 6378137.0

def ll2m(lat, lon): return np.c_[(np.asarray(lon) - LON0) * K_LON, (np.asarray(lat) - LAT0) * K_LAT]
def m2ll(m): m = np.asarray(m); return np.c_[LAT0 + m[..., 1] / K_LAT, LON0 + m[..., 0] / K_LON]

class Mosaic:
    def __init__(self, z=20, down=1):
        tdir = root / "verify/tiles" / str(z)
        xs = sorted(int(p.name) for p in tdir.iterdir()); ys = sorted({int(q.stem) for p in tdir.iterdir() for q in p.glob("*.jpg")})
        self.z, self.down, self.tx0, self.ty0 = z, down, xs[0], ys[0]
        self.n = 2 ** z; ts = 256 // down
        self.img = np.zeros(((ys[-1] - ys[0] + 1) * ts, (xs[-1] - xs[0] + 1) * ts, 3), np.uint8)
        for x in xs:
            for y in ys:
                p = tdir / str(x) / f"{y}.jpg"
                if not p.exists(): continue
                im = Image.open(p); im = im if down == 1 else im.resize((ts, ts), Image.LANCZOS)
                self.img[(y - ys[0]) * ts:(y - ys[0] + 1) * ts, (x - xs[0]) * ts:(x - xs[0] + 1) * ts] = np.asarray(im)
        self.ts = ts
        # ground resolution (m/px) at the resort latitude
        self.res = 2 * math.pi * R * math.cos(math.radians(LAT0)) / (self.n * ts)

    def ll2px(self, lat, lon):
        lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        x = (lon + 180) / 360 * self.n; y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * self.n
        return np.c_[(x - self.tx0) * self.ts, (y - self.ty0) * self.ts]
    def px2ll(self, px):
        px = np.asarray(px, float); x = px[..., 0] / self.ts + self.tx0; y = px[..., 1] / self.ts + self.ty0
        lon = x / self.n * 360 - 180; lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * y / self.n))))
        return np.c_[lat, lon]
    def m2px(self, m): ll = m2ll(m); return self.ll2px(ll[:, 0], ll[:, 1])
    def px2m(self, px): ll = self.px2ll(px); return ll2m(ll[:, 0], ll[:, 1])
