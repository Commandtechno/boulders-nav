#!/usr/bin/env python3
"""Assemble the single-file app: inject room/landmark data, the vector map (SVG) and the webfonts
into src/index.template.html and write index.html."""
import base64, json, pathlib, re, shutil

root = pathlib.Path(__file__).parent
tpl = (root / "src/index.template.html").read_text()
data = json.load(open(root / "data/appdata.json"))
svg = (root / "data/map.svg").read_text()
svg_inner = re.sub(r"^.*?<svg[^>]*>", "", svg, count=1, flags=re.S).rsplit("</svg>", 1)[0]   # keep only the map content
font = lambda n: "data:font/woff2;base64," + base64.b64encode((root / f"data/fonts/ABCDiatype-{n}.woff2").read_bytes()).decode()

# the JSON sits inside a <script type="application/json"> block, so escape any "</" sequences
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
out = (tpl.replace("__APPDATA__", payload).replace("__MAPSVG__", svg_inner)
       .replace("__FONT_REGULAR__", font("Regular")).replace("__FONT_MEDIUM__", font("Medium")).replace("__FONT_BOLD__", font("Bold")))
(root / "index.html").write_text(out)
print(f"wrote index.html ({len(out)/1e6:.2f} MB), {len(data['rooms'])} rooms, {len(data['landmarks'])} landmarks")

# verification page (verify/index.html): the same data over aerial imagery; map.webp comes from tools/warp_map.py
vtpl = (root / "src/verify.template.html").read_text()
tiles = (root / "verify/tiles.json").read_text()      # from tools/fetch_aerials.py
# georeferencing versions the page can switch between; each needs data/appdata.<id>.json and verify/map_<id>.webp
# (tools/warp_map.py --data data/appdata.<id>.json --name map_<id>). The first entry is what the app ships.
VERSIONS = [
    {"id": "affine", "label": "Raw: one affine transform", "note": "The drawing exactly as drawn, only rotated, scaled and placed. Landmarks miss by about 50 m."},
    {"id": "blend", "label": "Blend (original in casitas, roads elsewhere)", "note": "Original warp inside the casita areas, named-road warp for roads, villas and clubhouse."},
    {"id": "v1", "label": "Original warp (OSM roads near Lodge)", "note": "The first georeferencing: TPS fitted to OpenStreetMap roads, best near the Lodge."},
    {"id": "v2", "label": "Named-road warp", "note": "Each named road matched to the same road in OpenStreetMap. Roads within 2 m, casitas displaced."},
]
versions = [v for v in VERSIONS if (root / f"data/appdata.{v['id']}.json").exists() and (root / f"verify/map_{v['id']}.webp").exists()]
for v in versions:
    shutil.copy(root / f"data/appdata.{v['id']}.json", root / f"verify/data_{v['id']}.json")
(root / "verify/index.html").write_text(vtpl.replace("__VERSIONS__", json.dumps(versions)).replace("__TILESINFO__", tiles))
print(f"wrote verify/index.html with versions {[v['id'] for v in versions]}")
