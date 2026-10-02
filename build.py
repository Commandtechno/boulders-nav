#!/usr/bin/env python3
"""Assemble the single-file app: inject room/landmark data, the vector map (SVG) and the webfonts
into src/index.template.html and write index.html."""
import base64, json, pathlib, re

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
bounds = (root / "verify/map_bounds.json").read_text()
tiles = (root / "verify/tiles.json").read_text()      # from tools/fetch_aerials.py
(root / "verify/index.html").write_text(vtpl.replace("__APPDATA__", payload).replace("__MAPBOUNDS__", bounds).replace("__TILESINFO__", tiles))
print("wrote verify/index.html")
