#!/usr/bin/env python3
"""Assemble the single-file app: inject room/landmark data and the embedded map image
into src/index.template.html and write index.html."""
import base64, json, pathlib

root = pathlib.Path(__file__).parent
tpl = (root / "src/index.template.html").read_text()
data = json.load(open(root / "data/appdata.json"))
img_b64 = base64.b64encode((root / "data/map200.jpg").read_bytes()).decode()
font = lambda n: "data:font/woff2;base64," + base64.b64encode((root / f"data/fonts/ABCDiatype-{n}.woff2").read_bytes()).decode()

# the JSON sits inside a <script type="application/json"> block, so escape any "</" sequences
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
out = (tpl.replace("__APPDATA__", payload).replace("__MAPIMG__", "data:image/jpeg;base64," + img_b64)
       .replace("__FONT_REGULAR__", font("Regular")).replace("__FONT_MEDIUM__", font("Medium")).replace("__FONT_BOLD__", font("Bold")))
(root / "index.html").write_text(out)
print(f"wrote index.html ({len(out)/1e6:.2f} MB), {len(data['rooms'])} rooms, {len(data['landmarks'])} landmarks")
