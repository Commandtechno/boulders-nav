# Boulders Night Nav

Walking directions to any room at The Boulders Resort & Spa (Scottsdale/Carefree, AZ), built from the resort's printed property map. Routes follow the resort's roads and footpaths, with turn-by-turn prompts, a map that turns to face your heading, voice guidance and automatic rerouting.

Everything lives in one file: `index.html` (about 1.6 MB, 0.5 MB over the wire: the resort map as vector SVG, coordinates, path network and the ABC Diatype webfonts embedded). Styling follows the Runpod brand kit: purple `#5D29F0` on near-black, ABC Diatype, no borders. No server, no accounts, no tracking. Home room and GPS pins stay in the phone's browser storage.

## Deploy

Every push to `main` rebuilds `index.html` and publishes it to GitHub Pages through `.github/workflows/pages.yml`. Only `index.html` is published; the PDF and source data stay in the repo.

Open the Pages URL on your phone, then Share → Add to Home Screen so it opens full screen. On iPhone, tap **Enable compass** the first time.

## Using the app

- **Go**: type a room number (200–285, 300–373, 501–523, 601–667) or pick a place, then Go. **Save as home** remembers your room; **Take me home** is the big button.
- **Pin this room here**: stand at your door and tap it. The app averages GPS for 6 seconds and ends every route to that room at the exact spot. Do it once in daylight.
- **Navigate**: route on the resort map, instruction banner ("In 120 ft, turn left", then the following turn), remaining distance, minutes and arrival time. The map follows you heading-up; drag to look around and tap **Re-center** to resume. Voice prompts call out each turn.
- **Arrow**: a big arrow pointing along the path to the next waypoint, with Compass (rotates with the phone) or North-up modes.
- **Heading** comes from the phone's compass when it has one, otherwise from the gyroscope aligned to north by your GPS track while walking, otherwise from your direction of travel. iPhones ask for motion access on your first tap.
- **Settings**: feet/metres, voice, keep-awake, and a *simulate position* mode (tap the map to place yourself) for testing indoors.

## Accuracy

Casita, hacienda and villa positions come from the illustrated map warped onto real-world coordinates by matching its road network to OpenStreetMap. Typical error is 10–20 m, so expect to land within a building or two and then read the numbers on the doors; pinning your own room removes that error. Landmarks (Lodge, Golf Clubhouse, pools, ponds, tennis courts) were checked against surveyed coordinates.

## Verifying the data

`verify/` (published at `/verify/`) draws everything the app knows over the City of Scottsdale's 2026 aerial imagery (about 10 cm per pixel): the warped resort map with an opacity slider, all 217 rooms, 25 landmarks, the 3,332 walking-path edges and the links from each room to the path network. Drag a room label to its real door and **Copy JSON** to export corrections (also kept in the browser's storage). Tap anywhere for lat/lon and PDF-point coordinates. Esri World Imagery is available as a second basemap.

Regenerate the overlays with the pipeline venv (PyMuPDF, numpy, Pillow):

```sh
python3 tools/warp_map.py          # verify/map.webp: the PDF warped onto Web Mercator through the app's inverse grid
python3 tools/fetch_aerials.py     # verify/tiles: Scottsdale aerials cached as XYZ tiles (their server is too slow to stream)
```

## Rebuild locally

```sh
python3 build.py   # injects data/appdata.json, data/map.svg and data/fonts into src/index.template.html
```

`data/appdata.json` holds every room's lat/lon, the forward/inverse lookup grids that convert between map pixels and coordinates, and the walking graph (3,275 nodes, 12.8 km of paths).
