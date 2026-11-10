# Great Lakes ice thickness viewer

An animated map of modelled lake ice thickness for Lake Superior, Lake
Michigan, Lake Huron and Michigan inland lakes ≥ 10 ha, 1950/51–2024/25.

**This is model guidance, not a measurement. It is not a safety product and
must never be used to judge whether ice is safe.**

## Run it

The page fetches its layers, so it must be served over HTTP; opening
`index.html` from disk will not work.

```powershell
$env:PYTHONIOENCODING="utf-8"
cd C:\Users\evane\OneDrive\Desktop\CREATE\Map_Art\ice
python python\serve_web.py
```

Then open <http://127.0.0.1:8000/>.

The server binds to `127.0.0.1` only. To view it from a phone or another
computer on the same network, run `python python\serve_web.py --host 0.0.0.0`
and open `http://<this-machine-LAN-IP>:8000/`.

Requirements: a browser with WebGL2 and `DecompressionStream` (Chrome/Edge 80+,
Firefox 113+, Safari 16.4+). The page shows a readable error if either is
missing, or if the GPU's maximum texture size is below 8192 px.

## Controls

| Control | Action |
|---|---|
| Play / Pause, `Space` | start or stop the animation |
| Season slider | jump to any of the 75 seasons |
| Speed | 0.5×, 1×, 2×, 4× (frames advance at 12 fps at 1×) |
| `←` / `→` | step one frame |
| `+` / `−` buttons | zoom in and out around the centre |
| `Fit`, double click | return to the fitted view |
| Drag | pan |
| Wheel | zooms once you are zoomed in; at the fitted view it scrolls the page, so the notes below stay reachable |
| `Ctrl`/`Cmd` + wheel | zoom from any state |

`?season=1966` in the URL opens that season. The page opens on **1 February of
the most recent season**, not on 1 October, because a season starts with no ice
anywhere and would look empty.

Inland lakes fade in between about 1.8× and 3.2× zoom. At the statewide view a
10 ha lake is about 0.3 px, so its honest average would read as noise; zoom in
to see them.

## Page structure

`#stage` is a full-viewport map. Below it, `#notebook` is a short document:
Method, Results, Read more. The view is sized from the canvas aspect ratio, not
from the raster extent, so the map reaches every edge with no letterbox; the
shader paints land wherever the view runs past the data. Drawing stops while the
map is scrolled out of view.

## What is drawn

- In-scope water is shaded by the fitted v2 model
  `h = max(0, k·sqrt(max(0, S − C1·ln d)) − m·P)` with `k = 1.6865`,
  `C1 = 34.38`, `m = 0.3318`, `d` from the static depth texture and `S`, `P`
  from the current frame.
- The ramp runs dark navy (0) to white (≥ 20.32 cm = 8 in).
- Out-of-scope water (Erie, Ontario, St Clair, the connecting rivers) is flat
  dark water and is never shaded as ice.
- Land is near-black; labels are a muted light blue.

## Data flow

```
data/derived/era5/cfdd_YYYY.npz   S, 0.1 deg, 3-day steps
data/derived/pdd/pdd_YYYY.npz     P, same grid
        │  python/build_web_fields.py
        ▼
web/fields/season_YYYY.bin.gz     gzip; "ICEF1\0\0\0" + uint32 n,h,w
                                  (20-byte header), then S frames, then P frames
web/fields/manifest.json          grid, model constants, frame dates

data/derived/rasters/             3" canonical depth + mask tiles
        │  python/build_web_layers.py
        ▼
web/layers/gl_cov.png             coverage, 0/255
web/layers/gl_dep.png             depth code; 255 = out-of-scope water
web/layers/inland_cov.png
web/layers/inland_dep.png
web/layers/manifest.json          bounds, 6" step, decode rules
```

Why coverage **and** depth instead of one byte: a single mipmapped byte averages
land `0` with deep-water `200` into a plausible mid-depth and paints a bright
halo around every shoreline. Two channels average honestly, and the shader
recovers `code = dep·255/cov`; coverage then blends the shoreline against land.

Season files are 1 Oct → 30 Jun. S resets on 1 October, P is defined 1 March –
31 July, and stopping at 30 June avoids the August step where P returns to zero
while S is still large. Seasons are fetched on demand, the next two prefetched,
and four kept in GPU memory.

## Files

| File | Purpose |
|---|---|
| `index.html` | markup, legend, controls, disclaimer |
| `style.css` | dark UI |
| `app.js` | WebGL2 renderer, season streaming, zoom/pan, labels |
| `fields/` | per-season S/P, 89.3 MB total (git-ignorable build output) |
| `layers/` | 6" coverage + depth textures, 2.3 MB total |

## Checks

- `python python\verify_rasters.py` reads the 3" product back from PNG and
  checks it against the shoreline gate.
- `python python\validate_shader.py --samples shader_samples.json` compares GPU
  pixels with a Python reference. See `reports/model-v1.md` §9 for the result.
- When checking the page itself, use a **composited** screenshot or
  `document.elementFromPoint()`. `canvas.toDataURL()` and `gl.readPixels()` see
  only the canvas, so they cannot detect a DOM element drawn over it — an empty
  full-screen error panel once covered the whole map while both looked perfect.
