# Great Lakes ice thickness, 1950–2025

An animated map of **modelled** lake ice thickness for Lake Superior, Lake
Michigan, Lake Huron and the Michigan inland lakes of 10 ha and larger, for the
75 seasons 1950/51 – 2024/25.

**Live map:** <https://evanehodson.github.io/great-lakes-ice-thickness/>

> Model guidance, not a measurement. This is not a safety product and must never
> be used to judge whether ice is safe.

## What it shows

Ice thickness per pixel, from a fitted three-constant growth and melt model:

```
h = max(0, k · sqrt(max(0, S − C1 · ln d)) − m · P)
```

| Constant | Value | Meaning |
|---|---|---|
| `k` | 1.6865 | growth rate |
| `C1` | 34.38 | depth term, `ln d` with `d` in metres |
| `m` | 0.3318 | melt per thaw degree-hour |
| `C0` | 0 | fitted to zero and fixed |

`S` is cumulative freezing degree-days from 1 March, `P` is cumulative positive
degree-hours over the same window, both on a 0.1° ERA5 grid; `d` comes from a
static bathymetry raster. The ramp runs dark navy (0 cm) to white (≥ 20.32 cm =
8 in). Out-of-scope water (Erie, Ontario, St Clair, connecting rivers) is drawn
as flat dark water and never as ice.

Fit: train RMSE 13.4 cm; held-out 11.2 cm, MAE 8.7 cm, bias +1.4 cm, R² 0.556.
See `reports/model-v1.md` for the full method and `reports/STATUS.md` for the
current state, including open concerns.

## Controls

Play/pause (`Space`), season slider, speed, arrow keys to step frames, wheel to
zoom, drag to pan, double click to reset. `?season=1966` opens that season. The
page opens on 1 February of the most recent season because a season starts
ice-free and would look empty.

Inland lakes fade in between about 1.8× and 3.2× zoom: at statewide view a 10 ha
lake is about 0.3 px, so its honest average would read as noise.

## Run locally

```powershell
$env:PYTHONIOENCODING="utf-8"
python python\serve_web.py
```

Then open <http://127.0.0.1:8000/>. It must be served over HTTP, not opened from
`file://`, because the page fetches its data. Requires a browser with WebGL2 and
`DecompressionStream`.

## Layout

| Path | Contents |
|---|---|
| `web/` | the viewer (published to Pages) and its generated textures and fields |
| `python/` | every build, fit, verification and serving script |
| `reports/` | `STATUS.md` (single source of truth), `model-v1.md`, figures |
| `data/` | inputs and derived rasters — **not committed**, about 10 GB |

`web/fields/` (85 MB of per-season files) and `web/layers/` (2.2 MB of textures)
are committed on purpose: they are the build output that makes the page work
without shipping the 10 GB of source data.

## Publishing

Pushes to `main` deploy `web/` to GitHub Pages through
`.github/workflows/pages.yml`.
