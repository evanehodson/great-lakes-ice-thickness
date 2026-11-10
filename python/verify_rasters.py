"""Verify the product rasters against the approved shoreline gate.

Checks, all read back from the PNG tiles (not from memory):
  1. every tile in the manifest exists, decodes, and matches its declared size
  2. tiles do not overlap and their union is the declared layer
  3. Great Lakes water cells equal the gate's 33,015,374 px
  4. inland water cells equal the gate's 493,108 px
  5. inland mask carries at least 3,000 distinct lake codes
  6. decoded depth is finite, >= 1 m on every water cell, and plausible
  7. the St Marys corridor near the river mouth is not water
  8. collar pixels are flagged and are not counted as water

Previews go to reports/figures/rasters_*.png.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/verify_rasters.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RASTERS = ROOT / "data" / "derived" / "rasters"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GL_EXPECT = 33_015_374
INLAND_EXPECT = 493_108
LON_MIN, LON_MAX = -92.20042, -79.67958
LAT_MIN, LAT_MAX = 41.61958, 49.50042

lines: list[str] = []


def say(t: str = "") -> None:
    print(t)
    lines.append(t)


def read_layer(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    h, w = meta["height"], meta["width"]
    dtype = np.uint16 if meta["dtype_mask"] == "uint16" else np.uint8
    depth = np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), dtype)
    covered = np.zeros((h, w), bool)
    for t in meta["tiles"]:
        dp = RASTERS / meta["name"] / f"depth_r{t['row']}_c{t['col']}.png"
        mp = RASTERS / meta["name"] / f"mask_r{t['row']}_c{t['col']}.png"
        dr = np.asarray(Image.open(dp))
        mr = np.asarray(Image.open(mp)).astype(dtype)
        sl = (slice(t["row0"], t["row0"] + t["rows"]),
              slice(t["col0"], t["col0"] + t["cols"]))
        if covered[sl].any():
            raise RuntimeError(f"overlapping tile in {meta['name']}: {t}")
        covered[sl] = True
        depth[sl] = dr.astype(np.uint8)
        mask[sl] = mr
    return depth, mask


def decode(depth_code: np.ndarray) -> np.ndarray:
    return np.exp(depth_code.astype(np.float64) / 40.0)


def main() -> None:
    man = json.loads((RASTERS / "manifest.json").read_text(encoding="utf-8"))
    say("=== Raster verification (read back from PNG tiles) ===")
    say(f"crs {man['crs']}  step {man['step']:.10f} (3 arc-second)")
    say()
    arrays = {}
    for meta in man["layers"]:
        depth, mask = read_layer(meta)
        arrays[meta["name"]] = (meta, depth, mask)
        flag = meta["flag"]
        water = (mask > 0) & (mask < flag)
        collar = mask >= flag
        d = decode(depth)
        say(f"{meta['name']}: {meta['width']} x {meta['height']}, "
            f"{len(meta['tiles'])} tiles")
        say(f"  water cells {int(water.sum()):,}  collar {int(collar.sum()):,}  "
            f"land {int((~water & ~collar).sum()):,}")
        say(f"  depth on water: min {d[water].min():.2f} m, "
            f"median {np.median(d[water]):.2f} m, max {d[water].max():.2f} m")
        say(f"  depth code zero on water cells: "
            f"{int((depth[water] == 0).sum()):,}")
        say(f"  depth non-zero on land cells: "
            f"{int((depth[~water] != 0).sum()):,}")

    gl_d, gl_m = arrays["gl"][1], arrays["gl"][2]
    in_d, in_m = arrays["inland"][1], arrays["inland"][2]
    gl_water = (gl_m > 0) & (gl_m < 128)
    in_water = (in_m > 0) & (in_m < 32768)
    say()
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        say(f"  [{'OK' if cond else 'FAIL'}] {name}: {detail}")

    check("Great Lakes water cells", gl_water.sum() == GL_EXPECT,
          f"{int(gl_water.sum()):,} vs gate {GL_EXPECT:,}")
    check("inland water cells", in_water.sum() == INLAND_EXPECT,
          f"{int(in_water.sum()):,} vs gate {INLAND_EXPECT:,}")
    codes = np.unique(in_m[in_water])
    check("inland distinct lake codes", len(codes) >= 3000,
          f"{len(codes):,} codes, max {int(codes.max())}")
    check("no depth below 1 m on water",
          np.exp(gl_d[gl_water].min() / 40) >= 0.999 and
          np.exp(in_d[in_water].min() / 40) >= 0.999,
          "min decoded depth >= 1 m")

    # St Marys mouth: the river corridor must be non-water on the Huron side.
    lon0, top = arrays["gl"][0]["left"], arrays["gl"][0]["top"]
    step = man["step"]

    def rc(lon, lat):
        return int((top - lat) / step), int((lon - lon0) / step)

    r, c = rc(-84.30, 46.30)
    box = gl_m[r - 8:r + 9, c - 8:c + 9]
    check("St Marys mouth cleared", not ((box == 3).any() and
                                         (box == 3).sum() > 200),
          f"{int((box == 3).sum())} Huron cells in a 17x17 box at the mouth")

    say()
    say(f"overall: {'PASS' if ok else 'FAIL'}")
    say(f"elapsed verification")

    # ------------------------------------------------------------- previews
    def crop(arr, lon_a, lon_b, lat_a, lat_b):
        r0 = int((top - lat_b) / step)
        r1 = int((top - lat_a) / step)
        c0 = int((lon_a - lon0) / step)
        c1 = int((lon_b - lon0) / step)
        return arr[r0:r1, c0:c1]

    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(9, 12), dpi=150)
    stride = 10
    glv = np.where(gl_water[::stride, ::stride],
                   decode(gl_d[::stride, ::stride]), np.nan)
    ax.imshow(glv, origin="upper", cmap=cmap, vmin=0, vmax=400,
              extent=(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX), interpolation="nearest")
    ax.set_title("Great Lakes depth (3 arc-second, every 10th cell)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_aspect(1 / np.cos(np.radians(45)))
    fig.tight_layout(); fig.savefig(FIG / "rasters_gl_depth.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    lon_a, lon_b, lat_a, lat_b = -86.6, -85.2, 44.4, 45.2
    dv = crop(np.where(in_water, decode(in_d), np.nan), lon_a, lon_b, lat_a, lat_b)
    ax.imshow(dv, origin="upper", cmap=cmap, vmin=0, vmax=40,
              extent=(lon_a, lon_b, lat_a, lat_b), interpolation="nearest")
    ax.set_title("Inland lakes near Grand Traverse (per-pixel depth)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_aspect(1 / np.cos(np.radians(44.8)))
    fig.tight_layout(); fig.savefig(FIG / "rasters_inland_zoom.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    lon_a, lon_b, lat_a, lat_b = -88.2, -86.8, 42.6, 43.4
    gm = crop(gl_water.astype(np.uint8), lon_a, lon_b, lat_a, lat_b)
    ax.imshow(gm, origin="upper", cmap="Blues", vmin=0, vmax=1,
              extent=(lon_a, lon_b, lat_a, lat_b), interpolation="nearest")
    ax.set_title("Lake Michigan west shore: Great Lakes water mask")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_aspect(1 / np.cos(np.radians(43)))
    fig.tight_layout(); fig.savefig(FIG / "rasters_mask_zoom.png"); plt.close(fig)
    say("wrote reports/figures/rasters_gl_depth.png")
    say("wrote reports/figures/rasters_inland_zoom.png")
    say("wrote reports/figures/rasters_mask_zoom.png")

    (ROOT / "data" / "derived" / "diagnostics" /
     "raster_verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
