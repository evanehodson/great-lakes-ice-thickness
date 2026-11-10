"""Decimate the product rasters to the display textures the viewer uploads.

The canonical 3 arc-second rasters (depth + lake-code mask) are the auditable
product and stay on disk. The browser needs far less: ice thickness depends on
depth only, so lake identity is irrelevant at display time.

Why two channels. A single depth byte cannot be mipmapped: averaging a land 0
with a deep-water 200 yields 100, which the shader would read as real water and
paint a bright halo around every shore. So each layer ships a pair of uint8
images at 6 arc-second:

    <name>_cov.png   255 = water (in-scope or context), 0 = land
    <name>_dep.png   255 = out-of-scope context water, else the depth code
    depth code       depth_m = exp((code) / 40), floor 1 m; 0 on land

Mipmaps average both channels. The shader recovers the depth code as
dep / cov, which is exact wherever both average over the same water, and blends
land into the result by coverage. That antialiases shorelines and lets a lake
smaller than a display pixel fade out instead of stippling the map.

The other water the bathy rectangles catch (Erie, Ontario, St Clair, rivers)
is kept as a flat context colour so the basemap does not show it as land. It is
never shaded as ice.

Resolution: 6 arc-second (~185 m), a 2 x 2 block mean of the product.
    Great Lakes  cov + dep 7513 x 4729
    inland       cov + dep 4557 x 3868

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_web_layers.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
RASTERS = ROOT / "data" / "derived" / "rasters"
WEB = ROOT / "web" / "layers"
WEB.mkdir(parents=True, exist_ok=True)

WEB_STEP = 1.0 / 600.0      # 6 arc-second
SCALE = 40.0
CONTEXT = 255


def read_layer(meta: dict):
    """Reassemble a canonical layer from its PNG tiles."""
    h, w = meta["height"], meta["width"]
    dtype = np.uint16 if meta["dtype_mask"] == "uint16" else np.uint8
    depth = np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), dtype)
    for t in meta["tiles"]:
        r, c = t["row0"], t["col0"]
        dr = np.asarray(Image.open(
            RASTERS / meta["name"] / f"depth_r{t['row']}_c{t['col']}.png"))
        mr = np.asarray(Image.open(
            RASTERS / meta["name"] / f"mask_r{t['row']}_c{t['col']}.png"))
        depth[r:r + t["rows"], c:c + t["cols"]] = dr.astype(np.uint8)
        mask[r:r + t["rows"], c:c + t["cols"]] = mr.astype(dtype)
    return depth, mask


def _blocks(arr: np.ndarray):
    h, w = arr.shape
    ph, pw = h % 2, w % 2
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw)))
    return arr.reshape(arr.shape[0] // 2, 2, arr.shape[1] // 2, 2)


def block_keep(water: np.ndarray, mode: str) -> np.ndarray:
    count = _blocks(water.astype(np.uint8)).sum(axis=(1, 3))
    return (count > 0) if mode == "any" else (count >= 2)


def block_depth(depth: np.ndarray, water: np.ndarray, keep: np.ndarray
                ) -> np.ndarray:
    count = _blocks(water.astype(np.uint8)).sum(axis=(1, 3))
    dsum = _blocks(depth.astype(np.int32) * water.astype(np.int32)
                   ).sum(axis=(1, 3))
    mean = np.zeros(count.shape, np.int32)
    np.divide(dsum, np.maximum(count, 1), out=mean, casting="unsafe",
              where=count > 0)
    out = np.zeros(count.shape, np.uint8)
    out[keep] = np.clip(mean[keep], 0, 254).astype(np.uint8)
    return out


def bathy_union(meta: dict) -> np.ndarray:
    """3 arc-second bathy-water union, rebuilt from the three NOAA tiles."""
    import rasterio
    left, top = meta["left"], meta["top"]
    step = 1.0 / 1200.0
    h, w = meta["height"], meta["width"]
    out = np.zeros((h, w), bool)
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(ROOT / "data" / "raw" / "bathy" / f"{name}.tif") as ds:
            z = ds.read(1, masked=True)
            t = ds.transform
        wm = (~np.ma.getmaskarray(z)) & (z < 0)
        r0 = int(round((top - t.f) / step))
        c0 = int(round((t.c - left) / step))
        out[r0:r0 + z.shape[0], c0:c0 + z.shape[1]] |= wm
    return out


def main() -> None:
    t0 = time.time()
    src = json.loads((RASTERS / "manifest.json").read_text(encoding="utf-8"))
    layers = []
    lines = ["=== Web layers: coverage + depth, 6 arc-second ===",
             "encoding: cov 0/255; dep 255 = context water, else "
             "depth = exp(dep/40) m"]
    for meta in src["layers"]:
        mode = "any" if meta["name"] == "inland" else "majority"
        depth, mask = read_layer(meta)
        water = (mask > 0) & (mask < meta["flag"])
        keep = block_keep(water, mode)
        dep = block_depth(depth, water, keep)
        cov = np.where(keep, 255, 0).astype(np.uint8)

        if meta["name"] == "gl":
            other = block_keep(bathy_union(meta) & ~water, "any")
            other = other[:cov.shape[0], :cov.shape[1]]
            ctx = other & ~keep[:other.shape[0], :other.shape[1]]
            cov[ctx] = 255
            dep[ctx] = CONTEXT
            lines.append(f"  context water (Erie, Ontario, neighbours): "
                         f"{int(ctx.sum()):,} cells")

        for kind, arr in (("cov", cov), ("dep", dep)):
            out = WEB / f"{meta['name']}_{kind}.png"
            Image.fromarray(arr, mode="L").save(out, optimize=True,
                                                compress_level=9)
        inscope = (cov > 0) & (dep < CONTEXT)
        dmax = float(np.exp(dep[inscope].max() / SCALE)) if inscope.any() else 0
        size = sum((WEB / f"{meta['name']}_{k}.png").stat().st_size
                   for k in ("cov", "dep"))
        lines.append(
            f"{meta['name']}: {meta['width']}x{meta['height']} -> "
            f"{cov.shape[1]}x{cov.shape[0]} ({cov.size / 1e6:.1f} M cells), "
            f"block rule {mode}, in-scope water "
            f"{100 * inscope.sum() / cov.size:.2f}% of bbox, "
            f"depth max {dmax:.1f} m, {size / 1e6:.2f} MB")
        layers.append(dict(name=meta["name"], cov=f"{meta['name']}_cov.png",
                           dep=f"{meta['name']}_dep.png",
                           width=int(cov.shape[1]), height=int(cov.shape[0]),
                           left=meta["left"], top=meta["top"], step=WEB_STEP))

    manifest = dict(
        crs="EPSG:4269",
        step=WEB_STEP,
        source_step=src["step"],
        encoding=dict(
            cov="0 = land, 255 = water",
            dep="255 = out-of-scope context water, else "
                "depth_m = exp(dep / 40) with a 1 m floor, 0 on land",
            depth_scale=SCALE,
            decode="depth_code = dep / (cov / 255) after mipmapping"),
        note="rows run north to south; column 0 is the west edge",
        layers=layers,
    )
    (WEB / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")
    lines.append(f"wrote {WEB / 'manifest.json'}")
    lines.append(f"elapsed {time.time() - t0:.1f} s")
    (WEB / "build_report.txt").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
