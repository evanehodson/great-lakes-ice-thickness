"""Proposal only: output grids, clip masks and layer sizes for the rasters.

Nothing here is a product raster. This script measures what the proposed layers
would cost, using the real bathymetry and the real DNR polygons:

  - Great Lakes: native NOAA 3 arc-second grids (EPSG:4269), cropped to each
    lake's water bounding box plus a collar. Depth is uint8 log-encoded
    (round(ln d * 40), floor 1 m); the mask is a separate uint8 index layer.
  - Michigan inland: one 3 arc-second grid over the DNR bounding box, split
    into 2048 x 2048 tiles, with uint8 depth and uint16 lake index.

Mask encoding proposed here (so code 0 never means 1 m of water):

    value  0                    no lake
    value  lake_code            water pixel
    value  lake_code + FLAG     land collar, nearest lake code

FLAG is 128 for the uint8 Great Lakes mask and 32768 for the uint16 inland
mask, so the lake code itself is always recoverable.

Each array is written to a temporary PNG and measured, then deleted. No file
under data/derived is created except the report and the size CSV.

Outputs
  data/derived/diagnostics/raster_grid_proposal.txt
  data/derived/raster_layer_sizes.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/raster_grid_proposal.py
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
import json

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

STEP = 1.0 / 1200.0          # 3 arc-second, exactly the NOAA grid step
SCALE = 40.0                 # log depth encoding
GL_CODE = {"superior": 1, "michigan": 2, "huron": 3}
FLAG8 = 128
FLAG16 = 32768


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    d = np.maximum(depth_m, 1.0)
    return np.clip(np.rint(np.log(d) * SCALE), 0, 255).astype(np.uint8)


def png_bytes(arr: np.ndarray) -> int:
    mode = "L" if arr.dtype == np.uint8 else "I;16"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "layer.png"
        Image.fromarray(arr, mode=mode).save(p, "PNG", optimize=True)
        return p.stat().st_size


def tile_sizes(arr: np.ndarray, size: int = 2048) -> tuple[list[int], int]:
    """PNG bytes for every non-empty tile, and how many tiles hold data."""
    h, w = arr.shape
    sizes, nonempty = [], 0
    for r in range(0, h, size):
        for c in range(0, w, size):
            t = arr[r:r + size, c:c + size]
            if not t.any():
                continue
            nonempty += 1
            sizes.append(png_bytes(t))
    return sizes, nonempty


def crop_to_water(mask: np.ndarray, pad: int = 8) -> tuple[slice, slice]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    r0, r1 = max(0, rows[0] - pad), min(mask.shape[0], rows[-1] + pad + 1)
    c0, c1 = max(0, cols[0] - pad), min(mask.shape[1], cols[-1] + pad + 1)
    return slice(r0, r1), slice(c0, c1)


def collar(mask: np.ndarray, depth: np.ndarray, flag: int,
           passes: int = 2) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill land next to water with the nearest lake code.

    A 4-neighbour dilation. `mask` holds the lake code on water (below `flag`)
    and 0 elsewhere. Each pass gives a land pixel the code of its strongest
    water neighbour and copies that neighbour's depth, so the depth texture
    reaches the shoreline without a zero ring.
    """
    m = mask.copy()
    d = depth.copy()
    added = 0
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for _ in range(passes):
        best = np.zeros_like(m)
        best_d = np.zeros_like(d)
        for dr, dc in shifts:
            nm = np.roll(np.roll(m, dr, axis=0), dc, axis=1)
            nd = np.roll(np.roll(d, dr, axis=0), dc, axis=1)
            base = np.where(nm >= flag, nm - flag, nm).astype(m.dtype)
            take = (base > 0) & (base > best)
            best[take] = base[take]
            best_d[take] = nd[take]
        fill = (m == 0) & (best > 0)
        m[fill] = best[fill] + flag
        d[fill] = best_d[fill]
        added += int(fill.sum())
    return m, d, added


def main() -> None:
    lines: list[str] = []
    def say(t: str = "") -> None:
        print(t); lines.append(t)

    rows = []

    say("=== Raster output proposal: grids, masks and layer sizes ===")
    say("PROPOSAL ONLY. No product raster exists after this script.")
    say()
    say(f"common grid: EPSG:4269 (NAD83), {STEP:.10f} deg = 3 arc-second, ")
    say("north-up, pixel is area; the same lattice as the NOAA bathymetry")
    say("depth code: uint8, v = clip(round(ln d * 40), 0, 255), floor 1 m, d in m")
    say("mask code: 0 = none, low bits = lake code, high flag = land collar")
    say()

    # ------------------------------------------------------------ Great Lakes
    say("=== Great Lakes, native 3 arc-second ===")
    gl_rows = []
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            z = ds.read(1, masked=True).astype(np.float64)
            transform = ds.transform
            bounds = ds.bounds
        water = (np.ma.getmaskarray(z) == False) & (z < 0)   # noqa: E712
        depth = np.where(water, -np.ma.filled(z, 0.0), 0.0)
        code = encode_depth(depth)
        code[~water] = 0
        mask = np.zeros(z.shape, np.uint8)
        mask[water] = GL_CODE[name]

        rs, cs = crop_to_water(mask)
        code_c = code[rs, cs]
        mask_c = mask[rs, cs]

        full_depth = png_bytes(code)
        full_mask = png_bytes(mask)
        crop_depth = png_bytes(code_c)
        crop_mask = png_bytes(mask_c)
        raw_depth = int(water.sum())
        gl_rows.append(dict(
            lake=name, grid=f"{z.shape[0]}x{z.shape[1]}",
            crop=f"{code_c.shape[0]}x{code_c.shape[1]}",
            water_pixels=int(water.sum()),
            full_depth_png=full_depth, full_mask_png=full_mask,
            crop_depth_png=crop_depth, crop_mask_png=crop_mask,
            depth_png_pct=100 * crop_depth / raw_depth,
        ))
        say(f"{name}: full grid {z.shape[0]}x{z.shape[1]}, "
            f"water crop {code_c.shape[0]}x{code_c.shape[1]}, "
            f"{int(water.sum()):,} water pixels")
        say(f"   full depth PNG {full_depth/1e6:.2f} MB, full mask PNG "
            f"{full_mask/1e6:.2f} MB")
        say(f"   cropped depth PNG {crop_depth/1e6:.2f} MB, cropped mask PNG "
            f"{crop_mask/1e6:.2f} MB")
        rows.append(dict(layer=f"gl_{name}_depth", grid=f"{z.shape[0]}x{z.shape[1]}",
                         cropped=f"{code_c.shape[0]}x{code_c.shape[1]}",
                         pixels=int(water.sum()), png_bytes=crop_depth,
                         png_mb=round(crop_depth / 1e6, 2)))
        rows.append(dict(layer=f"gl_{name}_mask", grid=f"{z.shape[0]}x{z.shape[1]}",
                         cropped=f"{mask_c.shape[0]}x{mask_c.shape[1]}",
                         pixels=int(water.sum()), png_bytes=crop_mask,
                         png_mb=round(crop_mask / 1e6, 2)))
    gl = pd.DataFrame(gl_rows)
    say()
    say(f"Great Lakes total, cropped: depth "
        f"{gl.crop_depth_png.sum()/1e6:.2f} MB, mask "
        f"{gl.crop_mask_png.sum()/1e6:.2f} MB, both "
        f"{(gl.crop_depth_png.sum()+gl.crop_mask_png.sum())/1e6:.2f} MB")
    say()

    # ------------------------------------------- combined Great Lakes mosaic
    metas = {}
    for name in GL_CODE:
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            metas[name] = (ds.bounds.left, ds.bounds.right,
                           ds.bounds.top, ds.bounds.bottom, ds.transform)
    left = min(v[0] for v in metas.values())
    right = max(v[1] for v in metas.values())
    top = max(v[2] for v in metas.values())
    bottom = min(v[3] for v in metas.values())
    w = int(round((right - left) / STEP))
    h = int(round((top - bottom) / STEP))
    mos_mask = np.zeros((h, w), np.uint8)
    mos_depth = np.zeros((h, w), np.uint8)
    overlap_water = 0
    tile_water = {}
    tile_origin = {}
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            z = ds.read(1, masked=True).astype(np.float64)
            t = ds.transform
        water = (np.ma.getmaskarray(z) == False) & (z < 0)   # noqa: E712
        depth = np.where(water, -np.ma.filled(z, 0.0), 0.0)
        code = encode_depth(depth)
        code[~water] = 0
        r0 = int(round((top - t.f) / STEP))
        c0 = int(round((t.c - left) / STEP))
        hh, ww = z.shape
        tile_water[name] = water
        tile_origin[name] = (r0, c0)
        sub_m = mos_mask[r0:r0 + hh, c0:c0 + ww]
        sub_d = mos_depth[r0:r0 + hh, c0:c0 + ww]
        overlap_water += int((water & (sub_m > 0)).sum())
        sel = water & (sub_m == 0)
        sub_m[sel] = GL_CODE[name]
        sub_d[sel] = code[sel]

    pair_counts = {}
    for a, b in (("superior", "michigan"), ("superior", "huron"),
                 ("michigan", "huron")):
        ra, ca = tile_origin[a]
        rb, cb = tile_origin[b]
        ha, wa = tile_water[a].shape
        hb, wb = tile_water[b].shape
        r0, r1 = max(ra, rb), min(ra + ha, rb + hb)
        c0, c1 = max(ca, cb), min(ca + wa, cb + wb)
        both = (tile_water[a][r0 - ra:r1 - ra, c0 - ca:c1 - ca]
                & tile_water[b][r0 - rb:r1 - rb, c0 - cb:c1 - cb])
        pair_counts[(a, b)] = int(both.sum())

    say("combined Great Lakes mosaic (no resampling; union of the three grids):")
    say(f"   {h}x{w} = {h*w/1e6:.1f} M cells; "
        f"{int((mos_mask>0).sum()):,} water pixels")
    say(f"   water pixels present in more than one tile: {overlap_water:,}")
    say(f"   depth PNG {png_bytes(mos_depth)/1e6:.2f} MB; "
        f"mask PNG {png_bytes(mos_mask)/1e6:.2f} MB")
    gl_tiles_d, gl_tiles = tile_sizes(mos_depth)
    gl_tiles_m, _ = tile_sizes(mos_mask)
    say(f"   2048 tiles: {gl_tiles} non-empty; depth "
        f"{sum(gl_tiles_d)/1e6:.2f} MB, mask {sum(gl_tiles_m)/1e6:.2f} MB")
    rows.append(dict(layer="gl_mosaic_depth", grid=f"{h}x{w}", cropped="2048 tiles",
                     pixels=int((mos_mask > 0).sum()),
                     png_bytes=sum(gl_tiles_d),
                     png_mb=round(sum(gl_tiles_d) / 1e6, 2)))
    rows.append(dict(layer="gl_mosaic_mask", grid=f"{h}x{w}", cropped="2048 tiles",
                     pixels=int((mos_mask > 0).sum()),
                     png_bytes=sum(gl_tiles_m),
                     png_mb=round(sum(gl_tiles_m) / 1e6, 2)))
    say("   the collar pass adds a 2-pixel shore ring (<0.01 MB); it is not "
        "included in these PNGs")
    say()

    # ---------------------------------------------------------------- inland
    say("=== Michigan inland, 3 arc-second, 2048-pixel tiles ===")
    gj = json.loads((RAW / "dnr" / "lakes_10ha.geojson").read_text(encoding="utf-8"))
    feats = gj["features"]
    xs, ys = [], []
    for f in feats:
        g = f["geometry"]
        polys = ([g["coordinates"]] if g["type"] == "Polygon"
                 else g["coordinates"])
        for poly in polys:
            for ring in poly:
                for x, y in ring:
                    xs.append(x); ys.append(y)
    lon0 = np.floor(min(xs) / STEP) * STEP - STEP
    lon1 = np.ceil(max(xs) / STEP) * STEP + STEP
    lat0 = np.floor(min(ys) / STEP) * STEP - STEP
    lat1 = np.ceil(max(ys) / STEP) * STEP + STEP
    width = int(round((lon1 - lon0) / STEP))
    height = int(round((lat1 - lat0) / STEP))
    transform = rasterio.transform.from_origin(lon0, lat1, STEP, STEP)

    shapes = ((f["geometry"], i + 1) for i, f in enumerate(feats))
    mask = rasterize(shapes, out_shape=(height, width), transform=transform,
                     fill=0, dtype="uint16")
    water = mask > 0
    say(f"DNR bounds lon {min(xs):.4f} to {max(xs):.4f}, "
        f"lat {min(ys):.4f} to {max(ys):.4f}")
    say(f"grid {height}x{width} = {height*width/1e6:.1f} M cells, "
        f"{int(water.sum()):,} water pixels "
        f"({100*water.mean():.2f}% of the grid)")
    say(f"lake codes 1..{len(feats)} (lake_idx + 1, matching the geojson order)")

    # A flat per-lake depth code is the cheap lower bound for the depth layer.
    depth = pd.read_csv(DER / "mi_lake_depth.csv").set_index("lake_idx")
    flat = np.zeros(mask.shape, np.uint8)
    for idx in depth.index:
        sel = mask == idx + 1
        if not sel.any():
            continue
        flat[sel] = encode_depth(np.array([max(depth.at[idx, "max_depth_m"], 1.0)]))[0]

    mask_c, flat_c, added = collar(mask, flat, FLAG16, passes=2)
    say(f"collar: {added:,} land pixels next to a lake assigned the nearest "
        f"lake code (+{FLAG16} flag) and its depth code")

    mask_sizes, mask_tiles = tile_sizes(mask_c)
    say(f"mask: full PNG {png_bytes(mask_c)/1e6:.2f} MB; "
        f"{mask_tiles} non-empty 2048 tiles, {sum(mask_sizes)/1e6:.2f} MB total")
    rows.append(dict(layer="inland_mask", grid=f"{height}x{width}",
                     cropped="2048 tiles", pixels=int(water.sum()),
                     png_bytes=sum(mask_sizes),
                     png_mb=round(sum(mask_sizes) / 1e6, 2)))

    depth_sizes, depth_tiles = tile_sizes(flat_c)
    say(f"depth, flat per-lake lower bound: {depth_tiles} non-empty tiles, "
        f"{sum(depth_sizes)/1e6:.2f} MB total")
    say("   (contour and bowl detail will add entropy; treat this as a floor)")
    rows.append(dict(layer="inland_depth_lower_bound", grid=f"{height}x{width}",
                     cropped="2048 tiles", pixels=int(water.sum()),
                     png_bytes=sum(depth_sizes),
                     png_mb=round(sum(depth_sizes) / 1e6, 2)))
    say()

    # ------------------------------------------------------------- totals
    gl_bytes = sum(gl_tiles_d) + sum(gl_tiles_m)
    inland_bytes = sum(mask_sizes) + sum(depth_sizes)
    say("=== download total ===")
    say(f"Great Lakes mosaic (tiled depth + mask): {gl_bytes/1e6:.2f} MB")
    say(f"Michigan inland (tiled depth + mask): {inland_bytes/1e6:.2f} MB")
    say(f"depth and mask only, all regions: "
        f"{(gl_bytes+inland_bytes)/1e6:.2f} MB")
    say("if the Great Lakes are served as three per-lake files instead of one "
        f"mosaic: {(gl.crop_depth_png.sum()+gl.crop_mask_png.sum())/1e6:.2f} MB "
        "(no seam fix at the Straits)")
    say("the per-frame S and P fields are separate and already tiny "
        "(81 x 171 uint16, ~28 KB per field)")
    say()

    say("=== how Michigan and Huron join at the Straits ===")
    say("All three NOAA tiles sit on the same 3 arc-second lattice. Pixel offsets:")
    say("  Superior -> Michigan  col +5040, row +4092")
    say("  Superior -> Huron     col +9240, row +3600")
    say("  Michigan -> Huron     col +4200, row -492")
    say("so the mosaic needs no resampling.")
    say(f"Water present in more than one tile: {overlap_water:,} pixels.")
    say(f"  {pair_counts[('superior','michigan')]:,}  Superior-Michigan: "
        "the top rows of northern Lake Michigan")
    say(f"  {pair_counts[('superior','huron')]:,}  Superior-Huron: "
        "the St Marys River / North Channel")
    say(f"  {pair_counts[('michigan','huron')]:,}  Michigan-Huron: "
        "the one-pixel seam at lon -84.5004")
    say("The sizing probe resolves overlaps by tile order; the build must use "
        "nearest lake body instead, so the northern-Michigan pixels do not "
        "become Superior.")
    say("Michigan's east edge (-84.4996) and Huron's west edge (-84.5004) leave a "
        "0.0008 deg gap, one pixel cell at this step. The collar pass fills it "
        "from the nearest valid water, so no artificial seam is introduced. "
        "S and P are shared atmospheric fields; the only possible step is the "
        "two source grids' depth difference at the boundary, taken from the "
        "valid tile on each side.")
    say()
    say("GATE NOTE ON THE MASK. The bathy mask is elevation below the surface, "
        "so it also includes the St Marys River / North Channel. Scope is the "
        "three lakes. Two options: (a) clip with a Great Lakes shoreline "
        "polygon (one small download, shoreline at the NOAA 0 m contour); "
        "(b) exclude the connecting channels by a corridor mask. Recommendation: "
        "(a). Not applied; this needs approval.")
    say()
    say("PROPOSAL ONLY: no raster, tile or mask was written to the project.")

    (OUT / "raster_grid_proposal.txt").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    pd.DataFrame(rows).to_csv(DER / "raster_layer_sizes.csv", index=False)
    print(f"\nwrote {OUT / 'raster_grid_proposal.txt'}")


if __name__ == "__main__":
    main()
