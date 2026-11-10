"""Build the product rasters: depth (log-encoded uint8) and mask, 3 arc-second.

Great Lakes
  Grid is the union of the three NOAA 3 arc-second tiles (EPSG:4269). Mask is
  HydroLAKES lake body intersected with bathy water, St Marys River removed
  (NOAA ESI HYDROP, CC0-1.0), DNR inland polygon wins on overlap. Depth is
  -elevation at the same pixel, encoded v = clip(round(ln(max(d,1))*40),0,255).

Michigan inland
  Grid is the DNR bounding box + 1 cell, same 3 arc-second lattice. Mask is the
  DNR polygon, code lake_idx+1. Depth:
    - surveyed lakes with contours: contour lines carry their MinDepth (feet
      -> m); shoreline carries 0; the deepest contour centroid carries the
      lake max depth. Every water pixel takes the depth of its nearest such
      feature, which is contour interpolation.
    - lakes with no contour: a distance-from-shore bowl
      depth = max * (1 - (1-u)^a), u = d/dmax, a = mean/(max-mean), so the
      bowl reproduces the lake's hypsographic mean depth.
  Encoded with the same log scheme.

Collar
  A 2-pixel land ring next to water takes the nearest lake code plus a high
  flag (128 on uint8, 32768 on uint16) and the neighbour's depth. It exists to
  antialias the shoreline in the shader; it is not drawn as water.

Outputs (tiles are 2048 x 2048, row-major, empty tiles skipped)
  data/derived/rasters/gl/depth_r{row}_c{col}.png   uint8
  data/derived/rasters/gl/mask_r{row}_c{col}.png    uint8
  data/derived/rasters/inland/depth_r{row}_c{col}.png uint8
  data/derived/rasters/inland/mask_r{row}_c{col}.png  uint16
  data/derived/rasters/manifest.json
  data/derived/rasters/build_report.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_rasters.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
OUT = DER / "rasters"
for sub in ("gl", "inland"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

STEP = 1.0 / 1200.0
SCALE = 40.0
GL_CODE = {"Superior": 1, "Michigan": 2, "Huron": 3}
FLAG8 = 128
FLAG16 = 32768
TILE = 2048

HYDROLAKES = (RAW / "shoreline" / "hydrolakes" /
              "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp")
ESI_GDB = (RAW / "shoreline" / "esi_stmarys" /
           "Great_Lakes_St_Marys_River_ESI_2021" /
           "Great Lakes - St Marys River ESI - 2021.gdb")
DNR_GEOJSON = RAW / "dnr" / "lakes_10ha.geojson"
CONTOURS = RAW / "dnr" / "contours.geojson"
CONTOUR_MAP = DER / "contour_polygon_map.csv"

FT_TO_M = 0.3048

lines: list[str] = []


def say(t: str = "") -> None:
    print(t)
    lines.append(t)


def encode_depth(depth_m: np.ndarray) -> np.ndarray:
    d = np.maximum(depth_m, 1.0)
    return np.clip(np.rint(np.log(d) * SCALE), 0, 255).astype(np.uint8)


def shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = a[r - dr, c - dc], zero filled at the grid edge (no wrap)."""
    out = np.zeros_like(a)
    h, w = a.shape
    rd = slice(max(0, dr), h + min(0, dr))
    rs = slice(max(0, -dr), h + min(0, -dr))
    cd = slice(max(0, dc), w + min(0, dc))
    cs = slice(max(0, -dc), w + min(0, -dc))
    out[rd, cd] = a[rs, cs]
    return out


def collar(mask: np.ndarray, depth: np.ndarray, flag: int, passes: int = 2):
    m = mask.copy()
    d = depth.copy()
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
    added = 0
    for _ in range(passes):
        best = np.zeros_like(m)
        best_d = np.zeros_like(d)
        for dr, dc in shifts:
            nm = shift(m, dr, dc)
            nd = shift(d, dr, dc)
            base = np.where(nm >= flag, nm - flag, nm).astype(m.dtype)
            take = (base > 0) & (base > best)
            best[take] = base[take]
            best_d[take] = nd[take]
        fill = (m == 0) & (best > 0)
        m[fill] = (best[fill] + flag).astype(m.dtype)
        d[fill] = best_d[fill]
        added += int(fill.sum())
    return m, d, added


def write_tiles(name: str, depth: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Write both layers as 2048 tiles; return the tile index."""
    h, w = depth.shape
    outdir = OUT / name
    tiles = []
    for r in range(0, h, TILE):
        for c in range(0, w, TILE):
            dr = depth[r:r + TILE, c:c + TILE]
            mr = mask[r:r + TILE, c:c + TILE]
            if not mr.any():
                continue
            rn, cn = r // TILE, c // TILE
            Image.fromarray(dr, mode="L").save(
                outdir / f"depth_r{rn}_c{cn}.png", optimize=True)
            if mask.dtype == np.uint8:
                Image.fromarray(mr, mode="L").save(
                    outdir / f"mask_r{rn}_c{cn}.png", optimize=True)
            else:
                Image.fromarray(mr, mode="I;16").save(
                    outdir / f"mask_r{rn}_c{cn}.png", optimize=True)
            tiles.append(dict(row=rn, col=cn, row0=int(r), col0=int(c),
                              rows=int(dr.shape[0]), cols=int(dr.shape[1])))
    return tiles


# ---------------------------------------------------------------- Great Lakes
def dnr_on_gl(transform, h: int, w: int):
    """Rasterise the DNR lakes on their own lattice, then map onto the GL grid.

    This is the gate rule (DNR polygon wins inside its boundary). The DNR
    lattice is a half-cell offset from the GL lattice, so the polygons are
    rasterised once at their native origin and then shifted by the rounded
    offset, exactly as build_shoreline_mask.py did.
    """
    gj = json.loads(DNR_GEOJSON.read_text(encoding="utf-8"))
    feats = gj["features"]
    xs, ys = [], []
    for f in feats:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for pl in polys:
            for ring in pl:
                for x, y in ring:
                    xs.append(x)
                    ys.append(y)
    lon0 = np.floor(min(xs) / STEP) * STEP - STEP
    lat0 = np.floor(min(ys) / STEP) * STEP - STEP
    lat1 = np.ceil(max(ys) / STEP) * STEP + STEP
    dnr_w = int(round((np.ceil(max(xs) / STEP) * STEP + STEP - lon0) / STEP))
    dnr_h = int(round((lat1 - lat0) / STEP))
    dnr_tr = from_origin(float(lon0), float(lat1), STEP, STEP)
    dnr = rasterize(((f["geometry"], i + 1) for i, f in enumerate(feats)),
                    out_shape=(dnr_h, dnr_w), transform=dnr_tr, fill=0,
                    dtype="uint16")
    dr0 = int(round((transform.f - lat1) / STEP))
    dc0 = int(round((lon0 - transform.c) / STEP))
    rr0, rr1 = max(0, dr0), min(h, dr0 + dnr_h)
    cc0, cc1 = max(0, dc0), min(w, dc0 + dnr_w)
    return dnr, (rr0 - dr0, rr1 - dr0, cc0 - dc0, cc1 - dc0), (rr0, rr1, cc0, cc1)


def build_gl():
    metas = {}
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            metas[name] = dict(bounds=ds.bounds, transform=ds.transform)
    left = min(m["bounds"].left for m in metas.values())
    right = max(m["bounds"].right for m in metas.values())
    top = max(m["bounds"].top for m in metas.values())
    bottom = min(m["bounds"].bottom for m in metas.values())
    h = int(round((top - bottom) / STEP))
    w = int(round((right - left) / STEP))
    transform = from_origin(left, top, STEP, STEP)

    water = np.zeros((h, w), dtype=bool)
    bathy = {}
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            z = ds.read(1, masked=True).astype(np.float32)
            t = ds.transform
        wm = (~np.ma.getmaskarray(z)) & (z < 0)
        depth = np.where(wm, -np.ma.filled(z, 0.0), 0.0).astype(np.float32)
        r0 = int(round((top - t.f) / STEP))
        c0 = int(round((t.c - left) / STEP))
        bathy[name] = dict(r0=r0, c0=c0, water=wm, depth=depth)
        water[r0:r0 + z.shape[0], c0:c0 + z.shape[1]] |= wm

    hl = pyogrio.read_dataframe(
        HYDROLAKES, columns=["Hylak_id", "Lake_name"],
        where="Lake_name IN ('Superior','Michigan','Huron')",
        bbox=(-93, 41, -79, 49.6))
    lake_geom = {r["Lake_name"]: make_valid(r.geometry) for _, r in hl.iterrows()}

    def raster_poly(geom):
        b = geom.bounds
        r0 = max(0, int(np.floor((top - b[3]) / STEP)) - 2)
        r1 = min(h, int(np.ceil((top - b[1]) / STEP)) + 3)
        c0 = max(0, int(np.floor((b[0] - left) / STEP)) - 2)
        c1 = min(w, int(np.ceil((b[2] - left) / STEP)) + 3)
        win_tr = transform * rasterio.Affine.translation(c0, r0)
        arr = rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                        transform=win_tr, fill=0, dtype="uint8",
                        all_touched=False).astype(bool)
        return arr, (r0, r1, c0, c1)

    poly = np.zeros((h, w), dtype=np.uint8)
    for name, code in GL_CODE.items():
        arr, (r0, r1, c0, c1) = raster_poly(lake_geom[name])
        sub = poly[r0:r1, c0:c1]
        np.maximum(sub, np.where(arr, code, 0).astype(np.uint8), out=sub)

    esi = pyogrio.read_dataframe(ESI_GDB, layer="HYDROP")
    esi_union = unary_union([make_valid(g) for g in
                             esi[esi["WATER_CODE"] == "W"].geometry])
    esi_arr, (er0, er1, ec0, ec1) = raster_poly(
        esi_union.buffer(2 * STEP))
    poly[er0:er1, ec0:ec1][esi_arr & (poly[er0:er1, ec0:ec1] == 3)] = 0

    mask = np.where(water, poly, 0).astype(np.uint8)

    # DNR inland polygon wins inside its boundary (gate rule).
    dnr, (sr0, sr1, sc0, sc1), (rr0, rr1, cc0, cc1) = dnr_on_gl(transform, h, w)
    overlap = (dnr[sr0:sr1, sc0:sc1] > 0) & (mask[rr0:rr1, cc0:cc1] > 0)
    ov_px = int(overlap.sum())
    del dnr
    mask[rr0:rr1, cc0:cc1][overlap] = 0
    say(f"DNR inland polygon wins: {ov_px:,} Great Lakes px cleared; "
        f"mask now {int((mask > 0).sum()):,} px")

    depth_code = np.zeros((h, w), dtype=np.uint8)
    for name, code in GL_CODE.items():
        b = bathy[name.lower()]
        sel = (mask[b["r0"]:b["r0"] + b["water"].shape[0],
                    b["c0"]:b["c0"] + b["water"].shape[1]] == code)
        enc = encode_depth(np.where(b["water"], b["depth"], 0.0))
        sub = depth_code[b["r0"]:b["r0"] + b["water"].shape[0],
                         b["c0"]:b["c0"] + b["water"].shape[1]]
        sub[sel] = enc[sel]

    mask, depth_code, added = collar(mask, depth_code, FLAG8)
    say(f"Great Lakes: {h} x {w} = {h * w / 1e6:.1f} M cells, "
        f"{int((mask > 0).sum()):,} cells with a code, collar {added:,} px")
    tiles = write_tiles("gl", depth_code, mask)
    say(f"  wrote {len(tiles)} GL tiles")
    return dict(name="gl", width=w, height=h, left=left, top=top, step=STEP,
                tile=TILE, dtype_depth="uint8", dtype_mask="uint8",
                flag=FLAG8, tiles=tiles), mask, depth_code


# -------------------------------------------------------------------- inland
def contour_geom(g: dict):
    """Shapely line geometry for a contour feature (LineString or MultiLineString)."""
    if g["type"] == "LineString":
        return LineString(g["coordinates"]) if len(g["coordinates"]) >= 2 else None
    parts = [LineString(c) for c in g["coordinates"] if len(c) >= 2]
    return MultiLineString(parts) if parts else None


def load_contours():
    gj = json.loads(CONTOURS.read_text(encoding="utf-8"))
    by_key: dict[str, list] = {}
    for f in gj["features"]:
        key = f["properties"]["NewKey"]
        z = float(f["properties"]["MinDepth"]) * FT_TO_M
        by_key.setdefault(key, []).append((z, f["geometry"]))
    cmap = pd.read_csv(CONTOUR_MAP)
    key_to_lake = dict(zip(cmap["NewKey"], cmap["lake_idx"]))
    by_lake: dict[int, list] = {}
    for key, items in by_key.items():
        li = key_to_lake.get(key)
        if li is None:
            continue
        by_lake.setdefault(int(li), []).extend(items)
    return by_lake


def feature_geom(g: dict):
    """Polygon or MultiPolygon feature -> shapely geometry, keeping island holes."""
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    parts = [Polygon(rings[0], rings[1:]) for rings in polys]
    return make_valid(unary_union(parts) if len(parts) > 1 else parts[0])


def bowl_depth(mask: np.ndarray, max_d: float, mean_d: float) -> np.ndarray:
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    d = ndimage.distance_transform_edt(mask)
    dmax = d.max()
    if dmax <= 0:
        return np.zeros(mask.shape, np.float32)
    u = d / dmax
    if not np.isfinite(mean_d) or mean_d <= 0 or mean_d >= max_d:
        a = 0.67
    else:
        a = np.clip(mean_d / (max_d - mean_d), 0.3, 3.0)
    return (max_d * (1.0 - (1.0 - u) ** a)).astype(np.float32)


def contour_depth(mask: np.ndarray, max_d: float, items, geom, transform
                  ) -> np.ndarray:
    """Nearest-feature depth from contour lines, shoreline and the deep centre."""
    h, w = mask.shape
    label = np.full((h, w), np.nan, np.float32)
    # shoreline
    edge = mask & ~ndimage.binary_erosion(mask, iterations=1)
    label[edge] = 0.0
    # contours, rasterised as thin lines
    shapes = []
    for z, g in items:
        ln = contour_geom(g)
        if ln is not None:
            shapes.append((ln, float(z)))
    if shapes:
        arr = rasterize(shapes, out_shape=(h, w), transform=transform,
                        fill=np.nan, dtype="float32", all_touched=True)
        take = np.isfinite(arr) & mask
        label[take] = arr[take]
    # deep centre at the centroid of the deepest contour
    if items:
        deepest = max(items, key=lambda t: t[0])[1]
        ring = deepest["coordinates"] if deepest["type"] == "LineString" \
            else max(deepest["coordinates"], key=len)
        if len(ring) >= 3:
            try:
                cen = Polygon(ring).centroid
                if not cen.within(geom):
                    raise ValueError
            except Exception:  # noqa: BLE001 - degenerate or unclosed ring
                cen = geom.representative_point()
            rr = int(np.floor((transform.f - cen.y) / STEP))
            cc = int(np.floor((cen.x - transform.c) / STEP))
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                label[rr, cc] = max_d
    have = np.isfinite(label)
    if not have.any():
        return bowl_depth(mask, max_d, np.nan)
    # nearest labeled pixel for every cell
    _, (ri, ci) = ndimage.distance_transform_edt(~have, return_indices=True)
    out = label[ri, ci]
    out[~mask] = 0.0
    return out.astype(np.float32)


def build_inland():
    gj = json.loads(DNR_GEOJSON.read_text(encoding="utf-8"))
    feats = gj["features"]
    xs, ys = [], []
    for f in feats:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for pl in polys:
            for ring in pl:
                for x, y in ring:
                    xs.append(x)
                    ys.append(y)
    lon0 = np.floor(min(xs) / STEP) * STEP - STEP
    lon1 = np.ceil(max(xs) / STEP) * STEP + STEP
    lat0 = np.floor(min(ys) / STEP) * STEP - STEP
    lat1 = np.ceil(max(ys) / STEP) * STEP + STEP
    w = int(round((lon1 - lon0) / STEP))
    h = int(round((lat1 - lat0) / STEP))
    transform = from_origin(float(lon0), float(lat1), STEP, STEP)

    dep = pd.read_csv(DER / "mi_lake_depth.csv").set_index("lake_idx")
    contours = load_contours()

    mask = np.zeros((h, w), np.uint16)
    depth = np.zeros((h, w), np.uint8)
    n_contour = n_bowl = 0
    for i, f in enumerate(feats):
        code = i + 1
        geom = feature_geom(f["geometry"])
        b = geom.bounds
        r0 = max(0, int(np.floor((lat1 - b[3]) / STEP)) - 1)
        r1 = min(h, int(np.ceil((lat1 - b[1]) / STEP)) + 2)
        c0 = max(0, int(np.floor((b[0] - lon0) / STEP)) - 1)
        c1 = min(w, int(np.ceil((b[2] - lon0) / STEP)) + 2)
        if r1 <= r0 or c1 <= c0:
            continue
        tr = transform * rasterio.Affine.translation(c0, r0)
        m = rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                      transform=tr, fill=0, dtype="uint8",
                      all_touched=False).astype(bool)
        if not m.any():
            continue
        max_d = float(dep.at[i, "max_depth_m"])
        mean_d = float(dep.at[i, "mean_depth_m"])
        items = contours.get(i, [])
        if items:
            dd = contour_depth(m, max_d, items, geom, tr)
            n_contour += 1
        else:
            dd = bowl_depth(m, max_d, mean_d)
            n_bowl += 1
        mask[r0:r1, c0:c1][m] = code
        depth[r0:r1, c0:c1][m] = encode_depth(dd[m])

    mask, depth, added = collar(mask, depth, FLAG16)
    say(f"Michigan inland: {h} x {w} = {h * w / 1e6:.1f} M cells, "
        f"{int((mask > 0).sum()):,} cells with a code, collar {added:,} px")
    say(f"  contour-interpolated lakes {n_contour:,}, bowl lakes {n_bowl:,}")
    tiles = write_tiles("inland", depth, mask)
    say(f"  wrote {len(tiles)} inland tiles")
    return dict(name="inland", width=w, height=h, left=float(lon0),
                top=float(lat1), step=STEP, tile=TILE,
                dtype_depth="uint8", dtype_mask="uint16",
                flag=FLAG16, tiles=tiles), mask, depth


def main() -> None:
    t0 = time.time()
    say("=== Product rasters: depth + mask, 3 arc-second ===")
    say(f"depth encoding: v = clip(round(ln(max(d,1))*{SCALE}), 0, 255)")
    say("PNG tiles 2048 x 2048, row-major, empty tiles skipped")
    say()
    gl_meta, gl_mask, gl_depth = build_gl()
    say()
    in_meta, in_mask, in_depth = build_inland()
    say()

    manifest = dict(
        crs="EPSG:4269",
        step=STEP,
        encoding=dict(depth_decode="exp(v/40), floor 1 m", depth_scale=SCALE),
        mask=dict(gl_flag=FLAG8, inland_flag=FLAG16,
                  note="low bits are the lake code; high flag is the 2-px land collar"),
        layers=[gl_meta, in_meta],
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")
    say(f"wrote {OUT / 'manifest.json'}")
    say(f"elapsed {time.time() - t0:.1f} s")
    (OUT / "build_report.txt").write_text("\n".join(lines) + "\n",
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
