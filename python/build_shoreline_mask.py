"""Build the Great Lakes shore mask and measure it against the bathy water mask.

This is the shoreline-mask gate. It does not write a product raster. It writes
the mask counts, the DNR overlap counts and four preview images, then stops.

Lake bodies
  HydroLAKES v1.0 (CC-BY 4.0), one polygon each for Superior, Michigan and
  Huron. Huron already includes Georgian Bay and the North Channel, and the
  Straits of Mackinac sit in the Michigan/Huron pair.

Water
  The mask is the lake-body polygon intersected with NOAA 3 arc-second bathy
  water, so every mask pixel has a depth value. A water pixel needs both.

Connecting river removed
  NOAA ESI GL_St_Marys_River_2021 HYDROP water polygons (CC0-1.0), unioned,
  dilated by two output cells and subtracted from Huron. The ESI study area
  stops at the documented river mouth (46.05556 N, 83.90944 W); it does not
  enter the North Channel. The two-cell dilation absorbs the disagreement
  between the two shorelines, which otherwise leaves a one-pixel water wire
  along the river banks.

Inland rule
  Michigan DNR lakes >= 10 ha are rasterised on the same lattice. Where a DNR
  polygon overlaps a Great Lakes pixel the DNR polygon wins (the Great Lakes
  pixel is cleared) and the overlap is counted.

Reported counts, per lake
  kept              final mask pixels and area
  near removed      bathy water outside the polygon but within 3 km of it:
                    true shoreline disagreement
  far removed       bathy water more than 3 km outside the polygon: other
                    water bodies caught by the rectangular bathy tile (Lake
                    Erie, Lake Ontario, the Atlantic, the neighbouring lake).
                    Out of scope, not lost lake.
  polygon, no depth polygon area the bathy calls land or NoData; dropped so
                    the product never draws ice without a depth value.

Outputs
  data/derived/shoreline_mask_counts.csv
  data/derived/shoreline_mask_removed.csv
  data/derived/diagnostics/shoreline_mask_gate.txt
  reports/figures/gate_stmarys.png
  reports/figures/gate_straits.png
  reports/figures/gate_greenbay.png
  reports/figures/gate_saginaw.png
  reports/figures/gate_dnr_overlap.png

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_shoreline_mask.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
FIG = ROOT / "reports" / "figures"
for p in (DER, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

STEP = 1.0 / 1200.0
DILATE_CELLS = 2
NEAR_DEG = 0.03            # ~3 km
GL_CODE = {"Superior": 1, "Michigan": 2, "Huron": 3}
CODE_NAME = {v: k for k, v in GL_CODE.items()}

HYDROLAKES = (RAW / "shoreline" / "hydrolakes" /
              "HydroLAKES_polys_v10_shp" / "HydroLAKES_polys_v10.shp")
ESI_GDB = (RAW / "shoreline" / "esi_stmarys" /
           "Great_Lakes_St_Marys_River_ESI_2021" /
           "Great Lakes - St Marys River ESI - 2021.gdb")
DNR_GEOJSON = RAW / "dnr" / "lakes_10ha.geojson"

LAT_M = 110574.0
LON_M = 111320.0


def area_km2(mask: np.ndarray, top: float) -> float:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return 0.0
    counts = mask.sum(axis=1)[rows].astype(np.float64)
    lats = top - (rows + 0.5) * STEP
    pix = (STEP * LON_M * np.cos(np.radians(lats))) * (STEP * LAT_M)
    return float((counts * pix).sum() / 1e6)


def main() -> None:
    lines: list[str] = []

    def say(t: str = "") -> None:
        print(t)
        lines.append(t)

    say("=== Shoreline mask gate ===")
    say("No product raster is written by this script.")
    say()

    # ------------------------------------------------------------ mosaic grid
    metas = {}
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            metas[name] = dict(bounds=ds.bounds, transform=ds.transform,
                               shape=(ds.height, ds.width))
    left = min(m["bounds"].left for m in metas.values())
    right = max(m["bounds"].right for m in metas.values())
    top = max(m["bounds"].top for m in metas.values())
    bottom = min(m["bounds"].bottom for m in metas.values())
    height = int(round((top - bottom) / STEP))
    width = int(round((right - left) / STEP))
    transform = from_origin(left, top, STEP, STEP)
    say(f"grid EPSG:4269, 3 arc-second, {height} x {width} = "
        f"{height * width / 1e6:.1f} M cells")
    say(f"bounds lon {left:.5f}..{right:.5f}  lat {bottom:.5f}..{top:.5f}")
    say()

    # ------------------------------------------------------- bathy water mask
    water = np.zeros((height, width), dtype=bool)
    tile_water = {}
    for name in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{name}.tif") as ds:
            z = ds.read(1, masked=True).astype(np.float64)
            t = ds.transform
        w = (~np.ma.getmaskarray(z)) & (z < 0)
        r0 = int(round((top - t.f) / STEP))
        c0 = int(round((t.c - left) / STEP))
        hh, ww = z.shape
        tile_water[name] = dict(r0=r0, c0=c0, water=w, shape=(hh, ww))
        water[r0:r0 + hh, c0:c0 + ww] |= w
    say(f"bathy water, union of the three tiles: {int(water.sum()):,} px, "
        f"{area_km2(water, top):,.1f} km2")
    say("  the tiles are rectangles, so this union also catches other water "
        "bodies at their edges")
    say()

    # -------------------------------------------------------- lake body polys
    hl = pyogrio.read_dataframe(
        HYDROLAKES, columns=["Hylak_id", "Lake_name"],
        where="Lake_name IN ('Superior','Michigan','Huron')",
        bbox=(-93, 41, -79, 49.6))
    lake_geom = {r["Lake_name"]: make_valid(r.geometry) for _, r in hl.iterrows()}
    say("HydroLAKES v1.0, CC-BY 4.0:")
    for k in ("Superior", "Michigan", "Huron"):
        g = lake_geom[k]
        parts = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        nv = sum(len(p.exterior.coords) for p in parts)
        say(f"  {k}: {g.geom_type}, {len(parts)} part(s), {nv:,} exterior "
            f"vertices, area {g.area:.5f} deg2")
    say()

    # ------------------------------------------------------- ESI river water
    esi = pyogrio.read_dataframe(ESI_GDB, layer="HYDROP")
    esi_w = esi[esi["WATER_CODE"] == "W"]
    esi_union = unary_union([make_valid(g) for g in esi_w.geometry])
    say("NOAA ESI St Marys River 2021 HYDROP, CC0-1.0:")
    say(f"  {len(esi_w):,} water polygons, union {esi_union.geom_type}, "
        f"area {esi_union.area:.5f} deg2")
    say(f"  bounds lon {esi_union.bounds[0]:.5f}..{esi_union.bounds[2]:.5f} "
        f"lat {esi_union.bounds[1]:.5f}..{esi_union.bounds[3]:.5f}")
    say("  south edge stops at the documented mouth, 46.05556 N; the North "
        "Channel is outside")
    say()

    # ------------------------------------------------------------- rasterise
    def raster_poly(geom):
        b = geom.bounds
        r0 = max(0, int(np.floor((top - b[3]) / STEP)) - 2)
        r1 = min(height, int(np.ceil((top - b[1]) / STEP)) + 3)
        c0 = max(0, int(np.floor((b[0] - left) / STEP)) - 2)
        c1 = min(width, int(np.ceil((b[2] - left) / STEP)) + 3)
        win_tr = transform * rasterio.Affine.translation(c0, r0)
        arr = rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                        transform=win_tr, fill=0, dtype="uint8",
                        all_touched=False)
        return arr.astype(bool), (r0, r1, c0, c1)

    poly = np.zeros((height, width), dtype=np.uint8)
    for name, code in GL_CODE.items():
        arr, (r0, r1, c0, c1) = raster_poly(lake_geom[name])
        sub = poly[r0:r1, c0:c1]
        np.maximum(sub, np.where(arr, code, 0).astype(np.uint8), out=sub)

    esi_dil = esi_union.buffer(DILATE_CELLS * STEP)
    esi_arr, (er0, er1, ec0, ec1) = raster_poly(esi_dil)
    poly_pre = poly.copy()
    huron_before = int((poly == GL_CODE["Huron"]).sum())
    cut = esi_arr & (poly[er0:er1, ec0:ec1] == GL_CODE["Huron"])
    poly[er0:er1, ec0:ec1][cut] = 0
    cut_px = int(cut.sum())
    say(f"St Marys removal: ESI water dilated {DILATE_CELLS} cells, "
        f"{cut_px:,} Huron px cleared ({100 * cut_px / max(1, huron_before):.3f}%)")
    say()

    # ------------------------------------------- polygon ∩ bathy water = mask
    mask = np.where(water, poly, 0).astype(np.uint8)
    gap_px = int(((poly > 0) & ~water).sum())
    say(f"polygon but no bathy water: {gap_px:,} px "
        f"({area_km2((poly > 0) & ~water, top):,.1f} km2), "
        f"a one-pixel shoreline ring plus the Straits seam; left out of the mask")
    say()

    # ------------------------------------------------------------- per lake
    rows = []
    removed_rows = []
    say("per-lake mask accounting (area from latitude-corrected pixel areas)")
    say(f"{'lake':<9} {'kept px':>12} {'kept km2':>10} "
        f"{'near rm px':>11} {'near km2':>9} {'far rm px':>10} {'far km2':>9} "
        f"{'no-depth px':>11} {'no-depth km2':>12}")
    for name in ("Superior", "Michigan", "Huron"):
        code = GL_CODE[name]
        kept = mask == code
        near_buf, (nr0, nr1, nc0, nc1) = raster_poly(
            lake_geom[name].buffer(NEAR_DEG))
        near = np.zeros((height, width), dtype=bool)
        near[nr0:nr1, nc0:nc1] = near_buf
        t = tile_water[name.lower()]
        tw = np.zeros((height, width), dtype=bool)
        tw[t["r0"]:t["r0"] + t["shape"][0],
           t["c0"]:t["c0"] + t["shape"][1]] = t["water"]
        rm = tw & (poly_pre != code)
        near_rm = rm & near
        far_rm = rm & ~near
        nod = (poly == code) & ~water
        say(f"{name:<9} {int(kept.sum()):>12,} {area_km2(kept, top):>10,.1f} "
            f"{int(near_rm.sum()):>11,} {area_km2(near_rm, top):>9,.1f} "
            f"{int(far_rm.sum()):>10,} {area_km2(far_rm, top):>9,.1f} "
            f"{int(nod.sum()):>11,} {area_km2(nod, top):>12,.1f}")
        rows.append(dict(lake=name, kept_px=int(kept.sum()),
                         kept_km2=round(area_km2(kept, top), 1),
                         near_removed_px=int(near_rm.sum()),
                         near_removed_km2=round(area_km2(near_rm, top), 1),
                         far_removed_px=int(far_rm.sum()),
                         far_removed_km2=round(area_km2(far_rm, top), 1),
                         polygon_no_depth_px=int(nod.sum()),
                         polygon_no_depth_km2=round(area_km2(nod, top), 1)))
        removed_rows.append(dict(lake=name, raw_tile_water_px=int(tw.sum())))
    say()

    total = mask > 0
    say(f"final Great Lakes mask: {int(total.sum()):,} px, "
        f"{area_km2(total, top):,.1f} km2")
    for code, name in CODE_NAME.items():
        m = mask == code
        say(f"  {name:<9} {int(m.sum()):>10,} px  {area_km2(m, top):>10,.1f} km2")
    say()

    # ---------------------------------------------------------- DNR overlap
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
    lat1 = np.ceil(max(ys) / STEP) * STEP + STEP
    lat0 = np.floor(min(ys) / STEP) * STEP - STEP
    dnr_w = int(round((np.ceil(max(xs) / STEP) * STEP + STEP - lon0) / STEP))
    dnr_h = int(round((lat1 - lat0) / STEP))
    dnr_tr = from_origin(float(lon0), float(lat1), STEP, STEP)
    dnr_shapes = ((f["geometry"], i + 1) for i, f in enumerate(feats))
    dnr = rasterize(dnr_shapes, out_shape=(dnr_h, dnr_w), transform=dnr_tr,
                    fill=0, dtype="uint16")
    say(f"DNR inland lakes: {len(feats):,} polygons, {dnr_h} x {dnr_w} grid, "
        f"{int((dnr > 0).sum()):,} water px")
    dr0 = int(round((top - lat1) / STEP))
    dc0 = int(round((lon0 - left) / STEP))
    rr0, rr1 = max(0, dr0), min(height, dr0 + dnr_h)
    cc0, cc1 = max(0, dc0), min(width, dc0 + dnr_w)
    dnr_sub = dnr[rr0 - dr0:rr1 - dr0, cc0 - dc0:cc1 - dc0]
    gl_sub = mask[rr0:rr1, cc0:cc1]
    overlap = (dnr_sub > 0) & (gl_sub > 0)
    ov_px = int(overlap.sum())
    say(f"Great Lakes mask and DNR polygons overlap: {ov_px:,} px, "
        f"{area_km2(overlap, top):,.2f} km2")
    if ov_px:
        codes, counts = np.unique(gl_sub[overlap], return_counts=True)
        for c, n in zip(codes, counts):
            say(f"  inside {CODE_NAME[int(c)]}: {int(n):,} px")
        idx, counts = np.unique(dnr_sub[overlap], return_counts=True)
        order = np.argsort(counts)[::-1]
        dep = pd.read_csv(DER / "mi_lake_depth.csv")
        names = dict(zip(dep["lake_idx"], dep["name"]))
        areas = dict(zip(dep["lake_idx"], dep["area_km2"]))
        say("  overlapping DNR lakes:")
        for j in order:
            li = int(idx[j]) - 1
            say(f"    lake_idx {li:>4}  {str(names.get(li, '')).strip():<24} "
                f"{int(counts[j]):>6} px  {area_km2(overlap & (dnr_sub == idx[j]), top):>6.2f} km2 "
                f"of {areas.get(li, float('nan')):.2f} km2")
    say("rule: the DNR polygon wins inside its boundary; those pixels leave the "
        "Great Lakes mask")
    mask[rr0:rr1, cc0:cc1][overlap] = 0
    say(f"after the rule: Great Lakes mask {int((mask > 0).sum()):,} px, "
        f"{area_km2(mask > 0, top):,.1f} km2")
    say()

    # ------------------------------------------------------------- previews
    cmap = ListedColormap(["#0b1220", "#1f4e79", "#2e75b6", "#70ad47"])

    def render(bounds, title, path, stride=4):
        x0, y0, x1, y1 = bounds
        c0 = max(0, int((x0 - left) / STEP))
        c1 = min(width, int((x1 - left) / STEP))
        r0 = max(0, int((top - y1) / STEP))
        r1 = min(height, int((top - y0) / STEP))
        ss = mask[r0:r1, c0:c1]
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.imshow(ss[::stride, ::stride], origin="upper", cmap=cmap,
                  vmin=0, vmax=3, extent=(x0, x1, y0, y1),
                  interpolation="nearest")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG / path)
        plt.close(fig)
        say(f"  wrote reports/figures/{path}")

    say("previews")

    # St Marys: overview + mouth zoom, with the removed river outlined
    esi_line = esi_union.boundary
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), dpi=150)
    for ax, (bounds, title) in zip(axes, [
            ((-84.62, 46.00, -83.82, 46.58),
             "St Marys: Soo (top) to De Tour (bottom)"),
            ((-84.10, 45.98, -83.86, 46.14),
             "River mouth at 46.0556 N, 83.9094 W")]):
        x0, y0, x1, y1 = bounds
        c0 = max(0, int((x0 - left) / STEP))
        c1 = min(width, int((x1 - left) / STEP))
        r0 = max(0, int((top - y1) / STEP))
        r1 = min(height, int((top - y0) / STEP))
        ss = mask[r0:r1, c0:c1]
        ax.imshow(ss, origin="upper", cmap=cmap, vmin=0, vmax=3,
                  extent=(x0, x1, y0, y1), interpolation="nearest")
        for g in (esi_union.geoms if esi_union.geom_type.startswith("Multi")
                  else [esi_union]):
            if g.bounds[2] < x0 or g.bounds[0] > x1:
                continue
            for ring in [g.exterior, *g.interiors]:
                xy = np.array(ring.coords)
                ax.plot(xy[:, 0], xy[:, 1], color="#ff9f43", lw=0.5)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.grid(alpha=0.25)
    fig.suptitle("Final mask; orange = ESI St Marys River water (cleared)")
    fig.tight_layout()
    fig.savefig(FIG / "gate_stmarys.png")
    plt.close(fig)
    say("  wrote reports/figures/gate_stmarys.png")

    render((-84.95, 45.68, -84.30, 46.08),
           "Straits of Mackinac: Michigan (blue) and Huron (green)",
           "gate_straits.png", stride=2)
    render((-88.15, 44.45, -86.85, 45.85), "Green Bay",
           "gate_greenbay.png", stride=5)
    render((-84.35, 43.45, -83.35, 44.35), "Saginaw Bay",
           "gate_saginaw.png", stride=4)

    # DNR overlap overview
    x0 = left + cc0 * STEP
    x1 = left + cc1 * STEP
    y0 = top - rr1 * STEP
    y1 = top - rr0 * STEP
    img = np.zeros((rr1 - rr0, cc1 - cc0), dtype=np.uint8)
    img[mask[rr0:rr1, cc0:cc1] > 0] = 1
    img[overlap] = 2
    stride = max(1, int(round(img.shape[1] / 1400)))
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    ax.imshow(img[::stride, ::stride], origin="upper",
              cmap=ListedColormap(["#0b1220", "#2e75b6", "#ff4d4d"]),
              vmin=0, vmax=2, extent=(x0, x1, y0, y1),
              interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2)))
    ax.set_title(f"Great Lakes mask (blue) vs DNR overlap (red), {ov_px:,} px")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "gate_dnr_overlap.png")
    plt.close(fig)
    say("  wrote reports/figures/gate_dnr_overlap.png")
    say()

    # ------------------------------------------------------------- outputs
    pd.DataFrame(rows).to_csv(DER / "shoreline_mask_counts.csv", index=False)
    pd.DataFrame(removed_rows).to_csv(DER / "shoreline_mask_removed.csv",
                                      index=False)
    say(f"wrote {DER / 'shoreline_mask_counts.csv'}")
    say(f"wrote {DER / 'shoreline_mask_removed.csv'}")
    (OUT / "shoreline_mask_gate.txt").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    say(f"wrote {OUT / 'shoreline_mask_gate.txt'}")
    say()
    say("GATE: stop here. No product raster generated.")


if __name__ == "__main__":
    main()
