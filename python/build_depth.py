"""Build a per-lake depth table for Michigan inland lakes >= 10 ha.

Sources, in priority order:
  1. DNR surveyed deep point (MaxDepth, feet -> metres)
  2. DNR depth contours (MinDepth in feet), integrated into a hypsographic curve
     to estimate volume and therefore mean depth
  3. nothing -> flagged for the area->depth regression

Output: data/derived/mi_lake_depth.csv
"""

from __future__ import annotations

import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

ROOT = pathlib.Path(__file__).resolve().parents[1]
DNR = ROOT / "data" / "raw" / "dnr"
OUT = ROOT / "data" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

FT_TO_M = 0.3048
CRS = 3078  # Michigan GeoRef


def ring_area_m2(geom) -> float:
    """Area enclosed by a contour line, treating it as a closed ring."""
    if geom is None or geom.is_empty:
        return 0.0
    lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
    total = 0.0
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            total += Polygon(coords).area
        except Exception:  # noqa: BLE001 - degenerate ring
            continue
    return total


def hypsographic_mean_depth(
    depths_m: np.ndarray, areas_m2: np.ndarray, lake_area_m2: float, max_depth_m: float
) -> float:
    """Trapezoidal integration of area-vs-depth, from 0 to max depth.

    Contour layers contain small fragment rings (digitising debris, island
    outlines) whose area is a rounding error against the lake. They must be
    dropped, otherwise monotonicity forces every deeper area to zero. The
    deepest sounding is a point, so its area is set to zero.
    """
    order = np.argsort(depths_m)
    z = depths_m[order]
    a = areas_m2[order]
    a = np.minimum(a, lake_area_m2)

    keep = a >= 0.005 * lake_area_m2
    z, a = z[keep], a[keep]

    # start at the shoreline, end at the deepest point
    z = np.concatenate([[0.0], z, [max_depth_m]])
    a = np.concatenate([[lake_area_m2], a, [0.0]])
    a = np.minimum.accumulate(a)

    volume = np.trapezoid(a, z) if hasattr(np, "trapezoid") else np.trapz(a, z)
    return float(volume / lake_area_m2) if lake_area_m2 > 0 else np.nan


def main() -> None:
    lakes = (
        gpd.read_file(DNR / "lakes_10ha.geojson")
        .reset_index()
        .rename(columns={"index": "lake_idx"})
    )
    lakes_m = lakes.to_crs(CRS)
    lakes["area_m2"] = lakes_m.geometry.area

    # --- deep points -> polygon index -------------------------------------
    pts = gpd.read_file(DNR / "deep_points.geojson")
    pts["depth_ft"] = pd.to_numeric(pts["MaxDepth"], errors="coerce")
    pts = pts[pts["depth_ft"] > 0].drop_duplicates("NewKey")
    joined = gpd.sjoin_nearest(
        pts.to_crs(CRS)[["NewKey", "depth_ft", "geometry"]],
        lakes_m[["lake_idx", "geometry"]],
        how="left",
        max_distance=250,
    )
    joined = (
        joined.dropna(subset=["lake_idx"])
        .sort_values("depth_ft", ascending=False)
        .drop_duplicates("lake_idx")
    )
    point_depth = joined.set_index("lake_idx")["depth_ft"]

    # --- contours -> polygon index ----------------------------------------
    contours = gpd.read_file(DNR / "contours.geojson").to_crs(CRS)
    cmap = pd.read_csv(OUT / "contour_polygon_map.csv")
    key_to_poly = cmap.drop_duplicates("NewKey").set_index("NewKey")["lake_idx"]
    contours["lake_idx"] = contours["NewKey"].map(key_to_poly)
    contours = contours.dropna(subset=["lake_idx"])
    contours["depth_ft"] = pd.to_numeric(contours["MinDepth"], errors="coerce")
    contours = contours[contours["depth_ft"] > 0]
    contours["ring_area"] = contours.geometry.map(ring_area_m2)

    cont_stats: dict[int, dict] = {}
    for lake_idx, grp in contours.groupby("lake_idx"):
        by_depth = grp.groupby("depth_ft")["ring_area"].sum()
        depths_m = by_depth.index.to_numpy(dtype=float) * FT_TO_M
        areas = by_depth.to_numpy(dtype=float)
        cont_stats[int(lake_idx)] = {
            "contour_max_ft": float(by_depth.index.max()),
            "n_contours": int(len(grp)),
            "depths": depths_m,
            "areas": areas,
        }

    rows = []
    for _, lake in lakes.iterrows():
        idx = int(lake["lake_idx"])
        st = cont_stats.get(idx)
        has_point = idx in point_depth.index

        max_ft = np.nan
        max_src = None
        if has_point:
            max_ft = float(point_depth.loc[idx])
            max_src = "point"
        elif st is not None:
            max_ft = st["contour_max_ft"]
            max_src = "contour"

        mean_m = np.nan
        if st is not None and pd.notna(max_ft):
            mean_m = hypsographic_mean_depth(
                st["depths"], st["areas"], float(lake["area_m2"]), max_ft * FT_TO_M
            )

        rows.append(
            {
                "lake_idx": idx,
                "name": lake["Name"],
                "area_km2": float(lake["SqKM"]),
                "area_m2": float(lake["area_m2"]),
                "max_depth_ft": max_ft,
                "max_depth_m": max_ft * FT_TO_M if pd.notna(max_ft) else np.nan,
                "mean_depth_m": mean_m,
                "n_contours": st["n_contours"] if st else 0,
                "depth_source": (
                    "point+contour"
                    if has_point and st
                    else "point"
                    if has_point
                    else "contour"
                    if st
                    else "none"
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mi_lake_depth.csv", index=False)

    print("wrote", OUT / "mi_lake_depth.csv")
    print(df["depth_source"].value_counts().to_string())
    have = df[df["depth_source"] != "none"]
    print(f"\nwith depth: {len(have)} / {len(df)} ({len(have)/len(df):.1%})")
    print("\nmax depth (m):", have["max_depth_m"].describe(percentiles=[.5, .9]).round(1).to_dict())
    print("mean depth (m):", have["mean_depth_m"].describe(percentiles=[.5, .9]).round(1).to_dict())
    ratio = (have["mean_depth_m"] / have["max_depth_m"]).replace([np.inf, -np.inf], np.nan).dropna()
    print("mean/max ratio:", ratio.describe(percentiles=[.1, .5, .9]).round(2).to_dict())


if __name__ == "__main__":
    main()
