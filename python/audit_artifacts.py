"""Cross-source artifact audit.

Reproduces the findings in reports/artifacts.md:
  H1  harbour depths are a shoreline-edge artifact
  H2  duplicate lake names pick the wrong polygon's depth
  M1  stations matched to the wrong lake
  L4  total != black + white

Run with:  $env:PYTHONIOENCODING="utf-8"; python python/audit_artifacts.py
"""

from __future__ import annotations

import pathlib
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
RAW = ROOT / "data" / "raw"
BATHY = RAW / "bathy"
TIFS = ["superior.tif", "michigan.tif", "huron.tif"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    obs = pd.read_csv(DER / "observations.csv", parse_dates=["date"])
    stations = pd.read_csv(DER / "training_stations.csv")
    depths = pd.read_csv(DER / "mi_lake_depth.csv")
    lakes = gpd.read_file(RAW / "dnr" / "lakes_10ha.geojson")
    if lakes.crs and lakes.crs.to_epsg() != 4326:
        lakes = lakes.to_crs(4326)
    lakes = lakes.reset_index().rename(columns={"index": "lake_idx", "Name": "name"})
    lakes = lakes.merge(
        depths[["lake_idx", "max_depth_m", "area_km2"]], on="lake_idx", how="left"
    )
    return obs, stations, depths, lakes


def sample_depth(lon: float, lat: float, radius_px: int = 45) -> dict | None:
    """Nearest water pixel and neighbourhood median for a Great Lakes point."""
    for f in TIFS:
        with rasterio.open(BATHY / f) as ds:
            tr, sh = ds.transform, ds.shape
            col = int((lon - tr.c) / tr.a)
            row = int((lat - tr.f) / tr.e)
            if not (0 <= row < sh[0] and 0 <= col < sh[1]):
                continue
            r0, c0 = max(0, row - radius_px), max(0, col - radius_px)
            win = ds.read(
                1,
                window=rasterio.windows.Window(
                    c0, r0, 2 * radius_px + 1, 2 * radius_px + 1
                ),
            ).astype("float32")
            water = -win
            water[water <= 0] = np.nan
            if np.isnan(water).all():
                continue
            best, best_d = None, 1e9
            for i in range(water.shape[0]):
                for j in range(water.shape[1]):
                    if not np.isnan(water[i, j]):
                        d = ((i - (row - r0)) ** 2 + (j - (col - c0)) ** 2) ** 0.5
                        if d < best_d:
                            best_d, best = d, water[i, j]
            return dict(
                raster=f.replace(".tif", ""),
                nearest=float(best),
                median=float(np.nanmedian(water)),
                p95=float(np.nanpercentile(water, 95)),
                km_to_water=best_d * 0.000833 * 111,
            )
    return None


def h1_harbours(obs: pd.DataFrame) -> None:
    print("\n=== H1  harbour depths vs 3-4 km neighbourhood ===")
    h = (
        obs[obs.region == "gl_harbor"]
        .groupby("lake_name")
        .agg(lat=("lat", "first"), lon=("lon", "first"),
             used=("max_depth_m", "first"), n=("total_cm", "size"))
        .reset_index()
    )
    rows = []
    for _, r in h.iterrows():
        s = sample_depth(float(r.lon), float(r.lat))
        if s is None:
            continue
        rows.append(
            dict(lake=r.lake_name, obs=int(r.n), used=round(float(r.used), 2),
                 nearest=round(s["nearest"], 2), median=round(s["median"], 1),
                 ratio=round(float(r.used) / s["median"], 2) if s["median"] else np.nan)
        )
    d = pd.DataFrame(rows).sort_values("obs", ascending=False)
    print(d.to_string(index=False))
    print(f"obs with depth < 20% of neighbourhood median: "
          f"{int(d[d.ratio < 0.2].obs.sum())} of {int(d.obs.sum())}")


def h2_duplicate_names(obs, stations, depths, lakes) -> None:
    print("\n=== H2  duplicated lake names: used depth vs nearest same-name polygon ===")
    vc = depths.name.value_counts()
    mi = stations[(stations.region == "inland_mi") & stations.dnr_lake.notna()].copy()
    rows = []
    for _, r in mi.iterrows():
        sub = lakes[lakes.name == r.dnr_lake].copy()
        if not len(sub) or vc.get(r.dnr_lake, 0) < 2:
            continue
        p = Point(r.lon, r.lat)
        sub["km"] = sub.geometry.distance(p) * 111
        near = sub.loc[sub.km.idxmin()]
        used = float(depths[depths.name == r.dnr_lake].iloc[0].max_depth_m)
        err = used - float(near.max_depth_m)
        rows.append(
            dict(station=r["name"], matched=r.dnr_lake, polygons=len(sub),
                 used=round(used, 2), nearest_km=round(float(near.km), 2),
                 nearest=round(float(near.max_depth_m), 2),
                 nearest_area=round(float(near.area_km2), 2),
                 err_m=round(err, 2), obs=int(r.n_ice))
        )
    d = pd.DataFrame(rows)
    print(d.to_string(index=False))
    bad = d[d.err_m.abs() > 0.5]
    print(f"stations off by >0.5 m: {len(bad)}; obs affected: {int(bad.obs.sum())}")


def m1_wrong_lake(stations, lakes) -> None:
    print("\n=== M1  station name vs matched lake name ===")
    import re

    def norm(s: str) -> str:
        s = re.sub(r"[^a-z ]", " ", str(s).lower())
        s = re.sub(r"\b(lake|reservoir|lk)\b", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    mi = stations[(stations.region == "inland_mi") & stations.dnr_lake.notna()]
    rows = []
    for _, r in mi.iterrows():
        if norm(r["name"]) == norm(r.dnr_lake):
            continue
        same = lakes[lakes.name.map(norm) == norm(r["name"])]
        p = Point(r.lon, r.lat)
        matched = lakes[lakes.name == r.dnr_lake]
        rows.append(
            dict(station=r["name"], matched=r.dnr_lake,
                 matched_km=round(float(matched.distance(p).min() * 111), 2)
                 if len(matched) else np.nan,
                 same_name_km=round(float(same.distance(p).min() * 111), 2)
                 if len(same) else np.nan,
                 obs=int(r.n_ice))
        )
    print(pd.DataFrame(rows).to_string(index=False))


def l4_components(obs: pd.DataFrame) -> None:
    print("\n=== L4  total != black + white (|diff| > 2 cm) ===")
    s = obs[obs.black_cm.notna() & obs.white_cm.notna()].copy()
    s["diff"] = s.total_cm - (s.black_cm + s.white_cm)
    bad = s[s["diff"].abs() > 2].sort_values("diff", key=abs, ascending=False)
    print(bad[["source", "lake_name", "date", "total_cm", "black_cm",
               "white_cm", "diff"]].to_string(index=False))


def main() -> None:
    obs, stations, depths, lakes = load()
    print(f"observations: {len(obs):,} | lakes: {obs.lake_key.nunique()}")
    h1_harbours(obs)
    h2_duplicate_names(obs, stations, depths, lakes)
    m1_wrong_lake(stations, lakes)
    l4_components(obs)


if __name__ == "__main__":
    main()
