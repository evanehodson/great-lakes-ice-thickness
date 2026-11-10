"""Diagnose the Great Lakes point-depth convention and build depth variants.

Four Great Lakes stations sample a point depth far below the surrounding water
(East Nipigon, Little Traverse, Mackinaw City, DeTour). Their coordinates fall
on water, so this is a depth-convention question, not a coordinate or snapping
artifact: a station in a shallow bay measures a genuinely shallow column, while
the ice model's depth term is meant to represent the lake's heat storage.

The script computes, for every Great Lakes station in the pool, the depth at the
exact point, the median depth of water pixels within 1.5 km and within 3 km, and
the distance to the nearest water pixel. It then writes one depth per lake for
each variant:

  V0   point depth (current convention, all stations)
  V1   1.5 km median, applied to the four shallow stations only
  V2   drop the four shallow stations (depth blank; the fit excludes them)
  V3   1.5 km median, applied to every Great Lakes station
  V3a  3 km median, applied to every Great Lakes station
  V3b  max(point, 1.5 km median), applied to every Great Lakes station

Inland lakes are unchanged in every variant.

The neighbourhood medians use bathymetry only. No ice observation enters the
calculation, so no variant is selected from held-out data.

Outputs
  data/derived/diagnostics/gl_depth_neighbourhoods.csv
  data/derived/depth_variants.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/shallow_bay_check.py
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

BATHY = {lake: RAW / "bathy" / f"{lake}.tif" for lake in ("superior", "michigan", "huron")}
SHALLOW_SIDS = {147, 214, 306, 314}
RADII_KM = ((1.5, "1p5"), (3.0, "3"))
M_PER_DEG = 111_320.0


def window_medians(lon: float, lat: float) -> dict:
    """Point depth plus water-pixel medians inside the two radii."""
    out = dict(lake="-", point_m=np.nan, shore_km=np.nan,
               n_water_1p5=0, median_1p5=np.nan, n_water_3=0, median_3=np.nan)
    for lake, path in BATHY.items():
        with rasterio.open(path) as ds:
            b = ds.bounds
            if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                continue
            pad = 0.05  # degrees; comfortably covers 3 km
            try:
                win = from_bounds(lon - pad, lat - pad, lon + pad, lat + pad, ds.transform)
            except Exception:  # noqa: BLE001 - window outside the raster
                continue
            a = ds.read(1, window=win, boundless=True, fill_value=ds.nodata)
            if a.size == 0:
                continue
            rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            xs, ys = rasterio.transform.xy(ds.window_transform(win), rr, cc, offset="center")
            xs = np.asarray(xs, dtype=float).reshape(a.shape)
            ys = np.asarray(ys, dtype=float).reshape(a.shape)
            dx = (xs - lon) * M_PER_DEG * np.cos(np.deg2rad(lat))
            dy = (ys - lat) * M_PER_DEG
            dist = np.hypot(dx, dy)

            water = (a < 0) & np.isfinite(a) & (a != ds.nodata)
            if not water.any():
                continue
            depth = -a[water].astype(float)
            wdist = dist[water]

            out.update(lake=lake)
            out["point_m"] = float(depth[wdist.argmin()])
            out["shore_km"] = float(wdist.min() / 1000.0)
            for r, tag in RADII_KM:
                sel = wdist <= r * 1000.0
                out[f"n_water_{tag}"] = int(sel.sum())
                out[f"median_{tag}"] = (
                    float(np.median(depth[sel])) if sel.any() else np.nan
                )
            break
    return out


def main() -> None:
    lakes = pd.read_csv(DER / "training_lakes.csv")
    stations = pd.read_csv(DER / "training_stations.csv")
    obs = pd.read_csv(DER / "observations.csv")

    gl = lakes[lakes.region.astype(str).str.startswith("gl_")].copy()
    sid = gl.lake_key.str.replace("g00803:", "", regex=False).astype(int)
    smap = stations.set_index("sid")
    gl["sid"] = sid
    gl["station"] = sid.map(smap["name"])
    gl["station_lat"] = sid.map(smap["lat"])
    gl["station_lon"] = sid.map(smap["lon"])

    rows = []
    for _, r in gl.sort_values("sid").iterrows():
        m = window_medians(float(r.station_lon), float(r.station_lat))
        rows.append(dict(
            lake_key=r.lake_key, sid=int(r.sid), station=r.station, region=r.region,
            split=r.split, n_obs=r.n_obs, lat=r.station_lat, lon=r.station_lon,
            raster=m["lake"], shore_km=round(m["shore_km"], 3),
            depth_point_m=round(float(r.max_depth_m), 2),
            n_water_1p5=m["n_water_1p5"],
            median_1p5_m=round(m["median_1p5"], 2) if np.isfinite(m["median_1p5"]) else np.nan,
            n_water_3=m["n_water_3"],
            median_3_m=round(m["median_3"], 2) if np.isfinite(m["median_3"]) else np.nan,
        ))
    neigh = pd.DataFrame(rows)
    neigh["point_over_1p5"] = (neigh.depth_point_m / neigh.median_1p5_m).round(3)
    neigh["point_over_3"] = (neigh.depth_point_m / neigh.median_3_m).round(3)
    neigh["shallow_flag"] = neigh.sid.isin(SHALLOW_SIDS)
    neigh.to_csv(OUT / "gl_depth_neighbourhoods.csv", index=False)

    # --- depth variants -------------------------------------------------
    v = lakes[["lake_key", "region", "max_depth_m"]].copy()
    v["V0"] = v.max_depth_m
    v["V1"] = v.max_depth_m
    v["V2"] = v.max_depth_m
    v["V3"] = v.max_depth_m
    v["V3a"] = v.max_depth_m
    v["V3b"] = v.max_depth_m

    n = neigh.set_index("lake_key")
    for key, row in n.iterrows():
        if key not in set(v.lake_key):
            continue
        if row.sid in SHALLOW_SIDS:
            v.loc[v.lake_key == key, "V1"] = row.median_1p5_m
            v.loc[v.lake_key == key, "V2"] = np.nan
        v.loc[v.lake_key == key, "V3"] = row.median_1p5_m
        v.loc[v.lake_key == key, "V3a"] = row.median_3_m
        v.loc[v.lake_key == key, "V3b"] = max(
            row.depth_point_m, row.median_1p5_m
        )
    v = v.rename(columns={"max_depth_m": "depth_current_m"})
    v.to_csv(DER / "depth_variants.csv", index=False)

    # --- report ---------------------------------------------------------
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 30)
    print("=== Great Lakes station depth: point vs neighbourhood ===")
    cols = ["sid", "station", "region", "split", "n_obs", "depth_point_m",
            "median_1p5_m", "median_3_m", "shore_km", "point_over_1p5", "point_over_3"]
    print(neigh[cols].to_string(index=False))
    print()
    print("=== the four flagged stations ===")
    print(neigh[neigh.shallow_flag][cols].to_string(index=False))
    print()
    print("full block used for V1/V2:")
    print(f"  stations {sorted(SHALLOW_SIDS)}  observations "
          f"{int(neigh[neigh.shallow_flag].n_obs.sum())}")
    val = neigh[neigh.shallow_flag & (neigh.split == "val")]
    print(f"  validation observations in the block: {int(val.n_obs.sum())}; "
          f"common subset for V1/V2 = {int(len(obs) - val.n_obs.sum())}")
    print()
    print("point depth below 50% of the 1.5 km median:",
          int((neigh.point_over_1p5 < 0.5).sum()), "stations")
    print(neigh[neigh.point_over_1p5 < 0.5][["sid", "station", "depth_point_m",
          "median_1p5_m", "point_over_1p5"]].to_string(index=False))
    print()
    print("variant depth coverage (lakes, non-null):")
    for c in ("V0", "V1", "V2", "V3", "V3a", "V3b"):
        print(f"  {c:4s} {int(v[c].notna().sum())} / {len(v)}")
    print()
    print("variants changing a station depth by >1 m (all GL):")
    ch = v[v.region.astype(str).str.startswith("gl_")].copy()
    ch["d_V3"] = (ch.V3 - ch.V0).round(2)
    print(ch.loc[ch.d_V3.abs() > 1, ["lake_key", "V0", "V1", "V2", "V3", "V3a", "V3b"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
