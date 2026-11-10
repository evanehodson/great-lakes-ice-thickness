"""Normalise every ice-thickness observation into one table, attach a lake
depth, and freeze the train/validation split.

Sources
  g00803  Great Lakes + inland stations, 1966-79, fixed-width ASCII, cm
  ntl     Northern Temperate Lakes LTER, 1982-2026, cm
  eccc    ECCC Lake Ice database, metres

Depth convention
  The measurement is taken at the deep hole, so the water column below the
  instrument is the lake's maximum depth for inland lakes. For Great Lakes
  stations the depth is sampled from the NOAA 3 arc-second raster at the
  station coordinate (nearest water pixel if the coordinate is on shore).

Split
  Frozen. `data/derived/split_frozen.csv` holds the split fixed on 2026-11-06
  (80/20 by lake per region, Michigan inland 50/50, seed 20261106). This script
  never re-allocates it: a lake missing from that file is an error, because
  re-running the greedy allocator after dropping lakes reassigns surviving
  lakes.

Cleanup (scenario B, 2026-11-07)
  The 11 harbour stations (H1) and the three stations whose name and coordinate
  disagree (M1) are excluded from the observation pool. Inland depth is joined
  on the DNR polygon id (`lake_idx`), not on the lake name (H2).

Stations/lakes without a clean depth are dropped, not patched.

Outputs
  data/derived/observations.csv        one row per measurement
  data/derived/training_lakes.csv      one row per lake, with split
  data/derived/station_lake_match.csv  inland station -> DNR polygon audit
  data/derived/training_stations.csv   rewritten with lake_idx
"""

from __future__ import annotations

import pathlib
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
SEED = 20261106  # provenance of the frozen split; not used to allocate here
SPLIT_FROZEN = DER / "split_frozen.csv"

# Scenario B cleanup (reports/artifacts.md). H1: harbour depths are a
# shoreline-edge artifact. M1: station name and coordinate disagree.
# sid 115 (WARNERS LAKE) added 2026-11-07: no Michigan lake carries that name
# (checked against the MDNR statewide named-lakes list) and the nearest Warner
# Lake is 473 km away, while Andrus Lake is 0.35 km. M1 pattern -> drop.
HARBOUR_SIDS = {119, 125, 130, 134, 137, 140, 166, 167, 169, 201, 220}
WRONG_LAKE_SIDS = {115, 122, 204, 210}
DROP_SIDS = HARBOUR_SIDS | WRONG_LAKE_SIDS

# NTL published morphometry (lter.limnology.wisc.edu/core-study-lakes).
NTL_LAKES = {
    "AL": ("Allequash Lake",      46.038, -89.621,  8.0,  2.9),
    "BM": ("Big Muskellunge Lake", 46.021, -89.612, 21.3,  7.5),
    "CB": ("Crystal Bog",         46.008, -89.606,  2.5,  1.7),
    "CR": ("Crystal Lake",        46.003, -89.612, 20.4, 10.4),
    "FI": ("Fish Lake",           43.287, -89.652, 18.9,  6.6),
    "ME": ("Lake Mendota",        43.099, -89.405, 25.3, 12.8),
    "MO": ("Lake Monona",         43.063, -89.361, 22.5,  8.2),
    "SP": ("Sparkling Lake",      46.008, -89.701, 20.0, 10.9),
    "TB": ("Trout Bog",           46.041, -89.686,  7.9,  5.6),
    "TR": ("Trout Lake",          46.029, -89.665, 35.7, 14.6),
    "WI": ("Lake Wingra",         43.053, -89.425,  4.2,  2.7),
}
# ECCC sites in scope. Crystal Lake is the same waterbody as NTL CR.
ECCC_LAKES = {
    "crystal lake":       ("Crystal Lake",       46.0033, -89.6157, 20.4, 10.4, "ntl:CR"),
    "little turkey lake": ("Little Turkey Lake", 47.0430, -84.4080, 13.0, 5.3,  "eccc:LTL"),
}


# --------------------------------------------------------------------------
# g00803
# --------------------------------------------------------------------------
def read_g00803() -> pd.DataFrame:
    path = RAW / "g00803" / "extract" / "data2.dat"
    rows = []
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if len(line) < 11:
            continue
        try:
            sid = int(line[0:3])
            yy, mm, dd = int(line[4:6]), int(line[6:8]), int(line[8:10])
        except ValueError:
            continue

        def val(a: int, b: int) -> float:
            s = line[a:b].strip()
            try:
                return float(s) if s else np.nan
            except ValueError:
                return np.nan

        rows.append(
            dict(station_id=sid, year=1900 + yy, month=mm, day=dd,
                 total_cm=val(11, 14), black_cm=val(14, 17), white_cm=val(17, 20),
                 snow_cm=val(36, 38))
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    return df


# --------------------------------------------------------------------------
# NTL
# --------------------------------------------------------------------------
def read_ntl() -> pd.DataFrame:
    df = pd.read_csv(RAW / "ntl" / "ntl34_v10.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["sampledate"], errors="coerce")
    out = pd.DataFrame({
        "lakeid": df["lakeid"].astype(str).str.strip().str.upper(),
        "date": df["date"],
        "total_cm": pd.to_numeric(df["totice"], errors="coerce"),
        "white_cm": pd.to_numeric(df["whiteice"], errors="coerce"),
        "black_cm": pd.to_numeric(df["blueice"], errors="coerce"),
        "snow_cm": pd.to_numeric(df["avsnow"], errors="coerce"),
    })
    return out


# --------------------------------------------------------------------------
# ECCC
# --------------------------------------------------------------------------
def eccc_frame(path: pathlib.Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1", header=None, low_memory=False)
    has_lat = raw.apply(lambda r: r.astype(str).str.contains("Latitude", case=False).any(), axis=1)
    hdr = int(np.flatnonzero(has_lat.to_numpy())[0])
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    df = df.reset_index(drop=True)
    # Drop the two unit-annotation rows under the header.
    df = df[pd.to_numeric(df["Latitude"], errors="coerce").notna()]
    return df


def read_eccc() -> pd.DataFrame:
    frames = [eccc_frame(RAW / "eccc" / f) for f in ("ECCC_LI-TC_v6.csv", "External_LI-TC_v2.csv")]
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "site": df["Site Name"].astype(str).str.strip(),
        "date": pd.to_datetime(df["Sample Date"], errors="coerce", dayfirst=True),
        "total_cm": pd.to_numeric(df["Total Ice"], errors="coerce") * 100.0,
        "white_cm": pd.to_numeric(df["White Ice"], errors="coerce") * 100.0,
        "black_cm": pd.to_numeric(df["Black Ice"], errors="coerce") * 100.0,
        "snow_cm": pd.to_numeric(df["Snow Depth"], errors="coerce") * 100.0,
        "lat": pd.to_numeric(df["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(df["Longitude"], errors="coerce"),
    })
    return out


# --------------------------------------------------------------------------
# Great Lakes depth from the NOAA rasters
# --------------------------------------------------------------------------
def sample_gl_depth(lon: float, lat: float) -> tuple[float | None, str | None]:
    for lake in ("superior", "michigan", "huron"):
        with rasterio.open(RAW / "bathy" / f"{lake}.tif") as ds:
            b = ds.bounds
            if not (b.left <= lon <= b.right and b.bottom <= lat <= b.top):
                continue
            row, col = ds.index(lon, lat)
            r0, c0 = max(row - 30, 0), max(col - 30, 0)
            win = rasterio.windows.Window(c0, r0, min(61, ds.width - c0), min(61, ds.height - r0))
            a = ds.read(1, window=win)
            if a.size == 0:
                continue
            water = np.argwhere(a < 0)
            if water.size == 0:
                continue
            cy, cx = row - r0, col - c0
            d2 = ((water[:, 0] - cy) ** 2 + (water[:, 1] - cx) ** 2)
            r, c = water[d2.argmin()]
            return float(-a[r, c]), lake
    return None, None


# --------------------------------------------------------------------------
# Station -> DNR polygon id
# --------------------------------------------------------------------------
def match_inland_lakes(stations: pd.DataFrame) -> pd.DataFrame:
    """Attach the DNR polygon id to each Michigan inland station.

    Depth is a property of the polygon, not of the name: 371 lake names repeat
    across the state, so a name join picks an arbitrary one. The station
    coordinate is decisive when it falls inside a polygon; otherwise take the
    nearest polygon within 5 km (the historical station lists place the mark
    within ~1 km of the lake). This reproduces the H2 corrections exactly.
    """
    lakes = gpd.read_file(RAW / "dnr" / "lakes_10ha.geojson")
    if lakes.crs and lakes.crs.to_epsg() != 4326:
        lakes = lakes.to_crs(4326)
    lakes = lakes.reset_index().rename(columns={"index": "lake_idx", "Name": "name"})
    sindex = lakes.sindex

    out = stations.copy()
    for col in ("lake_idx", "lake_poly", "match_how", "match_km"):
        out[col] = pd.NA

    for i in out.index[out["region"] == "inland_mi"]:
        p = Point(float(out.at[i, "lon"]), float(out.at[i, "lat"]))
        # sindex.query(point) returns only polygons that intersect the point,
        # which is what "inside" needs; nearest must search the whole layer.
        inside = [k for k in sindex.query(p) if lakes.geometry.iloc[k].contains(p)]
        if inside:
            k = max(inside, key=lambda j: lakes.geometry.iloc[j].area)
            how, km = "inside", 0.0
        else:
            # Geodesic distance is not needed for a 5 km screen; the degree
            # distance times 111.32 is accurate to <1% at this latitude.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
                dist = lakes.geometry.distance(p) * 111.32
            near = dist[dist <= 5.0]
            if near.empty:
                continue
            k = int(near.idxmin())
            how, km = "nearest", float(near.min())
        out.at[i, "lake_idx"] = int(lakes.at[k, "lake_idx"])
        out.at[i, "lake_poly"] = lakes.at[k, "name"]
        out.at[i, "match_how"] = how
        out.at[i, "match_km"] = round(km, 2)

    out["lake_idx"] = out["lake_idx"].astype("Int64")
    return out


# --------------------------------------------------------------------------
def main() -> None:
    stations = pd.read_csv(DER / "training_stations.csv")
    depths = pd.read_csv(DER / "mi_lake_depth.csv")

    # H2: join inland depth on the polygon id, not the name.
    stations = match_inland_lakes(stations)
    stations.to_csv(DER / "training_stations.csv", index=False)
    (stations.loc[
        stations.region == "inland_mi",
        ["sid", "name", "lat", "lon", "dnr_lake",
         "lake_idx", "lake_poly", "match_how", "match_km"],
    ].to_csv(DER / "station_lake_match.csv", index=False))

    # Scenario B: H1 harbours and M1 wrong-lake stations leave the pool.
    pool = stations[~stations["sid"].isin(DROP_SIDS)].copy()

    obs_frames = []

    # --- g00803 ---------------------------------------------------------
    g = read_g00803()
    # inner join: stations excluded by scenario B, and the nine stations with
    # no coordinates in ice.loc, never enter the pool. They are a policy drop,
    # not a depth drop, so they must not inflate the "no depth" count.
    g = g.merge(pool[["sid", "name", "lat", "lon", "region", "dnr_lake"]],
                left_on="station_id", right_on="sid", how="inner")
    g["source"] = "g00803"
    g["lake_key"] = "g00803:" + g["station_id"].astype("Int64").astype(str)
    g["lake_name"] = g["name"]
    obs_frames.append(g[["source", "lake_key", "lake_name", "lat", "lon", "date",
                         "total_cm", "white_cm", "black_cm", "snow_cm", "region"]])

    # --- NTL ------------------------------------------------------------
    n = read_ntl()
    meta = pd.DataFrame(
        [(k, *v) for k, v in NTL_LAKES.items()],
        columns=["lakeid", "lake_name", "lat", "lon", "max_depth_m", "mean_depth_m"],
    )
    n = n.merge(meta, on="lakeid", how="left")
    n["source"] = "ntl"
    n["lake_key"] = "ntl:" + n["lakeid"]
    n["region"] = "inland_ntl"
    obs_frames.append(n[["source", "lake_key", "lake_name", "lat", "lon", "date",
                         "total_cm", "white_cm", "black_cm", "snow_cm", "region"]])

    # --- ECCC -----------------------------------------------------------
    e = read_eccc()
    e["key"] = e["site"].str.lower().str.strip()
    e = e[e["key"].isin(ECCC_LAKES)].copy()
    e["lake_name"] = e["key"].map(lambda k: ECCC_LAKES[k][0])
    e["lat"] = e["key"].map(lambda k: ECCC_LAKES[k][1])
    e["lon"] = e["key"].map(lambda k: ECCC_LAKES[k][2])
    e["lake_key"] = e["key"].map(lambda k: ECCC_LAKES[k][5])
    e["source"] = "eccc"
    # Crystal Lake is the same waterbody as NTL CR; keep one lake identity.
    e["region"] = np.where(e["lake_key"] == "ntl:CR", "inland_ntl", "inland_eccc")
    obs_frames.append(e[["source", "lake_key", "lake_name", "lat", "lon", "date",
                         "total_cm", "white_cm", "black_cm", "snow_cm", "region"]])

    obs = pd.concat(obs_frames, ignore_index=True)
    obs["date"] = pd.to_datetime(obs["date"], errors="coerce")
    obs = obs.dropna(subset=["date", "total_cm"])
    obs = obs[obs["total_cm"] > 0]

    # Collapse multi-point surveys to one value per lake-date. A dozen probes on
    # the same afternoon describe within-lake variability, not independent
    # evidence about a lake-scale model. Keep the spread as a measurement-error
    # term instead.
    obs = (
        obs.groupby(["lake_key", "date"], as_index=False)
        .agg(
            lake_name=("lake_name", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            region=("region", "first"),
            source=("source", "first"),
            total_cm=("total_cm", "mean"),
            sd_cm=("total_cm", "std"),
            n_points=("total_cm", "size"),
            white_cm=("white_cm", "mean"),
            black_cm=("black_cm", "mean"),
            snow_cm=("snow_cm", "mean"),
        )
    )
    obs["year"] = obs["date"].dt.year

    # --- lake table -----------------------------------------------------
    lakes = (obs.groupby("lake_key")
             .agg(lake_name=("lake_name", "first"), lat=("lat", "first"),
                  lon=("lon", "first"), region=("region", "first"),
                  n_obs=("total_cm", "size"),
                  first_year=("year", "min"), last_year=("year", "max"))
             .reset_index())

    # Michigan inland depth: DNR table, joined on the polygon id (H2)
    mi = pool[pool.region == "inland_mi"][["sid", "lake_idx"]].dropna()
    mi["lake_key"] = "g00803:" + mi["sid"].astype(int).astype(str)
    mi["lake_idx"] = mi["lake_idx"].astype(int)
    mi = mi.merge(depths[["lake_idx", "max_depth_m", "mean_depth_m"]],
                  on="lake_idx", how="left")
    lakes = lakes.merge(mi[["lake_key", "max_depth_m", "mean_depth_m"]], on="lake_key", how="left")

    # NTL + ECCC depth
    for key, (name, lat, lon, dmax, dmean) in NTL_LAKES.items():
        k = f"ntl:{key}"
        lakes.loc[lakes.lake_key == k, ["max_depth_m", "mean_depth_m"]] = [dmax, dmean]
    lakes.loc[lakes.lake_key == "eccc:LTL", ["max_depth_m", "mean_depth_m"]] = [13.0, 5.3]

    # Great Lakes depth: sample the raster at the station
    gl = lakes["region"].astype(str).str.startswith("gl_")
    for i in lakes.index[gl]:
        d, _ = sample_gl_depth(float(lakes.loc[i, "lon"]), float(lakes.loc[i, "lat"]))
        lakes.loc[i, "max_depth_m"] = d

    lakes["depth_source"] = np.where(
        lakes["region"].astype(str).str.startswith("gl_"), "noaa_bathy",
        np.where(lakes["lake_key"].str.startswith("ntl:"), "ntl_published",
                 np.where(lakes["lake_key"] == "eccc:LTL", "eccc_published", "dnr_bathymetry")))

    # Drop anything without a clean depth.
    before_df = lakes.copy()
    lakes = lakes[lakes["max_depth_m"].notna() & (lakes["max_depth_m"] > 0)].copy()
    dropped = len(before_df) - len(lakes)
    dropped_by_region = before_df[~before_df.lake_key.isin(lakes.lake_key)].groupby("region").size()

    # --- split ----------------------------------------------------------
    # Frozen on 2026-11-06. Never recomputed: dropping lakes changes the
    # regional targets and the RNG stream, which reassigns surviving lakes
    # (verified: 3 Michigan inland lakes drift). A lake absent from the frozen
    # file is an error, not an invitation to allocate.
    frozen = pd.read_csv(SPLIT_FROZEN).set_index("lake_key")["split"]
    missing = sorted(set(lakes["lake_key"]) - set(frozen.index))
    if missing:
        raise SystemExit(f"lakes missing from {SPLIT_FROZEN.name}: {missing}")
    lakes["split"] = lakes["lake_key"].map(frozen)
    if lakes["split"].isna().any():
        raise SystemExit("frozen split left lakes unassigned")

    obs = obs.merge(lakes[["lake_key", "split", "max_depth_m"]], on="lake_key", how="inner")

    obs = obs.sort_values(["lake_key", "date"]).reset_index(drop=True)
    obs.to_csv(DER / "observations.csv", index=False)
    lakes.sort_values(["region", "lake_key"]).to_csv(DER / "training_lakes.csv", index=False)

    print(f"observations: {len(obs):,}")
    print(f"lakes: {len(lakes)}  (dropped {dropped} without depth)")
    print()
    print("dropped by region:")
    print(dropped_by_region.to_string())
    print()
    print(lakes.groupby(["region", "split"]).agg(lakes=("lake_key", "size"), obs=("n_obs", "sum")).to_string())
    print()
    v = (obs.split == "val")
    print(f"validation share: {v.mean():.1%} of observations, "
          f"{lakes.split.eq('val').sum()}/{len(lakes)} lakes")
    print()
    print("depth range (m):", lakes.max_depth_m.describe(percentiles=[.5, .9]).round(1).to_dict())
    print()
    print("thickness (cm) by region:")
    print(obs.groupby("region")["total_cm"]
          .describe(percentiles=[.5, .95, .99]).round(1)
          .drop(columns=["count"]).to_string())
    print()
    top = obs.nlargest(5, "total_cm")[["source", "lake_name", "date", "total_cm"]]
    print("five thickest observations (outlier check):")
    print(top.to_string(index=False))
    print()
    print("Michigan inland split:")
    print(lakes[lakes.region == "inland_mi"][["lake_name", "n_obs", "max_depth_m", "split"]]
          .sort_values(["split", "n_obs"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
