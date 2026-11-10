"""Gate B follow-up: audit the spatial station -> lake match by name.

The M1 failures happened where a station name and its coordinate disagreed, and
`ice.loc` coordinates are only good to about +/-0.9 km. A purely spatial match
can therefore still land on the wrong small lake when names repeat. This script
changes nothing; it reports:

  1. the matched polygon, match method and distance for every inland station;
  2. the name-agreement flag between the station name and the matched polygon;
  3. the nearest polygon carrying the station's own name (the M1 test);
  4. the margin to the nearest *other* polygon sharing the matched name, which
     is the real duplicate-name risk;
  5. depth provenance for every Michigan inland lake in the pool.

Outputs
  data/derived/diagnostics/station_name_check.csv
  data/derived/diagnostics/inland_depth_provenance.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/check_station_names.py
"""

from __future__ import annotations

import pathlib
import re
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
warnings.filterwarnings("ignore", message="invalid value encountered in")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

GENERIC = r"\b(lakes?|lks?|lk|reservoirs?|ponds?)\b"


def norm(s: object) -> str:
    """Lower-case; drop punctuation, lake words and plurals."""
    s = str(s).lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(GENERIC, " ", s)
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in s.split()]
    return " ".join(toks).strip()


def name_parts(station: object) -> list[str]:
    """Normalised forms of a station name worth testing against polygons.

    Station names come in three shapes: plain (``BEAR LAKE``), prefixed with a
    place (``BEAVER ISLAND-LAKE BARNEY``), or suffixed with a harbour
    (``MUSKEGON LAKE-SNUG HARBOR``). Both the full string and each hyphen part
    are tested.
    """
    parts = {norm(station)}
    parts.update(norm(p) for p in re.split(r"[-–]", str(station)))
    return [p for p in parts if p]


def agreement(station: object, polygon: object) -> str:
    """exact / partial / MISMATCH, comparing the station with the polygon name."""
    poly = norm(polygon)
    parts = name_parts(station)
    if poly in parts:
        return "exact"
    for p in parts:
        if not p:
            continue
        if p in poly or poly in p:
            return "partial"
        common = set(p.split()) & set(poly.split())
        if common:
            return "partial"
    return "MISMATCH"


def main() -> None:
    match = pd.read_csv(DER / "station_lake_match.csv")
    depths = pd.read_csv(DER / "mi_lake_depth.csv")
    lakes_tbl = pd.read_csv(DER / "training_lakes.csv")
    obs = pd.read_csv(DER / "observations.csv")
    lakes = gpd.read_file(ROOT / "data" / "raw" / "dnr" / "lakes_10ha.geojson")
    lakes = lakes.reset_index().rename(columns={"index": "lake_idx", "Name": "name"})
    lakes["key"] = lakes.name.map(norm)

    obs_by_key = obs.groupby("lake_key").size()

    rows = []
    for _, r in match.iterrows():
        p = Point(float(r.lon), float(r.lat))
        drow = depths[depths.lake_idx == r.lake_idx]
        drow = drow.iloc[0] if len(drow) else None

        # M1 test: nearest polygon carrying the station's own name.
        keys = set(name_parts(r["name"]))
        same_station = lakes[lakes.key.isin(keys)]
        station_km = (
            round(float(same_station.distance(p).min() * 111.32), 2)
            if len(same_station) else np.nan
        )

        # Duplicate-name risk on the *matched* polygon name.
        if pd.notna(r.lake_idx):
            same_matched = lakes[lakes.key == norm(r.lake_poly)]
            others = same_matched[same_matched.lake_idx != int(r.lake_idx)]
            margin = (
                round(float(others.distance(p).min() * 111.32), 2)
                if len(others) else np.inf
            )
        else:
            same_matched, margin = lakes.iloc[0:0], np.nan

        rows.append(dict(
            sid=int(r.sid), station=r["name"], matched_polygon=r.lake_poly,
            match_how=r.match_how, match_km=r.match_km,
            name_agreement=agreement(r["name"], r.lake_poly),
            station_name_polygons=len(same_station),
            station_name_nearest_km=station_km,
            matched_name_polygons=len(same_matched),
            duplicate_margin_km=margin,
            depth_m=round(float(drow.max_depth_m), 2) if drow is not None else np.nan,
            depth_source=drow.depth_source if drow is not None else "-",
            n_contours=int(drow.n_contours) if drow is not None else -1,
            obs=int(obs_by_key.get(f"g00803:{int(r.sid)}", 0)),
        ))
        rows[-1]["status"] = "in pool" if rows[-1]["obs"] else "dropped"

    check = pd.DataFrame(rows)
    check.to_csv(OUT / "station_name_check.csv", index=False)

    # Depth provenance for every Michigan inland lake in the pool.
    idx_of = match.set_index("sid")["lake_idx"]
    inland = lakes_tbl[lakes_tbl.region == "inland_mi"].copy()
    inland["sid"] = inland.lake_key.str.replace("g00803:", "", regex=False).astype(int)
    inland["lake_idx"] = inland.sid.map(idx_of)
    inland = inland.drop(columns=["max_depth_m", "mean_depth_m", "depth_source"])
    inland = inland.merge(
        depths[["lake_idx", "area_km2", "max_depth_ft", "max_depth_m", "mean_depth_m",
                "depth_source", "n_contours", "depth_imputed", "max_depth_iqr_m"]],
        on="lake_idx", how="left",
    )
    inland.to_csv(OUT / "inland_depth_provenance.csv", index=False)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_colwidth", 40)
    print("=== inland station / polygon name and coordinate audit ===")
    print(check.to_string(index=False))
    print()
    print("name agreement:", check.name_agreement.value_counts().to_dict())
    print("matches with a same-name polygon closer than the 0.9 km coordinate "
          "uncertainty:", end=" ")
    risky = check[check.duplicate_margin_km < 0.9]
    print("none" if risky.empty else "")
    if not risky.empty:
        print(risky.to_string(index=False))
    print()
    print("station-name polygon farther than the matched one (M1 pattern):")
    flagged = check[(check.name_agreement != "exact")
                    & (check.station_name_nearest_km.notna())
                    & (check.station_name_nearest_km > check.match_km + 0.05)]
    print(flagged[["sid", "station", "matched_polygon", "match_km",
                   "station_name_nearest_km", "name_agreement", "obs"]]
          .to_string(index=False) if len(flagged) else "  none")
    print()
    print("=== Michigan inland depth provenance ===")
    cols = ["lake_name", "sid", "lake_idx", "n_obs", "area_km2", "max_depth_ft",
            "max_depth_m", "depth_source", "n_contours", "max_depth_iqr_m", "split"]
    print(inland[cols].sort_values(["split", "max_depth_m"]).to_string(index=False))


if __name__ == "__main__":
    main()
