"""Download Michigan DNR hydrography subsets needed for the ice map.

Outputs (data/raw/dnr):
  lakes_10ha.geojson   lake/pond polygons >= 10 ha (EPSG:4326)
  deep_points.geojson  surveyed deep points with MaxDepth
  contour_keys.json    distinct lake keys present in the contour layer

The DNR service is flaky; every request retries with backoff.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dnr"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://services3.arcgis.com/Jdnp1TjADvSDxMAX/arcgis/rest/services"
    "/DNRHydrologyOPENDATA/FeatureServer"
)
LAKES = 3
DEEP_POINTS = 0
CONTOURS = 1


def get(url: str, attempts: int = 5) -> dict:
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ice-research"})
            with urllib.request.urlopen(req, timeout=120) as fh:
                return json.loads(fh.read())
        except Exception as exc:  # noqa: BLE001 - service returns assorted failures
            last = exc
            wait = 2 ** i
            print(f"    retry {i + 1}/{attempts} in {wait}s: {str(exc)[:90]}")
            time.sleep(wait)
    raise RuntimeError(f"request failed: {url}") from last


def paged_geojson(layer: int, where: str, out_fields: str, page: int = 1000) -> dict:
    features: list[dict] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "where": where,
                "outFields": out_fields,
                "outSR": "4326",
                "geometryPrecision": "6",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page,
                "orderByFields": "OBJECTID",
            }
        )
        data = get(f"{BASE}/{layer}/query?{query}")
        batch = data.get("features", [])
        features.extend(batch)
        print(f"  layer {layer}: offset {offset} -> {len(batch)} features")
        if not data.get("properties", {}).get("exceededTransferLimit") or not batch:
            break
        offset += len(batch)
    return {"type": "FeatureCollection", "features": features}


def distinct_keys(layer: int, field: str) -> list[str]:
    keys: set[str] = set()
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": field,
                "returnDistinctValues": "true",
                "returnGeometry": "false",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": field,
            }
        )
        data = get(f"{BASE}/{layer}/query?{query}")
        batch = data.get("features", [])
        keys |= {f["attributes"][field] for f in batch if f["attributes"].get(field)}
        print(f"  distinct {field}: offset {offset} -> {len(batch)} rows, {len(keys)} keys")
        if not data.get("exceededTransferLimit") or not batch:
            break
        offset += len(batch)
    return sorted(keys)


def main() -> None:
    print("lakes >= 10 ha")
    lakes = paged_geojson(LAKES, "Type=2 AND SqKM>=0.1", "Name,Name2,Acres,SqKM,Elevation")
    (OUT / "lakes_10ha.geojson").write_text(json.dumps(lakes), encoding="utf-8")
    print(f"  wrote {len(lakes['features'])} polygons")

    print("surveyed deep points")
    pts = paged_geojson(DEEP_POINTS, "1=1", "NewKey,MaxDepth,LakeName,County,Latitude,Longitude")
    (OUT / "deep_points.geojson").write_text(json.dumps(pts), encoding="utf-8")
    print(f"  wrote {len(pts['features'])} points")

    print("contour lake keys")
    keys = distinct_keys(CONTOURS, "NewKey")
    (OUT / "contour_keys.json").write_text(json.dumps(keys), encoding="utf-8")
    print(f"  wrote {len(keys)} keys")


if __name__ == "__main__":
    main()
