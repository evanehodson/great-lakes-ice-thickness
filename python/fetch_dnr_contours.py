"""Download Michigan DNR inland lake depth contours and summarize them.

Contours carry the same NewKey as the deep points and, unlike the points,
they sit strictly inside each lake polygon. Grouping contour vertices by
NewKey therefore gives a reliable way to attach the DNR lake key (and its
MaxDepth) to a DNR hydro polygon.

Outputs (data/raw/dnr):
  contours.geojson     all contour polylines (EPSG:4326)
  contour_index.csv    one row per NewKey: n_lines, min/max depth, name
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dnr"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://services3.arcgis.com/Jdnp1TjADvSDxMAX/arcgis/rest/services"
    "/DNRHydrologyOPENDATA/FeatureServer"
)


def get(url: str, attempts: int = 5) -> dict:
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ice-research"})
            with urllib.request.urlopen(req, timeout=180) as fh:
                return json.loads(fh.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 2 ** i
            print(f"    retry {i + 1}/{attempts} in {wait}s: {str(exc)[:90]}")
            time.sleep(wait)
    raise RuntimeError(f"request failed: {url}") from last


def main() -> None:
    features: list[dict] = []
    offset = 0
    page = 1000
    while True:
        query = urllib.parse.urlencode(
            {
                "where": "1=1",
                "outFields": "NewKey,MinDepth,MaxDepth,Name",
                "outSR": "4326",
                "geometryPrecision": "6",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": page,
                "orderByFields": "OBJECTID",
            }
        )
        data = get(f"{BASE}/1/query?{query}")
        batch = data.get("features", [])
        features.extend(batch)
        print(f"  offset {offset} -> {len(batch)} lines (total {len(features)})")
        if not data.get("properties", {}).get("exceededTransferLimit") or not batch:
            break
        offset += len(batch)

    (OUT / "contours.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )

    rows = [
        {
            "NewKey": f["properties"].get("NewKey"),
            "MinDepth": f["properties"].get("MinDepth"),
            "MaxDepth": f["properties"].get("MaxDepth"),
            "Name": f["properties"].get("Name"),
        }
        for f in features
    ]
    df = pd.DataFrame(rows)
    summary = (
        df.dropna(subset=["NewKey"])
        .groupby("NewKey")
        .agg(
            n_lines=("NewKey", "size"),
            min_depth=("MinDepth", "min"),
            max_depth=("MaxDepth", "max"),
            name=("Name", "first"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "contour_index.csv", index=False)
    print(f"wrote {len(features)} lines, {len(summary)} distinct lakes")


if __name__ == "__main__":
    main()
