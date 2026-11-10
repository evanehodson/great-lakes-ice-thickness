"""Sample ERA5-Land CFDD at every ice observation and build the training matrix.

CFDD layout (see fetch_era5.py): one file per calendar year, 3-day steps,
81 x 171 at 0.1 deg, latitude descending 49.5 -> 41.5, longitude ascending
-93 -> -76. January-September of a file continues the season that began the
previous 1 October; October-December belongs to the new season.

Sampling is bilinear in space and linear in time between the stored 3-day
steps. Both are interpolation of the same field, so the value returned is the
CFDD at the observation date, not the nearest stored step.

2026: no ERA5 file yet. Those observations (all January-April 2026) are removed
from the training matrix and listed in
`data/derived/diagnostics/cfdd_missing_2026.csv` per decision D2.

The 1952 GHCN spot check compares the sampled field with freezing degree-days
computed from daily GHCN TMAX/TMIN at Duluth, Marquette and Ann Arbor. GHCN
files are cached in data/raw/ghcn; the check downloads only if missing.

Outputs
  data/derived/training_matrix.csv
  data/derived/diagnostics/cfdd_missing_2026.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/sample_cfdd.py
"""

from __future__ import annotations

import datetime as dt
import pathlib
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
ERA5 = DER / "era5"
GHCN = ROOT / "data" / "raw" / "ghcn"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

GHCN_STATIONS = {
    "Duluth MN": ("USW00014913", 46.8369, -92.2097),
    "Marquette MI": ("USW00014838", 46.5458, -87.3794),
    "Ann Arbor MI": ("USC00200230", 42.2981, -83.6639),
}
GHCN_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{sid}.dly"


class CFDD:
    def __init__(self, folder: pathlib.Path):
        self.folder = folder
        self.cache: dict[int, dict] = {}

    def year(self, year: int) -> dict | None:
        if year in self.cache:
            return self.cache[year]
        path = self.folder / f"cfdd_{year}.npz"
        if not path.exists():
            self.cache[year] = None
            return None
        z = np.load(path)
        rec = dict(
            cfdd=z["cfdd"].astype(np.float64),
            dates=np.array([np.datetime64(s) for s in z["dates"]]),
            lat=z["lat"].astype(np.float64),
            lon=z["lon"].astype(np.float64),
        )
        self.cache[year] = rec
        return rec

    def field(self, year: int, when: np.datetime64) -> np.ndarray | None:
        """CFDD field interpolated linearly in time to ``when``."""
        rec = self.year(year)
        if rec is None:
            return None
        dates = rec["dates"]
        when = np.datetime64(when, "D")
        hi = int(np.searchsorted(dates, when, side="left"))
        if hi < len(dates) and dates[hi] == when:
            return rec["cfdd"][hi]
        lo = hi - 1
        if lo < 0:
            return rec["cfdd"][0]
        if hi >= len(dates):
            return rec["cfdd"][-1]
        frac = (when - dates[lo]) / (dates[hi] - dates[lo])
        return rec["cfdd"][lo] * (1.0 - frac) + rec["cfdd"][hi] * frac

    def sample(self, lon: float, lat: float, when: np.datetime64) -> float:
        rec = self.year(int(str(when)[:4]))
        if rec is None:
            return np.nan
        fld = self.field(int(str(when)[:4]), when)
        if fld is None:
            return np.nan
        latv, lonv = rec["lat"], rec["lon"]

        # Latitude descends, longitude ascends.
        fi = (latv[0] - lat) / (latv[0] - latv[1])
        fj = (lon - lonv[0]) / (lonv[1] - lonv[0])
        fi = min(max(fi, 0.0), len(latv) - 1.001)
        fj = min(max(fj, 0.0), len(lonv) - 1.001)
        i0, j0 = int(fi), int(fj)
        di, dj = fi - i0, fj - j0
        v00, v01 = fld[i0, j0], fld[i0, j0 + 1]
        v10, v11 = fld[i0 + 1, j0], fld[i0 + 1, j0 + 1]
        top = v00 * (1 - dj) + v01 * dj
        bot = v10 * (1 - dj) + v11 * dj
        return float(top * (1 - di) + bot * di)


def ghcn_fdd(sid: str, year: int = 1952) -> tuple[float, int]:
    """Freezing degree-days from daily mean (TMAX+TMIN)/2, 1 Oct - 31 Dec."""
    GHCN.mkdir(parents=True, exist_ok=True)
    path = GHCN / f"{sid}.dly"
    if not path.exists():
        url = GHCN_URL.format(sid=sid)
        print(f"  downloading {url}")
        with urllib.request.urlopen(url, timeout=180) as r, open(path, "wb") as fh:
            fh.write(r.read())
    vals: dict[tuple[int, int], dict[str, float]] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        y, m, el = int(line[11:15]), int(line[15:17]), line[17:21]
        if y != year or el not in ("TMAX", "TMIN"):
            continue
        for d in range(1, 32):
            off = 21 + (d - 1) * 8
            raw = line[off:off + 5]
            if raw.strip() in ("-9999", ""):
                continue
            vals.setdefault((m, d), {})[el] = int(raw) / 10.0
    total, n = 0.0, 0
    for m in range(10, 13):
        for d in range(1, 32):
            try:
                dt.date(year, m, d)
            except ValueError:
                continue
            rec = vals.get((m, d), {})
            if "TMAX" in rec and "TMIN" in rec:
                total += max(0.0, -(rec["TMAX"] + rec["TMIN"]) / 2.0)
                n += 1
    return total, n


def main() -> None:
    obs = pd.read_csv(DER / "observations.csv", parse_dates=["date"])
    depths = pd.read_csv(DER / "depth_variants.csv")
    cfdd = CFDD(ERA5)

    obs["cfdd"] = [
        cfdd.sample(float(lo), float(la), np.datetime64(d.date()))
        for lo, la, d in zip(obs.lon, obs.lat, obs.date)
    ]
    obs["season"] = np.where(
        obs.date.dt.month >= 10, obs.date.dt.year, obs.date.dt.year - 1
    )
    obs["season_label"] = obs.season.astype(int).astype(str) + "-" + (
        (obs.season.astype(int) + 1) % 100
    ).astype(int).astype(str).str.zfill(2)

    obs = obs.merge(depths, on="lake_key", how="left", suffixes=("", "_variant"))
    if obs.V0.isna().any():
        bad = obs.loc[obs.V0.isna(), "lake_key"].unique()
        raise SystemExit(f"no depth for {len(bad)} lakes: {bad[:5]}")

    missing = obs[obs.cfdd.isna()].copy()
    missing.to_csv(OUT / "cfdd_missing_2026.csv", index=False)
    matrix = obs[obs.cfdd.notna()].copy()

    matrix.to_csv(DER / "training_matrix.csv", index=False)

    # --- report ---------------------------------------------------------
    pd.set_option("display.width", 240)
    print(f"observations: {len(obs):,}")
    print(f"CFDD sampled: {len(matrix):,}  missing: {len(missing)}")
    print("missing by year:", missing.date.dt.year.value_counts().to_dict())
    print("missing by source:", missing.source.value_counts().to_dict())
    print("missing by region:", missing.region.value_counts().to_dict())
    print()
    print("CFDD distribution (degC-day):")
    print(matrix.cfdd.describe(percentiles=[.01, .5, .99]).round(1).to_string())
    print()
    print("CFDD <= 0:", int((matrix.cfdd <= 0).sum()), "observations")
    print("CFDD by region:")
    print(matrix.groupby("region").cfdd.describe(percentiles=[.5])
          [["count", "mean", "min", "50%", "max"]].round(1).to_string())
    print()
    print("CFDD by month:")
    print(matrix.groupby(matrix.date.dt.month).cfdd.median().round(0).to_string())
    print()
    print("=== GHCN 1952 spot check (Oct-Dec freezing degree-days) ===")
    rows = []
    for label, (sid, la, lo) in GHCN_STATIONS.items():
        obs_fdd, n = ghcn_fdd(sid)
        era = cfdd.sample(lo, la, np.datetime64("1952-12-31"))
        rows.append(dict(station=label, ghcn_fdd=round(obs_fdd, 0),
                         era5_fdd=round(float(era), 0),
                         diff_pct=round(100 * (era - obs_fdd) / obs_fdd, 1),
                         days=n))
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("matrix columns:", list(matrix.columns))


if __name__ == "__main__":
    main()
