"""Build positive degree-day (P) fields from ERA5-Land 4-hourly t2m.

P is the cumulative sum of positive daily-mean temperature from 1 March, used
only inside the melt window 1 March - 31 July. Outside that window P is zero:
before 1 March there is no melt yet, and after 31 July the next ice season has
not started. This keeps the melt term from acting on October-December growth.

Layout matches the CFDD files exactly: one file per calendar year, the same
3-day dates, 81 x 171 at 0.1 deg, latitude descending 49.5 -> 41.5, longitude
ascending -93 -> -76, uint16. Files go to a new folder, `data/derived/pdd/`;
`data/derived/era5/` is only read, never modified.

This is the 1966 pilot. The raw CDS download is a ZIP container despite the
`.nc` suffix; it is unpacked to a temporary directory and removed afterwards.

Outputs
  data/derived/pdd/pdd_1966.npz
  data/derived/diagnostics/pdd_pilot_1966.txt
  data/derived/diagnostics/pdd_obs_1966.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_pdd.py
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import zipfile

import netCDF4
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5"
CFDD_DIR = ROOT / "data" / "derived" / "era5"
OUT_DIR = ROOT / "data" / "derived" / "pdd"
DIAG = ROOT / "data" / "derived" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

YEAR = 1966
MELT_START = (3, 1)
MELT_END = (7, 31)

GHCN = ROOT / "data" / "raw" / "ghcn"
GHCN_STATIONS = {
    "Duluth MN": ("USW00014913", 46.8369, -92.2097),
    "Marquette MI": ("USW00014838", 46.5458, -87.3794),
    "Ann Arbor MI": ("USC00200230", 42.2981, -83.6639),
}


def unpack(year: int) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Return a readable NetCDF path and a temp directory to clean up."""
    path = RAW / f"year_{year}_4h.nc"
    if not path.exists():
        raise SystemExit(f"missing raw file: {path}")
    if not zipfile.is_zipfile(path):
        return path, None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"pdd_{year}_"))
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        if len(members) != 1:
            raise SystemExit(f"{path}: expected one .nc member, found {members}")
        z.extract(members[0], tmp)
    return tmp / members[0], tmp


def daily_mean(nc_path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Daily mean temperature in degC, ISO date labels, lat, lon."""
    ds = netCDF4.Dataset(nc_path)
    try:
        vt = ds.variables["valid_time"]
        when = netCDF4.num2date(vt[:], vt.units, only_use_cftime_datetimes=False)
        t2m = np.asarray(ds.variables["t2m"][:], dtype=np.float64) - 273.15
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
    finally:
        ds.close()

    labels = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in when])
    uniq = np.unique(labels)
    daily = np.empty((len(uniq),) + t2m.shape[1:], dtype=np.float64)
    for i, label in enumerate(uniq):
        daily[i] = t2m[labels == label].mean(axis=0)
    return daily, uniq, lat, lon


def pdd_fields(daily: np.ndarray, labels: np.ndarray, year: int,
               store_dates: np.ndarray) -> np.ndarray:
    """Cumulative positive degree-days since 1 March, sampled on store_dates."""
    days = pd.to_datetime(pd.Series(labels))
    start = pd.Timestamp(year=year, month=MELT_START[0], day=MELT_START[1])
    end = pd.Timestamp(year=year, month=MELT_END[0], day=MELT_END[1])

    cum = np.cumsum(np.maximum(daily, 0.0), axis=0)
    before = np.flatnonzero(days < start)
    base = cum[before[-1]] if len(before) else 0.0
    inside = (days >= start) & (days <= end)
    growing = np.where(inside.to_numpy()[:, None, None], cum - base, 0.0)

    index = {label: i for i, label in enumerate(labels)}
    out = np.zeros((len(store_dates),) + daily.shape[1:], dtype=np.float64)
    for k, label in enumerate(store_dates):
        if label in index and start <= pd.Timestamp(str(label)) <= end:
            out[k] = growing[index[label]]
    return out


def sample(folder: pathlib.Path, prefix: str, lon: float, lat: float,
           when: np.datetime64) -> float:
    """Bilinear in space, linear in time between stored steps."""
    year = int(str(when)[:4])
    path = folder / f"{prefix}_{year}.npz"
    if not path.exists():
        return np.nan
    z = np.load(path)
    fld = z[prefix].astype(np.float64)
    dates = np.array([np.datetime64(s) for s in z["dates"]])
    latv, lonv = z["lat"].astype(float), z["lon"].astype(float)

    when = np.datetime64(when, "D")
    hi = int(np.searchsorted(dates, when, side="left"))
    if hi < len(dates) and dates[hi] == when:
        field = fld[hi]
    else:
        lo = hi - 1
        if lo < 0:
            field = fld[0]
        elif hi >= len(dates):
            field = fld[-1]
        else:
            frac = (when - dates[lo]) / (dates[hi] - dates[lo])
            field = fld[lo] * (1 - frac) + fld[hi] * frac

    fi = (latv[0] - lat) / (latv[0] - latv[1])
    fj = (lon - lonv[0]) / (lonv[1] - lonv[0])
    fi = min(max(fi, 0.0), len(latv) - 1.001)
    fj = min(max(fj, 0.0), len(lonv) - 1.001)
    i0, j0 = int(fi), int(fj)
    di, dj = fi - i0, fj - j0
    top = field[i0, j0] * (1 - dj) + field[i0, j0 + 1] * dj
    bot = field[i0 + 1, j0] * (1 - dj) + field[i0 + 1, j0 + 1] * dj
    return float(top * (1 - di) + bot * di)


def ghcn_pdd(sid: str, year: int, m0: int = 3, m1: int = 5) -> tuple[float, int]:
    """Positive degree-days from daily mean (TMAX+TMIN)/2 over Mar-May."""
    path = GHCN / f"{sid}.dly"
    if not path.exists():
        raise SystemExit(f"missing cached GHCN file: {path}")
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
    for m in range(m0, m1 + 1):
        for d in range(1, 32):
            try:
                pd.Timestamp(year=year, month=m, day=d)
            except ValueError:
                continue
            rec = vals.get((m, d), {})
            if "TMAX" in rec and "TMIN" in rec:
                total += max(0.0, (rec["TMAX"] + rec["TMIN"]) / 2.0)
                n += 1
    return total, n


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    usage = shutil.disk_usage(ROOT)
    say(f"disk free: {usage.free / 1e9:.1f} GB (pilot needs < 0.1 GB; full pull")
    say("about 1.0 GB raw plus ~0.15 GB P)")
    say()

    nc_path, tmp = unpack(YEAR)
    try:
        daily, labels, lat, lon = daily_mean(nc_path)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    say(f"daily mean: {daily.shape[0]} days x {daily.shape[1]} x {daily.shape[2]}")
    say(f"date range: {labels[0]} to {labels[-1]}")
    say(f"temperature range: {daily.min():.1f} to {daily.max():.1f} degC")
    say()

    cfdd = np.load(CFDD_DIR / f"cfdd_{YEAR}.npz")
    store_dates = np.asarray(cfdd["dates"])
    if not np.allclose(lat, cfdd["lat"].astype(float)):
        raise SystemExit("latitude grid does not match CFDD")
    if not np.allclose(lon, cfdd["lon"].astype(float)):
        raise SystemExit("longitude grid does not match CFDD")
    say(f"grid matches cfdd_{YEAR}.npz exactly: {len(lat)} x {len(lon)}")
    say(f"stored steps: {len(store_dates)} ({store_dates[0]} to {store_dates[-1]})")
    say()

    pdd = pdd_fields(daily, labels, YEAR, store_dates)
    when = pd.to_datetime(pd.Series(store_dates))
    start = pd.Timestamp(year=YEAR, month=3, day=1)
    end = pd.Timestamp(year=YEAR, month=7, day=31)

    pre = (when < start).to_numpy()
    post = (when > end).to_numpy()
    say(f"P is zero before 1 March: {bool(np.all(pdd[pre] == 0))} "
        f"({int(pre.sum())} steps)")
    say(f"P is zero after 31 July:  {bool(np.all(pdd[post] == 0))} "
        f"({int(post.sum())} steps)")
    say(f"P max inside the window: {pdd.max():.0f} degC-day")
    say()

    out = OUT_DIR / f"pdd_{YEAR}.npz"
    np.savez_compressed(out, pdd=np.clip(np.rint(pdd), 0, 65535).astype(np.uint16),
                        dates=store_dates, lat=lat, lon=lon,
                        valid_from=np.array(["03-01"]))

    weeks = {"2026-03-01": "Mar 1", "2026-04-01": "Apr 1",
             "2026-05-01": "May 1", "2026-06-01": "Jun 1",
             "2026-07-31": "Jul 31"}
    rows = []
    for iso, label in weeks.items():
        iso = iso.replace("2026", str(YEAR))
        idx = int(np.argmin(np.abs(when - pd.Timestamp(iso)).to_numpy()))
        rows.append(dict(step=store_dates[idx],
                         P_mean=float(pdd[idx].mean()),
                         P_max=float(pdd[idx].max())))
    say("mean / max P on representative steps:")
    say(pd.DataFrame(rows).round(1).to_string(index=False))
    say()

    # ---- observations -------------------------------------------------------
    obs = pd.read_csv(ROOT / "data" / "derived" / "observations.csv",
                      parse_dates=["date"])
    obs = obs[obs.date.dt.year == YEAR].copy()
    obs["cfdd"] = [
        sample(CFDD_DIR, "cfdd", float(lo), float(la), np.datetime64(d.date()))
        for lo, la, d in zip(obs.lon, obs.lat, obs.date)
    ]
    obs["pdd"] = [
        sample(OUT_DIR, "pdd", float(lo), float(la), np.datetime64(d.date()))
        for lo, la, d in zip(obs.lon, obs.lat, obs.date)
    ]
    keep = ["lake_key", "date", "region", "total_cm", "cfdd", "pdd", "split"]
    say(f"{YEAR} observations: {len(obs)}")
    say(obs[keep].round(1).to_string(index=False))
    say()

    # ---- GHCN March-May check ----------------------------------------------
    say("=== GHCN positive degree-days, 1 March - 31 May "
        f"{YEAR} (daily mean) ===")
    rows = []
    for label, (sid, la, lo) in GHCN_STATIONS.items():
        gh, n = ghcn_pdd(sid, YEAR)
        era = sample(OUT_DIR, "pdd", lo, la, np.datetime64(f"{YEAR}-05-31"))
        rows.append(dict(station=label, ghcn_pdd=round(gh, 0),
                         era5_pdd=round(float(era), 0),
                         diff_pct=round(100 * (era - gh) / gh, 1), days=n))
    say(pd.DataFrame(rows).to_string(index=False))
    say()
    say("GATE: 1966 P pilot complete. Sampler and GHCN check ran; the full")
    say("pull is not started. Awaiting approval to pull 1 Mar - 31 Jul")
    say("sequentially for 1966-2025, then 1950-1965.")

    out = OUT_DIR / f"pdd_{YEAR}.npz"
    obs[keep].to_csv(DIAG / f"pdd_obs_{YEAR}.csv", index=False)
    (DIAG / f"pdd_pilot_{YEAR}.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"wrote {DIAG / f'pdd_pilot_{YEAR}.txt'}")
    print(f"wrote {DIAG / f'pdd_obs_{YEAR}.csv'}")


if __name__ == "__main__":
    main()
