"""Fetch ERA5-Land 2 m temperature and build cumulative freezing degree-days.

CDS 'derived-era5-land-daily-statistics' is unusable at present (requests are
accepted and never started). The raw 'reanalysis-era5-land' dataset works, so
we pull four synoptic hours per day (00/06/12/18 UTC), average them into a
daily mean, and accumulate freezing degree-days from 1 October.

Outputs, per calendar year, in data/derived/era5:
  cfdd_YYYY.npz   uint16 freezing degree-days (degC-day), 3-day steps,
                  shape (steps, 81, 171), lat descending 49.5 -> 41.5
  carry_YYYY.npy  uint16 field: CFDD accumulated 1 Oct -> 31 Dec of that year
                  (the seed for the following calendar year)

CFDD resets on 1 October, so Jan-Sep of a file belongs to the season that began
the previous October, while Oct-Dec belongs to the new season.

The script is resumable: years whose cfdd file already exists are skipped, and
the Oct-Dec carry field makes restarting mid-series safe.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import time
import zipfile

import netCDF4
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "derived" / "era5"
RAW = ROOT / "data" / "raw" / "era5"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

AREA = [49.5, -93.0, 41.5, -76.0]
HOURS = ["00:00", "06:00", "12:00", "18:00"]
STEP_DAYS = 3
FIRST_YEAR = 1950
LAST_YEAR = 2025
SEASON_START_MONTH = 10


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (OUT / "fetch.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def retrieve(year: int) -> pathlib.Path:
    """Download one calendar year of 4-hourly 2 m temperature."""
    import cdsapi

    dest = RAW / f"era5_land_{year}.zip"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    client = cdsapi.Client(quiet=True, progress=False)
    for attempt in range(1, 6):
        try:
            client.retrieve(
                "reanalysis-era5-land",
                {
                    "variable": ["2m_temperature"],
                    "year": str(year),
                    "month": [f"{m:02d}" for m in range(1, 13)],
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": HOURS,
                    "area": AREA,
                    "format": "netcdf",
                },
                str(dest),
            )
            return dest
        except Exception as exc:  # noqa: BLE001 - CDS raises assorted errors
            log(f"  {year}: attempt {attempt} failed: {str(exc)[:160]}")
            if attempt == 5:
                raise
            time.sleep(30 * attempt)
    raise RuntimeError("unreachable")


def read_year(path: pathlib.Path) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Return daily-mean temperature in degC and the matching dates."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".nc")]
        if not names:
            raise RuntimeError(f"no netCDF inside {path}")
        blob = zf.read(names[0])
    ds = netCDF4.Dataset("inmem", memory=blob)
    try:
        var = ds.variables["t2m"]
        timevar = ds.variables["valid_time"]
        raw = np.array(var[:], dtype=np.float32)
        times = netCDF4.num2date(
            timevar[:], timevar.units, getattr(timevar, "calendar", "standard")
        )
        lat = np.array(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.array(ds.variables["longitude"][:], dtype=np.float64)
    finally:
        ds.close()

    if raw.ndim != 3:
        raise RuntimeError(f"unexpected t2m shape {raw.shape}")

    # Group by calendar date. ERA5-Land begins at 01:00 UTC on 1950-01-01, so the
    # first day carries three hours rather than four.
    dates: list = []
    buckets: list[list[int]] = []
    for i, t in enumerate(times):
        d = dt.date(t.year, t.month, t.day)
        if not dates or dates[-1] != d:
            dates.append(d)
            buckets.append([])
        buckets[-1].append(i)

    daily = np.empty((len(dates), *raw.shape[1:]), dtype=np.float32)
    for j, idx in enumerate(buckets):
        daily[j] = raw[idx].mean(axis=0)
    daily -= 273.15
    return daily, dates, lat, lon


def process(year: int) -> None:
    cfdd_path = OUT / f"cfdd_{year}.npz"
    if cfdd_path.exists():
        log(f"{year}: already done, skipping")
        return

    carry_path = OUT / f"carry_{year - 1}.npy"
    carry = np.load(carry_path) if carry_path.exists() else None

    t0 = time.time()
    log(f"{year}: downloading")
    raw_path = retrieve(year)
    daily, dates, lat, lon = read_year(raw_path)

    fdd = np.zeros_like(daily, dtype=np.float32)
    running = np.zeros(daily.shape[1:], dtype=np.float32) if carry is None else carry.astype(np.float32)
    for i, day in enumerate(dates):
        if day.month == SEASON_START_MONTH and day.day == 1:
            running[:] = 0.0
        running += np.maximum(0.0, -daily[i])
        fdd[i] = running

    # Carry for the next year = value at 31 December.
    dec31 = [i for i, d in enumerate(dates) if d.month == 12 and d.day == 31]
    carry_out = fdd[dec31[-1]] if dec31 else fdd[-1]
    np.save(OUT / f"carry_{year}.npy", np.clip(np.rint(carry_out), 0, 65535).astype(np.uint16))

    # Sample every STEP_DAYS, always including the final day.
    keep = list(range(0, len(dates), STEP_DAYS))
    if keep[-1] != len(dates) - 1:
        keep.append(len(dates) - 1)
    sampled = np.clip(np.rint(fdd[keep]), 0, 65535).astype(np.uint16)

    np.savez_compressed(
        cfdd_path,
        cfdd=sampled,
        dates=np.array([dates[i].isoformat() for i in keep]),
        lat=lat,
        lon=lon,
        valid_from=np.array(["1950-10-01"]),
    )
    raw_path.unlink(missing_ok=True)
    log(
        f"{year}: done in {time.time() - t0:.0f}s  "
        f"steps={sampled.shape[0]} max={int(sampled.max())} degC-day"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=FIRST_YEAR)
    ap.add_argument("--last", type=int, default=LAST_YEAR)
    args = ap.parse_args()
    for year in range(args.first, args.last + 1):
        process(year)


if __name__ == "__main__":
    main()
