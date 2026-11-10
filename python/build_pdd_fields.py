"""Build the production H-definition P field for each year as its raw file lands.

Definition H (selected by the pre-written 1966-1970 GHCN rule):
    P = cumulative sum from 1 March of mean over 24 hours of max(0, T)
zero outside 1 March - 31 July. Stored on the exact CFDD 3-day grid, uint16,
`data/derived/pdd/pdd_YYYY.npz`.

This script does not touch the pull. It watches `data/raw/era5/pdd_YYYY_hourly.zip`
and builds whatever is ready, in order, so P and the pull proceed together. It
is idempotent: an existing file carrying `definition = H` and the right dates is
skipped. The old 1966 pilot file was built with M4, so it is backed up to
`data/derived/backups/pre_pdd_H/` before being rebuilt.

After 1966-2025 are all present it runs `verify_pdd_fields.py` once. It then
continues with 1950-1965 and exits. No model fit is run.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_pdd_fields.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import time
import zipfile

import netCDF4
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "era5"
CFDD_DIR = ROOT / "data" / "derived" / "era5"
OUT = ROOT / "data" / "derived" / "pdd"
DIAG = ROOT / "data" / "derived" / "diagnostics"
BK = ROOT / "data" / "derived" / "backups" / "pre_pdd_H"
OUT.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)
BK.mkdir(parents=True, exist_ok=True)

ORDER = list(range(1966, 2026)) + list(range(1950, 1966))
RULE_YEARS = list(range(1966, 2026))
MIN_COMPLETE = 50_000_000
STABLE_SECONDS = 20
DEADLINE_HOURS = 26
LOG = DIAG / "pdd_build_log.txt"


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def raw_path(year: int) -> pathlib.Path:
    return RAW / f"pdd_{year}_hourly.zip"


def out_path(year: int) -> pathlib.Path:
    return OUT / f"pdd_{year}.npz"


def raw_ready(year: int) -> bool:
    p = raw_path(year)
    if not p.exists() or p.stat().st_size < MIN_COMPLETE:
        return False
    size = p.stat().st_size
    time.sleep(STABLE_SECONDS)
    return p.exists() and p.stat().st_size == size


def is_valid(year: int, store_dates: np.ndarray) -> bool:
    p = out_path(year)
    if not p.exists():
        return False
    try:
        z = np.load(p)
        return ("definition" in z and str(z["definition"][0]) == "H"
                and "pdd" in z and np.array_equal(z["dates"], store_dates)
                and z["pdd"].shape == (len(store_dates), 81, 171)
                and z["pdd"].dtype == np.uint16)
    except Exception:  # noqa: BLE001
        return False


def unpack(year: int) -> tuple[pathlib.Path, pathlib.Path | None]:
    path = raw_path(year)
    if not zipfile.is_zipfile(path):
        return path, None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"pdd_{year}_"))
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        if len(members) != 1:
            raise RuntimeError(f"{path}: expected one .nc member, got {members}")
        z.extract(members[0], tmp)
    return tmp / members[0], tmp


def build(year: int, store_dates: np.ndarray, lat: np.ndarray,
          lon: np.ndarray) -> bool:
    if is_valid(year, store_dates):
        return False
    nc_path, tmp = unpack(year)
    try:
        ds = netCDF4.Dataset(nc_path)
        try:
            vt = ds.variables["valid_time"]
            when = netCDF4.num2date(vt[:], vt.units, only_use_cftime_datetimes=False)
            t2m = np.asarray(ds.variables["t2m"][:], dtype=np.float32)
            flat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            flon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        finally:
            ds.close()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    if not np.allclose(flat, lat) or not np.allclose(flon, lon):
        raise RuntimeError(f"{year}: grid does not match CFDD")

    labels = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in when])
    uniq = np.unique(labels)
    positive = np.maximum(t2m - 273.15, 0.0)
    daily = np.stack([positive[labels == d].mean(axis=0) for d in uniq])
    days = pd.to_datetime(pd.Series(uniq))
    start = pd.Timestamp(year=year, month=3, day=1)
    end = pd.Timestamp(year=year, month=7, day=31)

    cum = np.cumsum(daily, axis=0, dtype=np.float64)
    before = np.flatnonzero(days < start)
    base = cum[before[-1]] if len(before) else 0.0
    inside = ((days >= start) & (days <= end)).to_numpy()
    growing = np.where(inside[:, None, None], cum - base, 0.0)

    index = {d: i for i, d in enumerate(uniq)}
    field = np.zeros((len(store_dates),) + daily.shape[1:], dtype=np.float64)
    for k, label in enumerate(store_dates):
        if label in index and start <= pd.Timestamp(str(label)) <= end:
            field[k] = growing[index[label]]

    dest = out_path(year)
    if dest.exists():  # stale definition; back it up before replacing
        shutil.copy2(dest, BK / dest.name)
        log(f"{year}: backed up stale file to {BK / dest.name}")
    np.savez_compressed(dest, pdd=np.clip(np.rint(field), 0, 65535).astype(np.uint16),
                        dates=store_dates, lat=lat, lon=lon,
                        valid_from=np.array(["03-01"]),
                        definition=np.array(["H"]))
    log(f"{year}: wrote {dest.name}, max {field.max():.0f} degC-day, "
        f"{dest.stat().st_size / 1e6:.2f} MB")
    return True


def main() -> None:
    log("P builder started (definition H)")
    started = time.time()
    pending = set(ORDER)
    verified = (DIAG / "pdd_verify_1966_2025.txt").exists()

    while pending:
        if (time.time() - started) / 3600 > DEADLINE_HOURS:
            log(f"deadline reached; pending: {sorted(pending)}")
            raise SystemExit(1)
        progressed = False
        for year in ORDER:
            if year not in pending:
                continue
            cfdd = np.load(CFDD_DIR / f"cfdd_{year}.npz")
            store_dates = np.asarray(cfdd["dates"])
            lat = cfdd["lat"].astype(float)
            lon = cfdd["lon"].astype(float)
            if is_valid(year, store_dates):
                pending.discard(year)
                continue
            if not raw_ready(year):
                continue
            try:
                changed = build(year, store_dates, lat, lon)
            except Exception as exc:  # noqa: BLE001
                log(f"{year}: build failed: {str(exc)[:300]}")
                continue
            if changed or out_path(year).exists():
                pending.discard(year)
                progressed = True

            if not verified and all(out_path(y).exists() for y in RULE_YEARS):
                log("1966-2025 complete; running E2 verification")
                import verify_pdd_fields
                verify_pdd_fields.main()
                verified = True
                log("verification written; gate. Continuing with 1950-1965.")
        if pending and not progressed:
            time.sleep(60)

    log(f"all years built: {len(ORDER)} files in {OUT}. Builder exiting.")
    if not verified:
        log("1966-2025 verification was not triggered; check for missing years")


if __name__ == "__main__":
    main()
