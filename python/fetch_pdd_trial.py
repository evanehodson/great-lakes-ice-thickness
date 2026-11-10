"""One sequential hourly ERA5-Land t2m request: 1 Mar - 31 Jul 1966.

This is a timing and size trial, not the bulk pull. It submits exactly one
request at all 24 hours and keeps the raw file on disk
(`data/raw/era5/pdd_trial_1966_hourly.zip`). Wall time, compressed size and
the request payload are written to
`data/derived/diagnostics/pdd_trial_1966_timing.txt` and `.json`.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/fetch_pdd_trial.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import shutil
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "era5"
DIAG = ROOT / "data" / "derived" / "diagnostics"
RAW.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

AREA = [49.5, -93.0, 41.5, -76.0]
YEAR = 1966
MONTHS = ["03", "04", "05", "06", "07"]
REQUEST = {
    "variable": ["2m_temperature"],
    "year": [str(YEAR)],
    "month": MONTHS,
    "day": [f"{d:02d}" for d in range(1, 32)],
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": AREA,
    "format": "netcdf",
}
DEST = RAW / f"pdd_trial_{YEAR}_hourly.zip"


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with (DIAG / "pdd_trial_1966_timing.txt").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    free = shutil.disk_usage(ROOT).free
    log(f"disk free before trial: {free / 1e9:.1f} GB")
    if free < 2e9:
        raise SystemExit("less than 2 GB free; refusing to start")

    import cdsapi

    client = cdsapi.Client(quiet=True, progress=False)
    t0 = time.time()
    size = 0
    for attempt in range(1, 4):
        try:
            log(f"submitting one request, attempt {attempt}")
            client.retrieve("reanalysis-era5-land", REQUEST, str(DEST))
            size = DEST.stat().st_size
            break
        except Exception as exc:  # noqa: BLE001 - CDS raises assorted errors
            log(f"attempt {attempt} failed: {str(exc)[:240]}")
            if attempt == 3:
                raise
            time.sleep(30 * attempt)

    elapsed = time.time() - t0
    log(f"done in {elapsed:.0f}s; file {size / 1e6:.1f} MB at {DEST.name}")
    log(f"linear estimate for 76 years: {76 * elapsed / 3600:.1f} h, "
        f"{76 * size / 1e9:.2f} GB")

    (DIAG / "pdd_trial_1966_timing.json").write_text(
        json.dumps(dict(request=REQUEST, elapsed_s=round(elapsed, 1),
                        bytes=size, path=str(DEST),
                        completed=dt.datetime.now().isoformat(timespec="seconds")),
                   indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
