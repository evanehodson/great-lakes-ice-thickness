"""Sequential, resumable hourly ERA5-Land t2m pull for the melt term.

One CDS request per year, 1 March - 31 July, all 24 hours, on the exact CFDD
grid. Order:

  block 1  1966-1970   (needed for the P-definition rule)
  block 2  1971-2025
  block 3  1950-1965

Every raw file is kept (`data/raw/era5/pdd_YYYY_hourly.zip`). A year whose file
already exists and looks complete is skipped, so the script can be restarted at
any point. Each year is retried up to four times with backoff. A year that still
fails is recorded and the pull continues; failures are listed at the end and in
`data/derived/diagnostics/pdd_pull_unresolved.txt` for a decision. **No year is
ever turned into P = 0.**

When block 1 finishes, `apply_pdd_rule.py` runs automatically: it builds H, P4
and M4 for 1966-1970 and applies the pre-written GHCN rule. The pull then
continues. No production P field is derived here; that waits at the gate.

Status is written after every year to
`data/derived/diagnostics/pdd_pull_status.json`, so progress survives a crash.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/pull_pdd.py
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
MONTHS = ["03", "04", "05", "06", "07"]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]
MIN_COMPLETE = 50_000_000      # a complete year is ~95 MB; below this, re-pull
MIN_FREE_BYTES = 20_000_000_000
MAX_ATTEMPTS = 4

BLOCKS = [
    ("rule", list(range(1966, 1971))),
    ("main", list(range(1971, 2026))),
    ("backfill", list(range(1950, 1966))),
]

STATUS = DIAG / "pdd_pull_status.json"
LOG = DIAG / "pdd_pull_log.txt"
RULE_MARKER = DIAG / "pdd_rule_1966_1970_done.txt"


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def raw_path(year: int) -> pathlib.Path:
    return RAW / f"pdd_{year}_hourly.zip"


def load_status() -> dict:
    if STATUS.exists():
        try:
            return json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"years": {}, "unresolved": {}, "started": dt.datetime.now().isoformat(timespec="seconds")}


def save_status(status: dict) -> None:
    status["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def adopt_trial_file() -> None:
    """The 1966 timing trial is the year-1966 artifact; copy it to the name."""
    dest = raw_path(1966)
    trial = RAW / "pdd_trial_1966_hourly.zip"
    if not dest.exists() and trial.exists():
        shutil.copy2(trial, dest)
        log(f"copied trial file to {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def request_year(client, year: int) -> int:
    dest = raw_path(year)
    payload = {
        "variable": ["2m_temperature"],
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "area": AREA,
        "format": "netcdf",
    }
    t0 = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            log(f"{year}: request attempt {attempt}")
            client.retrieve("reanalysis-era5-land", payload, str(dest))
            size = dest.stat().st_size if dest.exists() else 0
            if size >= MIN_COMPLETE:
                log(f"{year}: done in {time.time() - t0:.0f}s, {size / 1e6:.1f} MB")
                return size
            log(f"{year}: file only {size} bytes, treating as incomplete")
        except Exception as exc:  # noqa: BLE001 - CDS raises assorted errors
            log(f"{year}: attempt {attempt} failed: {str(exc)[:300]}")
        if attempt < MAX_ATTEMPTS:
            time.sleep(60 * attempt)
    return 0


def main() -> None:
    free = shutil.disk_usage(ROOT).free
    log(f"disk free before pull: {free / 1e9:.1f} GB")
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"need at least {MIN_FREE_BYTES / 1e9:.0f} GB free")

    adopt_trial_file()
    status = load_status()
    order = [y for _, years in BLOCKS for y in years]

    for block, years in BLOCKS:
        missing = [y for y in years if not raw_path(y).exists()
                   or raw_path(y).stat().st_size < MIN_COMPLETE]
        done_already = len(years) - len(missing)
        log(f"block {block}: {len(years)} years, {done_already} already on disk, "
            f"{len(missing)} to fetch")
        if not missing:
            continue

        import cdsapi

        client = cdsapi.Client(quiet=True, progress=False)
        for year in missing:
            free = shutil.disk_usage(ROOT).free
            if free < MIN_FREE_BYTES:
                save_status(status)
                raise SystemExit(f"disk free fell to {free / 1e9:.1f} GB; stopping")
            size = request_year(client, year)
            if size:
                status["years"][str(year)] = {
                    "status": "ok", "bytes": size,
                    "completed": dt.datetime.now().isoformat(timespec="seconds"),
                }
                status["unresolved"].pop(str(year), None)
            else:
                status["years"][str(year)] = {
                    "status": "failed",
                    "completed": dt.datetime.now().isoformat(timespec="seconds"),
                }
                status["unresolved"][str(year)] = "all attempts failed"
                log(f"{year}: UNRESOLVED, continuing; listed for a decision")
            save_status(status)

        if block == "rule" and not RULE_MARKER.exists():
            ready = all(raw_path(y).exists() and raw_path(y).stat().st_size >= MIN_COMPLETE
                        for y in years)
            if ready:
                log("block rule complete; running the 1966-1970 P-definition rule")
                try:
                    import apply_pdd_rule
                    apply_pdd_rule.main()
                    RULE_MARKER.write_text(
                        dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
                    log("rule report written; production P derivation waits at the gate")
                except Exception as exc:  # noqa: BLE001
                    log(f"rule step failed: {str(exc)[:300]}; pull continues")
            else:
                log("block rule has unresolved years; rule step deferred")

    unresolved = sorted(int(y) for y in status["unresolved"])
    log(f"pull finished. on disk: {len(list(RAW.glob('pdd_*_hourly.zip')))}, "
        f"unresolved: {unresolved if unresolved else 'none'}")
    if unresolved:
        text = "\n".join(str(y) for y in unresolved) + "\n"
        (DIAG / "pdd_pull_unresolved.txt").write_text(
            "Years that failed every attempt. These are NOT set to P = 0; they "
            "need a decision.\n" + text, encoding="utf-8")
        log(f"wrote {DIAG / 'pdd_pull_unresolved.txt'}")


if __name__ == "__main__":
    main()
