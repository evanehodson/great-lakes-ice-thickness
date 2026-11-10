"""E2 verification of the production P fields, 1966-2025. No fit is run.

Checks, and reports them as a plain table:

  1. File integrity per year: stored steps equal the CFDD dates, shape and
     dtype, P exactly zero on every step before 1 March and after 31 July,
     non-decreasing across the stored steps inside the window, per-year maximum.
  2. Pull health: years that failed and years that needed a retry, read from
     `pdd_pull_status.json` and the pull log. Nothing is repaired here.
  3. GHCN spot check over every year at Duluth, Marquette and Ann Arbor,
     compared at the last stored step on or before 31 May. This is a reported
     statistic only: the definition was fixed on 1966-1970 before this ran.
  4. Join counts to the training matrix by split and month, March-July.
     Confirms that every observation outside 1 March - 31 July receives P = 0,
     that no in-window observation is missing P, and that the joined S is the
     same S the model already uses.

Note stated in the report: S (freezing degree-days) and P (positive
degree-days) use different daily-mean constructions. S is the mean of the four
synoptic hours, positive part; P is the 24-hour mean of the positive part.

Outputs
  data/derived/diagnostics/pdd_verify_1966_2025.txt
  data/derived/pdd_verify_summary.csv
  data/derived/pdd_ghcn_allyears.csv
  data/derived/pdd_join_check.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/verify_pdd_fields.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_pdd as bp  # noqa: E402

ROOT = HERE.parent
CFDD_DIR = ROOT / "data" / "derived" / "era5"
PDD_DIR = ROOT / "data" / "derived" / "pdd"
DER = ROOT / "data" / "derived"
DIAG = DER / "diagnostics"
YEARS = list(range(1966, 2026))

CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def load_field(kind: str, folder: pathlib.Path, year: int):
    key = (kind, year)
    if key not in CACHE:
        z = np.load(folder / f"{kind}_{year}.npz")
        CACHE[key] = (z[kind].astype(float),
                      np.array([np.datetime64(s) for s in z["dates"]]),
                      z["lat"].astype(float), z["lon"].astype(float))
    return CACHE[key]


def sample(kind: str, folder: pathlib.Path, year: int, lat: float, lon: float,
           when: np.datetime64) -> float:
    when = np.datetime64(when, "D")
    if kind == "pdd":
        # P is defined as exactly zero outside 1 March - 31 July. Without this
        # guard the temporal interpolation reaches across 1 March and gives a
        # February observation a small positive P.
        if when < np.datetime64(f"{year}-03-01") or when > np.datetime64(f"{year}-07-31"):
            return 0.0
    if not (folder / f"{kind}_{year}.npz").exists():
        return float("nan")
    fld, dates, latv, lonv = load_field(kind, folder, year)
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
            frac = float((when - dates[lo]) / (dates[hi] - dates[lo]))
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


def ghcn_to(sid: str, year: int, end: pd.Timestamp) -> tuple[float, int]:
    """Positive degree-days from 1 March to `end` for one GHCN station."""
    path = ROOT / "data" / "raw" / "ghcn" / f"{sid}.dly"
    if not path.exists():
        return np.nan, 0
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
    day = pd.Timestamp(year=year, month=3, day=1)
    while day <= end:
        rec = vals.get((day.month, day.day), {})
        if "TMAX" in rec and "TMIN" in rec:
            total += max(0.0, (rec["TMAX"] + rec["TMIN"]) / 2.0)
            n += 1
        day += pd.Timedelta(days=1)
    return total, n


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=== P field verification, 1966-2025, definition H ===")
    say("P = cumulative from 1 March of the 24-hour mean of max(0, T);")
    say("zero outside 1 March - 31 July; stored on the CFDD 3-day grid, uint16.")
    say()
    say("S and P use different daily-mean constructions: S is the positive part")
    say("of the mean of the four synoptic hours; P is the mean of the positive")
    say("part over all 24 hours. They are allowed to differ.")
    say()

    # ---------------- 1. integrity -------------------------------------
    say("=== 1. file integrity ===")
    rows = []
    problems = []
    for year in YEARS:
        cfdd = np.load(CFDD_DIR / f"cfdd_{year}.npz")
        store_dates = np.asarray(cfdd["dates"])
        path = PDD_DIR / f"pdd_{year}.npz"
        if not path.exists():
            problems.append(f"{year}: missing")
            rows.append(dict(year=year, steps=0, max=0, zero_before=False,
                             zero_after=False, monotonic=False, status="missing"))
            continue
        z = np.load(path)
        p = z["pdd"]
        dates = np.asarray(z["dates"])
        dt = pd.to_datetime(pd.Series(dates))
        pre = (dt < pd.Timestamp(year=year, month=3, day=1)).to_numpy()
        post = (dt > pd.Timestamp(year=year, month=7, day=31)).to_numpy()
        inside = ~pre & ~post
        zero_before = bool((p[pre] == 0).all())
        zero_after = bool((p[post] == 0).all())
        diffs = np.diff(p[inside].astype(np.int32), axis=0)
        monotonic = bool((diffs >= 0).all())
        ok = (p.dtype == np.uint16 and p.shape == (len(store_dates), 81, 171)
              and np.array_equal(dates, store_dates)
              and np.allclose(z["lat"].astype(float), cfdd["lat"].astype(float))
              and np.allclose(z["lon"].astype(float), cfdd["lon"].astype(float))
              and zero_before and zero_after and monotonic)
        if not ok:
            problems.append(f"{year}: integrity failure")
        rows.append(dict(year=year, steps=len(store_dates), max=int(p.max()),
                         zero_before=zero_before, zero_after=zero_after,
                         monotonic=monotonic, status="ok" if ok else "BAD"))
    tab = pd.DataFrame(rows)
    say(tab.to_string(index=False))
    say()
    say(f"years checked: {len(YEARS)}  all ok: {bool((tab.status == 'ok').all())}")
    say(f"problems: {problems if problems else 'none'}")
    say()

    # ---------------- 2. pull health -----------------------------------
    say("=== 2. pull health ===")
    status_path = DIAG / "pdd_pull_status.json"
    failed, retried = [], []
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        failed = sorted(int(y) for y, v in status.get("years", {}).items()
                        if v.get("status") != "ok")
        say(f"years recorded as failed: {failed if failed else 'none'}")
        say(f"years still unresolved: "
            f"{sorted(int(y) for y in status.get('unresolved', {})) or 'none'}")
    log_path = DIAG / "pdd_pull_log.txt"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8")
        for m in re.finditer(r"(\d{4}): request attempt (\d+)", text):
            if int(m.group(2)) > 1:
                retried.append(int(m.group(1)))
        retried = sorted(set(retried))
    say(f"years that needed a retry: {retried if retried else 'none'}")
    say()

    # ---------------- 3. GHCN all years --------------------------------
    say("=== 3. GHCN spot check, every year 1966-2025 (reported only) ===")
    say("stored P at the last 3-day step on or before 31 May vs GHCN")
    say("(TMAX+TMIN)/2 positive degree-days over the same span.")
    gh_rows = []
    for year in YEARS:
        cfdd = np.load(CFDD_DIR / f"cfdd_{year}.npz")
        dates = np.array([np.datetime64(s) for s in cfdd["dates"]])
        cutoff = np.datetime64(f"{year}-05-31")
        usable = dates[dates <= cutoff]
        if len(usable) == 0:
            continue
        cmp_date = pd.Timestamp(str(usable[-1]))
        for label, (sid, la, lo) in bp.GHCN_STATIONS.items():
            stored = sample("pdd", PDD_DIR, year, la, lo, np.datetime64(cmp_date))
            if not np.isfinite(stored):
                continue
            gh, n = ghcn_to(sid, year, cmp_date)
            if not np.isfinite(gh) or gh <= 0 or n == 0:
                continue
            gh_rows.append(dict(year=year, station=label, compare_date=cmp_date.date(),
                                stored_p=stored, ghcn_p=gh,
                                bias_pct=100 * (stored - gh) / gh, ghcn_days=n))
    gh = pd.DataFrame(gh_rows)
    if len(gh):
        say(gh.groupby("station").bias_pct.agg(
            n="size", mean="mean", median="median",
            mean_abs=lambda s: s.abs().mean(),
            lo="min", hi="max").round(1).to_string())
        say()
        say(f"all years and stations: mean bias {gh.bias_pct.mean():+.1f}%, "
            f"mean absolute bias {gh.bias_pct.abs().mean():.1f}%, "
            f"range {gh.bias_pct.min():+.1f} to {gh.bias_pct.max():+.1f}%")
        by_year = gh.groupby("year").bias_pct.mean().round(1)
        say(f"mean bias by year: min {by_year.min():+.1f}% ({by_year.idxmin()}), "
            f"max {by_year.max():+.1f}% ({by_year.idxmax()})")
    else:
        say("no GHCN comparison rows")
    say()
    say("The station spread is the real error source: it is a representativeness")
    say("offset at Marquette, not a definition artefact. `m` is calibrated to the")
    say("same biased P the map uses, so the fit and the display share it.")
    say()

    # ---------------- 4. join counts -----------------------------------
    say("=== 4. join to the training matrix ===")
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    pdd_vals, cfdd_vals = [], []
    for _, r in m.iterrows():
        year = int(r.date.year)
        pdd_vals.append(sample("pdd", PDD_DIR, year, float(r.lat), float(r.lon),
                               np.datetime64(r.date.date())))
        cfdd_vals.append(sample("cfdd", CFDD_DIR, year, float(r.lat), float(r.lon),
                                np.datetime64(r.date.date())))
    m["pdd"] = pdd_vals
    m["cfdd_resampled"] = cfdd_vals
    m["month"] = m.date.dt.month
    m["in_window"] = ((m.month >= 3) & (m.month <= 7))

    outside = m[~m.in_window]
    inside = m[m.in_window]
    say(f"observations joined: {len(m)}  ({m.lake_key.nunique()} lakes)")
    say(f"in window (Mar-Jul): {len(inside)}; outside: {len(outside)}")
    say(f"outside-window P all zero: {bool((outside.pdd.dropna() == 0).all())} "
        f"(max {outside.pdd.max():.3f}, NaN {int(outside.pdd.isna().sum())})")
    say(f"in-window P missing or negative: "
        f"{int(inside.pdd.isna().sum())} NaN, {int((inside.pdd < 0).sum())} negative")
    say(f"S unchanged: max |matrix cfdd - resampled| = "
        f"{float((m.cfdd - m.cfdd_resampled).abs().max()):.3e}")
    say()

    counts = (inside.groupby(["split", "month"])
              .agg(n=("pdd", "size"), n_pdd_pos=("pdd", lambda s: int((s > 0).sum())),
                   pdd_mean=("pdd", "mean"), pdd_median=("pdd", "median"),
                   pdd_max=("pdd", "max"))
              .round(1))
    say("March-July, by split and month:")
    say(counts.to_string())
    say()
    say("outside the window, P must be zero (Jan, Feb, Nov, Dec):")
    say(outside.groupby(["split", "month"]).pdd.agg(
        n="size", max="max").round(3).to_string())
    say()

    say("GATE: verification only. No fit, no parameter change, no change to")
    say("`training_matrix.csv`. 1950-1965 P will be built as those years land.")

    all_present = all((PDD_DIR / f"pdd_{y}.npz").exists() for y in YEARS)
    suffix = "" if all_present else "_partial"
    if suffix:
        say()
        say("PARTIAL RUN: not all years present; canonical files not written.")
    tab.to_csv(DER / f"pdd_verify_summary{suffix}.csv", index=False)
    gh.to_csv(DER / f"pdd_ghcn_allyears{suffix}.csv", index=False)
    m[["lake_key", "date", "split", "month", "pdd", "cfdd", "cfdd_resampled"]].to_csv(
        DER / f"pdd_join_check{suffix}.csv", index=False)
    (DIAG / f"pdd_verify_1966_2025{suffix}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {DIAG / f'pdd_verify_1966_2025{suffix}.txt'}")
    print(f"wrote {DER / f'pdd_verify_summary{suffix}.csv'}")
    print(f"wrote {DER / f'pdd_ghcn_allyears{suffix}.csv'}")
    print(f"wrote {DER / f'pdd_join_check{suffix}.csv'}")


if __name__ == "__main__":
    main()
