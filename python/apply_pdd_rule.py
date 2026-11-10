"""Apply the pre-written P-definition rule over 1966-1970.

For each year in 1966-1970, read the hourly raw file and build cumulative P to
31 May three ways on the CFDD grid:

  H   hourly-integrated: sum over 24 h of max(0, T)/24
  P4  mean of positive parts over the four synoptic hours
  M4  positive part of the four-hour mean (the CFDD-consistent method)

Compare each against GHCN daily (TMAX+TMIN)/2 positive degree-days at Duluth,
Marquette and Ann Arbor, 1 March - 31 May.

Selection rule, fixed before any comparison was seen:
  primary    smallest mean absolute percent station bias over all year-station
             pairs
  tie-break  smallest mean per-year spread across stations (max - min)

Only temperature and GHCN enter. No ice observation and no held-out value is
read. This script stops at the rule: it does not derive the production P field
for 1950-2025.

Outputs
  data/derived/diagnostics/pdd_rule_1966_1970.txt
  data/derived/pdd_definition_rule_1966_1970.csv   (per year x station)
  data/derived/pdd_definition_rule_summary.csv     (per definition)

Run:  $env:PYTHONIOENCODING="utf-8"; python python/apply_pdd_rule.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys

import netCDF4
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_pdd as bp  # noqa: E402
from compare_pdd_definitions import bilinear, unpack  # noqa: E402

ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "era5"
DIAG = ROOT / "data" / "derived" / "diagnostics"
SYNOPTIC = (0, 6, 12, 18)
YEARS = list(range(1966, 1971))


def year_values(year: int) -> dict[str, float] | None:
    """P to 31 May at each GHCN station, for the three definitions."""
    path = RAW / f"pdd_{year}_hourly.zip"
    if not path.exists() or path.stat().st_size < 50_000_000:
        return None
    nc_path, tmp = unpack(path)
    try:
        ds = netCDF4.Dataset(nc_path)
        try:
            vt = ds.variables["valid_time"]
            when = netCDF4.num2date(vt[:], vt.units, only_use_cftime_datetimes=False)
            t2m = np.asarray(ds.variables["t2m"][:], dtype=np.float64) - 273.15
            lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
        finally:
            ds.close()
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    hours = np.array([t.hour for t in when])
    dates = np.array([f"{t.year:04d}-{t.month:02d}-{t.day:02d}" for t in when])
    uniq = np.unique(dates)
    syn = np.isin(hours, SYNOPTIC)

    daily = {
        "H": np.stack([t2m[dates == d].clip(min=0).sum(axis=0) / 24.0 for d in uniq]),
        "P4": np.stack([np.maximum(t2m[(dates == d) & syn], 0.0).mean(axis=0)
                       for d in uniq]),
        "M4": np.stack([np.maximum(t2m[(dates == d) & syn].mean(axis=0), 0.0)
                       for d in uniq]),
    }
    days = pd.to_datetime(pd.Series(uniq))
    start = pd.Timestamp(year=year, month=3, day=1)
    end = pd.Timestamp(year=year, month=5, day=31)
    inside = ((days >= start) & (days <= end)).to_numpy()

    out: dict[str, float] = {}
    for name, arr in daily.items():
        cum = np.cumsum(arr, axis=0)
        before = np.flatnonzero(days < start)
        base = cum[before[-1]] if len(before) else np.zeros(arr.shape[1:])
        growing = np.where(inside[:, None, None], cum - base, 0.0)
        at_end = int(np.flatnonzero(inside)[-1])
        for label, (sid, la, lo) in bp.GHCN_STATIONS.items():
            out[f"{label}|{name}"] = bilinear(growing[at_end], lat, lon, la, lo)
    return out


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=== P-definition rule, 1966-1970 ===")
    say("rule fixed in advance: smallest mean absolute percent station bias,")
    say("then smallest mean per-year spread across stations (max - min).")
    say("temperature and GHCN only; no ice observation, no held-out value.")
    say()

    rows = []
    missing = []
    for year in YEARS:
        vals = year_values(year)
        if vals is None:
            missing.append(year)
            continue
        for label, (sid, la, lo) in bp.GHCN_STATIONS.items():
            gh, n = bp.ghcn_pdd(sid, year)
            if not gh:
                continue
            for name in ("H", "P4", "M4"):
                v = vals[f"{label}|{name}"]
                rows.append(dict(year=year, station=label, definition=name,
                                 era5_p=v, ghcn_p=gh, days=n,
                                 bias_pct=100 * (v - gh) / gh))

    if missing:
        say(f"incomplete years, excluded: {missing}")
        say()
    if not rows:
        raise SystemExit("no complete year available; run the pull first")

    df = pd.DataFrame(rows)
    say("cumulative P to 31 May, degC-day, by year and station:")
    for year in sorted(df.year.unique()):
        g = df[df.year == year]
        say(f"\n{year}:")
        say(g.pivot(index="station", columns="definition", values="era5_p")
              .reindex(["Duluth MN", "Marquette MI", "Ann Arbor MI"])
              [["H", "P4", "M4"]].round(0).assign(
                  GHCN=g.groupby("station").ghcn_p.first()).to_string())
        say(g.pivot(index="station", columns="definition", values="bias_pct")
              .reindex(["Duluth MN", "Marquette MI", "Ann Arbor MI"])
              [["H", "P4", "M4"]].round(1).to_string())

    say()
    say("all year-station pairs, percent bias:")
    say(df.pivot_table(index=["year", "station"], columns="definition",
                       values="bias_pct").round(1).to_string())

    summary = []
    for name in ("H", "P4", "M4"):
        g = df[df.definition == name]
        per_year_spread = g.groupby("year").bias_pct.agg(lambda s: s.max() - s.min())
        summary.append(dict(
            definition=name,
            n_pairs=len(g),
            mean_abs_bias=float(g.bias_pct.abs().mean()),
            mean_bias=float(g.bias_pct.mean()),
            mean_year_spread=float(per_year_spread.mean()),
            pooled_spread=float(g.bias_pct.max() - g.bias_pct.min()),
            duluth_mean_abs=float(g[g.station == "Duluth MN"].bias_pct.abs().mean()),
            marquette_mean_abs=float(g[g.station == "Marquette MI"].bias_pct.abs().mean()),
            annarbor_mean_abs=float(g[g.station == "Ann Arbor MI"].bias_pct.abs().mean()),
        ))
    s = pd.DataFrame(summary).sort_values(["mean_abs_bias", "mean_year_spread"])
    say()
    say("ranking by the pre-written rule:")
    say(s.round(2).to_string(index=False))
    say()
    say("by station, mean absolute bias:")
    say(s[["definition", "duluth_mean_abs", "marquette_mean_abs",
           "annarbor_mean_abs"]].round(1).to_string(index=False))

    complete_years = sorted(int(y) for y in df.year.unique())
    suffix = "" if len(complete_years) == len(YEARS) else "_partial"
    winner = s.iloc[0].definition
    say()
    say(f"WINNER: {winner}" + (" (partial, 1966 only)" if suffix else ""))
    say()
    say("Read with care:")
    say("- Marquette's bias barely depends on the definition; it is a station/")
    say("  grid representativeness offset, not a daily-mean artefact.")
    say("- S (freezing degree-days) and P (positive degree-days) may use")
    say("  different daily-mean definitions; nothing requires them to match.")
    say("- The 1966 ranking rested on Duluth alone. This table is the actual")
    say("  decision.")
    say()
    say("GATE: the definition is fixed by the table above. Deriving the")
    say("production P field for 1950-2025 is a separate step and is not done.")

    if suffix:
        say(f"PARTIAL: only {complete_years} available; canonical rule files")
        say("are not written until all five years are present.")
    base = f"pdd_definition_rule_1966_1970{suffix}"
    df.to_csv(ROOT / "data" / "derived" / f"{base}.csv", index=False)
    s.to_csv(ROOT / "data" / "derived" / f"pdd_definition_rule_summary{suffix}.csv",
             index=False)
    (DIAG / f"pdd_rule_1966_1970{suffix}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {DIAG / (name if (name := f'pdd_rule_1966_1970{suffix}.txt') else '')}")
    print(f"wrote {ROOT / 'data' / 'derived' / (base + '.csv')}")


if __name__ == "__main__":
    main()
