"""Compare three positive-degree-day definitions against GHCN, 1966.

Reads the single hourly trial download (`pdd_trial_1966_hourly.zip`) and builds
P three ways on the exact CFDD grid:

  H   hourly-integrated   sum over 24 hours of max(0, T_h) / 24
  P4  mean of positive parts   mean over the 4 synoptic hours of max(0, T_h)
  M4  positive part of the 4-hour mean   max(0, mean over the 4 hours)

H and P4 never let a sub-zero morning cancel a thawing afternoon; M4, the
current CFDD-consistent method, does. All three are cumulative from 1 March and
zero outside 1 March - 31 July.

Comparison is against GHCN daily (TMAX+TMIN)/2 positive degree-days at Duluth,
Marquette and Ann Arbor, 1 March - 31 May 1966.

Selection rule, written before looking at the comparison: over 1966-1970 choose
the definition with the smallest mean absolute percent station bias; break ties
on the smallest spread (max - min) across stations. No ice observations and no
held-out data enter the choice.

The trial supplies 1966 only, so this script reports the 1966 comparison and
states what the full rule still needs.

Outputs
  data/derived/pdd_trial/pdd_1966_H.npz, _P4.npz, _M4.npz
  data/derived/diagnostics/pdd_definition_comparison_1966.txt
  data/derived/pdd_definition_comparison_1966.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/compare_pdd_definitions.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import zipfile

import netCDF4
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_pdd as bp  # noqa: E402

ROOT = HERE.parent
RAW = ROOT / "data" / "raw" / "era5"
CFDD_DIR = ROOT / "data" / "derived" / "era5"
OUT = ROOT / "data" / "derived" / "pdd_trial"
DIAG = ROOT / "data" / "derived" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

YEAR = 1966
TRIAL = RAW / f"pdd_trial_{YEAR}_hourly.zip"
SYNOPTIC = (0, 6, 12, 18)
START = pd.Timestamp(year=YEAR, month=3, day=1)
END = pd.Timestamp(year=YEAR, month=7, day=31)


def unpack(path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path | None]:
    if not path.exists():
        raise SystemExit(f"trial file not found: {path}")
    if not zipfile.is_zipfile(path):
        return path, None
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pdd_trial_"))
    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.endswith(".nc")]
        if len(members) != 1:
            raise SystemExit(f"{path}: expected one .nc member, got {members}")
        z.extract(members[0], tmp)
    return tmp / members[0], tmp


def bilinear(field: np.ndarray, lat: np.ndarray, lon: np.ndarray,
             la: float, lo: float) -> float:
    fi = (lat[0] - la) / (lat[0] - lat[1])
    fj = (lo - lon[0]) / (lon[1] - lon[0])
    fi = min(max(fi, 0.0), len(lat) - 1.001)
    fj = min(max(fj, 0.0), len(lon) - 1.001)
    i0, j0 = int(fi), int(fj)
    di, dj = fi - i0, fj - j0
    top = field[i0, j0] * (1 - dj) + field[i0, j0 + 1] * dj
    bot = field[i0 + 1, j0] * (1 - dj) + field[i0 + 1, j0 + 1] * dj
    return float(top * (1 - di) + bot * di)


def main() -> None:
    nc_path, tmp = unpack(TRIAL)
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
    say_lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        say_lines.append(text)

    say(f"trial file: {TRIAL.name} ({TRIAL.stat().st_size / 1e6:.1f} MB)")
    say(f"hourly field: {t2m.shape} = {len(uniq)} days x {len(hours) // len(uniq)} hours")
    say(f"hours present: {sorted(set(hours.tolist()))}")
    say(f"grid: {len(lat)} x {len(lon)}, lat {lat[0]:.1f} -> {lat[-1]:.1f}, "
        f"lon {lon[0]:.1f} -> {lon[-1]:.1f}")
    say()

    cfdd = np.load(CFDD_DIR / f"cfdd_{YEAR}.npz")
    store_dates = np.asarray(cfdd["dates"])
    if not np.allclose(lat, cfdd["lat"].astype(float)):
        raise SystemExit("latitude grid does not match CFDD")
    if not np.allclose(lon, cfdd["lon"].astype(float)):
        raise SystemExit("longitude grid does not match CFDD")
    say(f"grid matches cfdd_{YEAR}.npz exactly; {len(store_dates)} stored steps")
    say()

    syn = np.isin(hours, SYNOPTIC)
    daily = {}
    daily["H"] = np.stack([
        t2m[dates == d].clip(min=0).sum(axis=0) / 24.0 for d in uniq])
    daily["P4"] = np.stack([
        np.maximum(t2m[(dates == d) & syn], 0.0).mean(axis=0) for d in uniq])
    daily["M4"] = np.stack([
        np.maximum(t2m[(dates == d) & syn].mean(axis=0), 0.0) for d in uniq])

    days = pd.to_datetime(pd.Series(uniq))
    inside = ((days >= START) & (days <= END)).to_numpy()
    pre = (days < START).to_numpy()
    post = (days > END).to_numpy()
    store_dt = pd.to_datetime(pd.Series(store_dates))
    pre_store = (store_dt < START).to_numpy()
    post_store = (store_dt > END).to_numpy()
    index = {d: i for i, d in enumerate(uniq)}

    fields = {}
    for name, arr in daily.items():
        cum = np.cumsum(arr, axis=0)
        before = np.flatnonzero(days < START)
        base = cum[before[-1]] if len(before) else np.zeros(arr.shape[1:])
        growing = np.where(inside[:, None, None], cum - base, 0.0)
        out = np.zeros((len(store_dates),) + arr.shape[1:], dtype=np.float64)
        for k, label in enumerate(store_dates):
            if label in index and START <= pd.Timestamp(str(label)) <= END:
                out[k] = growing[index[label]]
        fields[name] = out
        np.savez_compressed(OUT / f"pdd_{YEAR}_{name}.npz",
                            pdd=np.clip(np.rint(out), 0, 65535).astype(np.uint16),
                            dates=store_dates, lat=lat, lon=lon,
                            valid_from=np.array(["03-01"]))
        say(f"{name:3s}: max inside window {out.max():7.0f} degC-day, "
            f"zero before 1 Mar {bool((out[pre_store] == 0).all())}, "
            f"zero after 31 Jul {bool((out[post_store] == 0).all())}")

    for name, out in fields.items():
        assert (out[post_store] == 0).all(), f"{name}: nonzero P after 31 July"
    say("all three are zero on every step after 31 July")
    say()

    later = store_dates == f"{YEAR}-05-31"
    say("P on 31 May 1966, degC-day:")
    say(f"  {'station':14s} {'H':>8s} {'P4':>8s} {'M4':>8s} {'GHCN':>8s}")
    rows = []
    for label, (sid, la, lo) in bp.GHCN_STATIONS.items():
        gh, n = bp.ghcn_pdd(sid, YEAR)
        vals = {name: bilinear(fields[name][np.flatnonzero(later)[0]], lat, lon, la, lo)
                for name in fields}
        say(f"  {label:14s} {vals['H']:8.0f} {vals['P4']:8.0f} {vals['M4']:8.0f} {gh:8.0f}")
        rows.append(dict(station=label, ghcn=gh, **vals, ghcn_days=n))

    df = pd.DataFrame(rows)
    for name in ("H", "P4", "M4"):
        df[f"bias_{name}"] = 100 * (df[name] - df.ghcn) / df.ghcn
    say()
    say("percent bias vs GHCN (positive = ERA5 high):")
    say(df[["station", "bias_H", "bias_P4", "bias_M4"]].round(1).to_string(index=False))
    say()
    summary = []
    for name in ("H", "P4", "M4"):
        mad = float(df[f"bias_{name}"].abs().mean())
        spread = float(df[f"bias_{name}"].max() - df[f"bias_{name}"].min())
        summary.append(dict(definition=name, mean_abs_bias=mad, spread=spread))
    s = pd.DataFrame(summary).sort_values(["mean_abs_bias", "spread"])
    say("1966 ranking by the pre-written rule (mean absolute bias, then spread):")
    say(s.round(1).to_string(index=False))
    say()
    winner = s.iloc[0].definition
    say(f"1966 winner: {winner}")
    say()
    say("GATE. The rule spans 1966-1970; only 1966 is available until the pull.")
    say("If the hourly pull is chosen, 1967-1970 hourly fields settle the choice.")
    say("If the 4-sample pull is chosen, storing both P4 and M4 for 1966-1970")
    say("settles it. No ice observation or held-out value enters this choice.")

    df.to_csv(ROOT / "data" / "derived" / "pdd_definition_comparison_1966.csv",
              index=False)
    (DIAG / "pdd_definition_comparison_1966.txt").write_text(
        "\n".join(say_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}/*.npz")
    print(f"wrote {DIAG / 'pdd_definition_comparison_1966.txt'}")


if __name__ == "__main__":
    main()
