"""Verify that adding G00803 snow_cm changed nothing except that column.

Compares the post-change derived files with the backups in
`data/derived/backups/pre_g00803_snow/`:

  observations.csv           row order, every column except snow_cm, counts, split
  training_matrix.csv        row order, every column except snow_cm, counts
  fit_results.csv            all fitted parameters and metrics
  validation_predictions.csv all held-out predictions

Also checks that the new G00803 snow values reproduce a direct parse of
data2.dat columns [36:38], and lists where snow_cm is referenced in the
pipeline (to show that no fitting or validation code consumes it).

Outputs
  data/derived/diagnostics/snow_column_verification.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/verify_snow_column.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
BK = DER / "backups" / "pre_g00803_snow"
DIAG = DER / "diagnostics"

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def compare(old: pd.DataFrame, new: pd.DataFrame, label: str,
            key: list[str]) -> None:
    say(f"=== {label} ===")
    say(f"shape old {old.shape}  new {new.shape}")
    if old.shape != new.shape:
        raise SystemExit(f"{label}: shape changed")
    same_key = (old[key].reset_index(drop=True)
                == new[key].reset_index(drop=True)).all(axis=None)
    say(f"row order and keys identical: {bool(same_key)}")
    if not bool(same_key):
        raise SystemExit(f"{label}: row order changed")

    cols = [c for c in old.columns if c != "snow_cm"]
    pd.testing.assert_frame_equal(
        new[cols].reset_index(drop=True), old[cols].reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9,
    )
    say(f"all {len(cols)} other columns identical: True")
    say()


def main() -> None:
    old = pd.read_csv(BK / "observations.csv", parse_dates=["date"])
    new = pd.read_csv(DER / "observations.csv", parse_dates=["date"])
    compare(old, new, "observations.csv", ["lake_key", "date"])

    say("counts")
    say(f"  rows: {len(new):,} (backup {len(old):,})")
    say(f"  lakes: {new.lake_key.nunique()} (backup {old.lake_key.nunique()})")
    splits = new.split.value_counts()
    say(f"  split: train {splits.get('train')}, val {splits.get('val')}")
    say(f"  split identical to backup: "
        f"{bool((new.split.to_numpy() == old.split.to_numpy()).all())}")
    say()

    say("snow_cm before vs after")
    say(f"  backup non-null: {int(old.snow_cm.notna().sum())}")
    say(f"  new non-null:    {int(new.snow_cm.notna().sum())}")
    say(new.groupby("source").snow_cm.agg(["count", "size"]).to_string())
    say()

    # Direct parse cross-check for G00803.
    path = ROOT / "data" / "raw" / "g00803" / "extract" / "data2.dat"
    direct = {}
    for line in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if len(line) < 56:
            continue
        try:
            sid = int(line[0:3])
            yy, mm, dd = int(line[4:6]), int(line[6:8]), int(line[8:10])
        except ValueError:
            continue
        raw = line[36:38].strip()
        try:
            sd = float(raw) if raw else np.nan
        except ValueError:
            sd = np.nan
        direct[(sid, 1900 + yy, mm, dd)] = sd

    g = new[new.source == "g00803"].copy()
    sid = g.lake_key.str.split(":").str[1].astype(int)
    keys = list(zip(sid, g.date.dt.year, g.date.dt.month, g.date.dt.day))
    expect = np.array([direct.get(k, np.nan) for k in keys], dtype=float)
    got = g.snow_cm.to_numpy(dtype=float)
    both = ~np.isnan(expect)
    agree = np.allclose(got[both], expect[both], equal_nan=True)
    say(f"G00803 direct-parse cross-check: {int(both.sum()):,} populated records, "
        f"all values identical: {bool(agree)}")
    if not agree:
        bad = np.flatnonzero(got[both] != expect[both])[:5]
        say(f"  mismatches: {len(bad)}")
    say()

    # training matrix
    old_m = pd.read_csv(BK / "training_matrix.csv", parse_dates=["date"])
    new_m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    compare(old_m, new_m, "training_matrix.csv", ["lake_key", "date"])

    # fit outputs
    for name in ("fit_results.csv", "validation_predictions.csv"):
        o = pd.read_csv(BK / name)
        n = pd.read_csv(DER / name)
        pd.testing.assert_frame_equal(n, o, check_dtype=False,
                                      check_exact=False, rtol=1e-9, atol=1e-9)
        say(f"{name}: identical to backup ({len(n):,} rows)")

    say()
    say("snow_cm references in the pipeline:")
    for script in ("fit_model.py", "bootstrap_model.py", "build_pdd.py",
                   "sample_cfdd.py", "build_observations.py"):
        text = (ROOT / "python" / script).read_text(encoding="utf-8").splitlines()
        hits = [f"    line {i}: {t.strip()}" for i, t in enumerate(text, 1)
                if "snow_cm" in t]
        say(f"  {script}: {len(hits)} reference(s)")
        for h in hits:
            say(h)

    (DIAG / "snow_column_verification.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {DIAG / 'snow_column_verification.txt'}")


if __name__ == "__main__":
    main()
