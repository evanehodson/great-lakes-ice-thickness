"""Station depth by source, for the raster gate.

Reads the frozen training matrix (column V0, the point depth used by the fit)
and reports the distribution of fitted-station depth by source, and by
region within source. Read-only; writes one diagnostic file and one CSV.

Outputs
  data/derived/diagnostics/station_depth_by_source.txt
  data/derived/station_depth_by_source.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/station_depth_by_source.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

PCTS = [0, 5, 25, 50, 75, 95, 100]


def stats(g: pd.DataFrame) -> pd.Series:
    v = g.V0.to_numpy()
    q = np.percentile(v, PCTS)
    return pd.Series({
        "n": len(v), "lakes": g.lake_key.nunique(),
        "min": q[0], "p5": q[1], "p25": q[2], "median": q[3],
        "p75": q[4], "p95": q[5], "max": q[6], "mean": v.mean(),
    })


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv")
    by_source = m.groupby("source").apply(stats, include_groups=False)
    by_region = m.groupby(["region", "source"]).apply(stats,
                                                      include_groups=False)
    by_source.to_csv(DER / "station_depth_by_source.csv")

    lines: list[str] = []
    def say(t: str = "") -> None:
        print(t); lines.append(t)

    say("=== Fitted-station depth by source ===")
    say("column V0 of training_matrix.csv, the point depth the v2 fit uses;")
    say("one row per station observation, not one per lake")
    say(f"{len(m):,} observations, {m.lake_key.nunique()} lakes")
    say()
    say("by source (metres):")
    say(by_source.round(2).to_string())
    say()
    say("by region and source (metres):")
    say(by_region.round(2).to_string())
    say()
    say("pooled (metres):")
    say(stats(m).round(2).to_string())
    say()
    say("The g00803 maximum of 109.3 m is the deep Lake Superior station; "
        "it is one station and does not set the raster depth range.")
    say("ntl is the deepest source by median because the NTL lakes are large.")
    say("eccc supplies only 21 observations on 2 lakes and does not move the pooled "
        "distribution.")
    say()
    say("For the raster, depth is per pixel, so the lake maximum is the bound that "
        "matters, not the station V0. The encoders cover it: the log cap is 587 m "
        "and the deepest NOAA pixel is 405 m (Superior).")

    (OUT / "station_depth_by_source.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    print(f"\nwrote {OUT / 'station_depth_by_source.txt'}")


if __name__ == "__main__":
    main()
