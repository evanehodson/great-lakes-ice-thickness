"""Proposal only: an 8-bit log-depth encoding for the raster product.

Current scheme: uint8, 1 m steps, values 0-255, so 255 m is the hard cap.
Proposed scheme: v = clip(round(ln(d) * 40), 0, 255), d in metres, floor 1 m,
decode d_hat = exp(v / 40).

Nothing is applied. This script only measures the proposal:
  - how many Superior and Michigan lake pixels the 255 m cap clips today;
  - the depth quantization step and its error in the growth argument (degC-day);
  - the resulting ice-thickness error in cm for small lakes;
  - the same error for the current 1 m scheme, for comparison;
  - how many Michigan inland lakes sit below the 1 m floor.

Outputs
  data/derived/diagnostics/depth_encoding_proposal.txt
  data/derived/depth_encoding_error.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/depth_encoding_proposal.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import rasterio

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bathy"
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"

K = 1.6865
C1 = 34.38
SCALE = 40.0
LOG_MAX = float(np.exp(255.0 / SCALE))     # 586.5 m
LOG_DSTEP = C1 * (0.5 / SCALE)             # degC-day, half a code step
CURRENT_CAP = 255.0


def pixel_area_km2(lat: float) -> float:
    dy = 0.0008333333 * 111.32
    dx = dy * np.cos(np.deg2rad(lat))
    return dx * dy


def lake_stats(name: str) -> dict:
    with rasterio.open(RAW / f"{name}.tif") as ds:
        z = ds.read(1, masked=True).astype(np.float64)
        lat = float((ds.bounds.top + ds.bounds.bottom) / 2)
    depth = np.ma.masked_where(np.ma.getmaskarray(z) | (z >= 0), -z)
    px = pixel_area_km2(lat)
    valid = int(depth.count())
    over_cap = int(np.ma.sum(depth > CURRENT_CAP))
    over_log = int(np.ma.sum(depth > LOG_MAX))
    under_floor = int(np.ma.sum((depth > 0) & (depth < 1.0)))
    return dict(lake=name, pixels=valid, area_km2=valid * px,
                max_depth_m=float(depth.max()),
                px_over_255=over_cap, pct_over_255=100 * over_cap / valid,
                area_over_255_km2=over_cap * px,
                px_over_log_cap=over_log,
                px_under_1m=under_floor, pct_under_1m=100 * under_floor / valid,
                pixel_area_m2=px * 1e6)


def main() -> None:
    lines: list[str] = []
    def say(t: str = "") -> None:
        print(t); lines.append(t)

    say("=== 8-bit depth encoding: proposal, not applied ===")
    say()
    say("current: uint8, 1 m steps, cap 255 m")
    say(f"proposed: v = clip(round(ln d * 40), 0, 255), floor 1 m; "
        f"decode exp(v/40)")
    say(f"  code step {1/SCALE:.4f} in ln(d) = {100*(np.exp(1/SCALE)-1):.2f}% "
        f"in d; cap {LOG_MAX:.1f} m")
    say(f"  depth-term error from one half step: C1 * {0.5/SCALE:.4f} = "
        f"{LOG_DSTEP:.3f} degC-day, independent of depth")
    say()

    say("=== bathymetry: what the 255 m cap clips today ===")
    stats = [lake_stats(n) for n in ("superior", "michigan", "huron")]
    tab = pd.DataFrame(stats)
    say(tab[["lake", "pixels", "area_km2", "max_depth_m", "px_over_255",
             "pct_over_255", "area_over_255_km2", "px_over_log_cap",
             "px_under_1m", "pct_under_1m"]].round(2).to_string(index=False))
    say()
    say("pixels are 3 arc-second cells on the source grid; area uses the "
        "cell size at the lake's mid-latitude")
    say()

    # ---- error table
    rows = []
    for d in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
        # current scheme, worst error inside the 1 m bin that holds d
        if d < 1.5:
            cur = C1 * np.log(1.5 / 1.0)
        else:
            cur = C1 * np.log(d / (d - 0.5))
        for s in (50.0, 100.0, 200.0, 400.0):
            arg = s - C1 * np.log(d)
            if arg <= 0:
                rows.append(dict(depth_m=d, S=s, dS_log=LOG_DSTEP,
                                 dS_current=cur, dh_log_cm=0.0,
                                 dh_current_cm=0.0, note="at/below onset"))
                continue
            grad = K / (2 * np.sqrt(arg))
            rows.append(dict(depth_m=d, S=s, dS_log=LOG_DSTEP,
                             dS_current=cur,
                             dh_log_cm=grad * LOG_DSTEP,
                             dh_current_cm=grad * cur, note=""))
    err = pd.DataFrame(rows)
    err.to_csv(DER / "depth_encoding_error.csv", index=False)

    say("=== thickness error from depth quantization (cm), v2 growth term ===")
    piv_log = err.pivot(index="depth_m", columns="S", values="dh_log_cm")
    piv_cur = err.pivot(index="depth_m", columns="S", values="dh_current_cm")
    say("log scheme, max |dh| in cm:")
    say(piv_log.round(3).to_string())
    say()
    say("current 1 m scheme, max |dh| in cm:")
    say(piv_cur.round(3).to_string())
    say()
    say("Near the onset the derivative is unbounded, but ice is clipped at 0; "
        f"the worst a code step can add there is k*sqrt({LOG_DSTEP:.2f}) = "
        f"{K*np.sqrt(LOG_DSTEP):.2f} cm, and half of that after the clip.")
    say()

    mi = pd.read_csv(DER / "mi_lake_depth.csv")
    below = mi[mi.max_depth_m < 1.0]
    say("=== Michigan inland lakes ===")
    say(f"lakes in the depth table: {len(mi):,}; "
        f"max depth median {mi.max_depth_m.median():.2f} m, "
        f"min {mi.max_depth_m.min():.2f} m, max {mi.max_depth_m.max():.1f} m")
    say(f"lakes below the 1 m floor: {len(below)} "
        f"({100*len(below)/len(mi):.2f}%)")
    say(f"max depth over 255 m: {int((mi.max_depth_m > 255).sum())} "
        "(the current cap never binds inland)")
    say()
    say("Recommendation: log encoding removes the cap problem and cuts the "
        "depth-term quantization error at small lakes by roughly an order of "
        "magnitude; the floor still collapses the few lakes under 1 m, which "
        "are below the model's valid depth range anyway.")
    say()
    say("PROPOSAL ONLY: nothing in the raster product has been changed.")

    (OUT / "depth_encoding_proposal.txt").write_text("\n".join(lines) + "\n",
                                                     encoding="utf-8")
    print(f"\nwrote {OUT / 'depth_encoding_proposal.txt'}")


if __name__ == "__main__":
    main()
