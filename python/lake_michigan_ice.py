"""How much of each Great Lake does the model freeze?

The complaint that started this script: the map shows Lake Michigan frozen
across the deep basin every winter, which is implausible. The model has only
air temperature, depth and a melt term, so the question is how much of the basin
it actually fills, and whether the depth term is doing the work it is supposed
to do.

Reads the 3 arc-second product rasters (depth + lake-code mask) and the per
season S/P fields, evaluates the model at full resolution over a lake window,
and reports:

  ice   fraction of lake area with h > 0
  thick fraction of lake area with h >= 20.32 cm (the ramp's white point)
  mean  area-weighted mean thickness
  deep  the same, restricted to cells deeper than 100 m

Also writes the values at the deepest cell, because that is where a model with
no heat storage must be most wrong, and the S that the deepest cell would need
in order to stay ice-free. That second number is the structural test: depth can
only suppress ice while S < C1 ln d, and C1 ln d grows very slowly, so a deep
basin freezes as soon as a cold winter pushes S past it.

Run:
  $env:PYTHONIOENCODING="utf-8"
  python python/lake_michigan_ice.py
  python python/lake_michigan_ice.py --lake Michigan --seasons 1977 1994 2014 2025
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import struct
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
RASTERS = ROOT / "data" / "derived" / "rasters"
FIELDS = ROOT / "web" / "fields"
OUT = ROOT / "data" / "derived" / "diagnostics"

CODES = {"Superior": 1, "Michigan": 2, "Huron": 3}
FLAG8 = 128
CLAMP_CM = 20.32
DEEP_M = 100.0

# A lake window in degrees, deliberately generous; the lake code does the
# clipping, the window only bounds the read.
WINDOWS = {
    "Superior": (-92.3, -84.2, 46.2, 49.2),
    "Michigan": (-88.4, -84.4, 41.4, 46.3),
    "Huron": (-84.8, -79.6, 42.9, 46.5),
}


def load_manifest() -> dict:
    return json.loads((RASTERS / "manifest.json").read_text(encoding="utf-8"))


def load_window(layer: dict, lon0: float, lon1: float, lat0: float, lat1: float):
    """Assemble depth and mask over a lon/lat window from the PNG tiles."""
    step = layer["step"]
    left, top = layer["left"], layer["top"]
    c0 = max(0, int(np.floor((lon0 - left) / step)))
    c1 = min(layer["width"], int(np.ceil((lon1 - left) / step)))
    r0 = max(0, int(np.floor((top - lat1) / step)))
    r1 = min(layer["height"], int(np.ceil((top - lat0) / step)))

    mask = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
    dep = np.zeros_like(mask)
    for t in layer["tiles"]:
        tr0, tc0 = t["row0"], t["col0"]
        tr1, tc1 = tr0 + t["rows"], tc0 + t["cols"]
        ir0, ir1 = max(r0, tr0), min(r1, tr1)
        ic0, ic1 = max(c0, tc0), min(c1, tc1)
        if ir0 >= ir1 or ic0 >= ic1:
            continue
        name = f"r{t['row']}_c{t['col']}.png"
        m = np.array(Image.open(RASTERS / "gl" / f"mask_{name}"))
        d = np.array(Image.open(RASTERS / "gl" / f"depth_{name}"))
        sub = (slice(ir0 - r0, ir1 - r0), slice(ic0 - c0, ic1 - c0))
        tsub = (slice(ir0 - tr0, ir1 - tr0), slice(ic0 - tc0, ic1 - tc0))
        mask[sub] = m[tsub]
        dep[sub] = d[tsub]

    lon = left + (np.arange(c0, c1) + 0.5) * step
    lat = top - (np.arange(r0, r1) + 0.5) * step
    return mask, dep, lon, lat


def load_season(year: int):
    """S, P, dates and grid from a season file."""
    raw = gzip.decompress((FIELDS / f"season_{year}.bin.gz").read_bytes())
    if raw[:5] != b"ICEF1":
        raise RuntimeError("bad magic")
    n, h, w = struct.unpack("<III", raw[8:20])
    body = np.frombuffer(raw, dtype="<u2", offset=20, count=2 * n * h * w)
    S = body[: n * h * w].reshape(n, h, w)
    P = body[n * h * w:].reshape(n, h, w)

    man = json.loads((FIELDS / "manifest.json").read_text(encoding="utf-8"))
    s_meta = next(s for s in man["seasons"] if s["season"] == year)
    return S, P, s_meta["frames"], man["grid"], man["model"]


def bilinear(field: np.ndarray, grid: dict, lon: np.ndarray, lat: np.ndarray):
    """Bilinear sample a (rows=lat descending, cols=lon ascending) field."""
    lon0, lat0 = float(grid["lon"][0]), float(grid["lat"][0])
    step = float(grid["lon"][1]) - float(grid["lon"][0])
    hmax, wmax = field.shape[0] - 1, field.shape[1] - 1

    fx = np.clip((lon - lon0) / step, 0.0, wmax)
    fy = np.clip((lat0 - lat) / step, 0.0, hmax)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, wmax)
    y1 = np.minimum(y0 + 1, hmax)
    tx = (fx - x0).astype(np.float32)
    ty = (fy - y0).astype(np.float32)

    f = field.astype(np.float32)
    a = f[y0, x0] * (1 - tx) + f[y0, x1] * tx
    b = f[y1, x0] * (1 - tx) + f[y1, x1] * tx
    return a * (1 - ty) + b * ty


def thickness(S: np.ndarray, P: np.ndarray, depth_m: np.ndarray, model: dict):
    arg = np.maximum(0.0, S - float(model["C1"]) * np.log(np.maximum(depth_m, 1.0)))
    return np.maximum(0.0, float(model["k"]) * np.sqrt(arg) - float(model["m"]) * P)


def season_stats(season: int, lake: str, stride: int = 6) -> dict:
    man = load_manifest()
    gl = next(l for l in man["layers"] if l["name"] == "gl")
    lon0, lon1, lat0, lat1 = WINDOWS[lake]
    mask, dep, lon, lat = load_window(gl, lon0, lon1, lat0, lat1)

    code = mask.astype(np.int16) & 0x7F
    water = (mask & FLAG8) == 0
    sel = water & (code == CODES[lake])
    if not sel.any():
        raise RuntimeError(f"no {lake} cells in window")

    depth = np.exp(dep / 40.0).astype(np.float32)
    lon2 = np.broadcast_to(lon[None, :], mask.shape)
    lat2 = np.broadcast_to(lat[:, None], mask.shape)
    wgt = np.cos(np.deg2rad(lat2)) * sel

    S, P, frames, grid, model = load_season(season)

    # Find the peak frame on a subsample; the full-resolution pass is expensive
    # and only the peak matters here.
    sub = sel[::stride, ::stride]
    lon_s = lon2[::stride, ::stride][sub]
    lat_s = lat2[::stride, ::stride][sub]
    best, best_i = -1.0, 0
    for i in range(S.shape[0]):
        h = thickness(bilinear(S[i], grid, lon_s, lat_s),
                      bilinear(P[i], grid, lon_s, lat_s),
                      depth[::stride, ::stride][sub], model)
        frac = float((h > 0).mean())
        if frac > best:
            best, best_i = frac, i

    i = best_i
    s_at = bilinear(S[i], grid, lon2[sel], lat2[sel])
    p_at = bilinear(P[i], grid, lon2[sel], lat2[sel])
    h = thickness(s_at, p_at, depth[sel], model)
    area = wgt[sel]
    total = float(area.sum())
    deep = depth[sel] > DEEP_M
    deepest = float(depth[sel].max())
    j = int(np.argmax(depth[sel]))

    return dict(
        season=f"{season}-{str(season + 1)[-2:]}",
        date=frames[i],
        frame=i,
        cells=int(sel.sum()),
        area_km2=round(float(sel.sum()) * 0.0008333333 ** 2 * 111.32 ** 2
                       * np.cos(np.deg2rad(float(lat.mean()))), 1),
        ice_pct=round(100.0 * float(area[h > 0].sum() / total), 1),
        thick_pct=round(100.0 * float(area[h >= CLAMP_CM].sum() / total), 1),
        mean_cm=round(float((h * area).sum() / total), 1),
        max_cm=round(float(h.max()), 1),
        deep_pct=round(100.0 * float(area[deep][h[deep] > 0].sum()
                                     / max(area[deep].sum(), 1e-9)), 1),
        deep_mean_cm=round(float((h[deep] * area[deep]).sum()
                                 / max(area[deep].sum(), 1e-9)), 1),
        deepest_m=round(deepest, 1),
        s_at_deepest=round(float(s_at[j]), 1),
        s_needed_zero=round(float(model["C1"]) * float(np.log(deepest)), 1),
        h_at_deepest_cm=round(float(h[j]), 1),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default="Michigan", choices=sorted(CODES))
    ap.add_argument("--seasons", nargs="*", type=int,
                    default=[1963, 1970, 1977, 1979, 1994, 2003, 2012, 2014, 2019, 2024])
    ap.add_argument("--all", action="store_true", help="every season 1950-2024")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", type=pathlib.Path,
                    default=OUT / "lake_ice_area.csv")
    args = ap.parse_args(argv)

    seasons = list(range(1950, 2025)) if args.all else args.seasons

    rows = []
    print(f"=== modelled ice over Lake {args.lake} ===")
    print(f"{'season':>9} {'peak date':>12} {'ice%':>6} {'>=20.3cm%':>10} "
          f"{'mean':>6} {'max':>6} {'deep>100m%':>11} {'deep mean':>10} "
          f"{'deepest':>8} {'S deep':>7} {'S zero':>7} {'h deep':>7}")
    for year in seasons:
        r = season_stats(year, args.lake, args.stride)
        rows.append(r)
        print(f"{r['season']:>9} {r['date']:>12} {r['ice_pct']:>6.1f} "
              f"{r['thick_pct']:>10.1f} {r['mean_cm']:>6.1f} {r['max_cm']:>6.1f} "
              f"{r['deep_pct']:>11.1f} {r['deep_mean_cm']:>10.1f} "
              f"{r['deepest_m']:>8.1f} {r['s_at_deepest']:>7.1f} "
              f"{r['s_needed_zero']:>7.1f} {r['h_at_deepest_cm']:>7.1f}")

    if rows:
        ice = np.array([r["ice_pct"] for r in rows])
        thick = np.array([r["thick_pct"] for r in rows])
        print(f"\n{len(rows)} seasons: ice cover median {np.median(ice):.1f}% "
              f"(min {ice.min():.1f}, max {ice.max():.1f}); "
              f"fully frozen (100%) in {int((ice >= 99.9).sum())} seasons")
        print(f"at or above the white point: median {np.median(thick):.1f}%")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys())
        with args.out.open("w", encoding="utf-8", newline="") as f:
            f.write("lake," + ",".join(cols) + "\n")
            for r in rows:
                f.write(args.lake + "," + ",".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
