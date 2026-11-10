"""Build the per-season S / P fields the browser needs to shade ice thickness.

The viewer does not ship thickness itself (it depends on per-pixel depth and
would be 9,000 frames of raster). It ships the two smooth climate fields on the
existing 0.1 degree grid and evaluates

    h = max(0, k * sqrt(max(0, S - C1 * ln d)) - m * P)

in the fragment shader, where d is decoded from the static depth layer.

Each season runs 1 October -> 30 June. That is the ice season: S resets on
1 October, P is defined 1 March - 31 July, and by the end of June the melt term
has driven h to zero everywhere. Stopping at 30 June avoids the August jump
that would occur when P returns to zero while S is still large.

Output (one file per season, gzip):
  web/fields/season_YYYY.bin.gz
    magic  b"ICEF1"
    uint32 n_frames, uint32 height, uint32 width   (little endian)
    S      uint16[n_frames * height * width]
    P      uint16[n_frames * height * width]
  web/fields/manifest.json
    grid, model constants, frame dates, per-season sizes

Run:  $env:PYTHONIOENCODING="utf-8"; python python/build_web_fields.py
"""

from __future__ import annotations

import gzip
import json
import pathlib
import struct
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFDD = ROOT / "data" / "derived" / "era5"
PDD = ROOT / "data" / "derived" / "pdd"
PARAMS = ROOT / "data" / "derived" / "model_parameters.csv"
WEB = ROOT / "web" / "fields"
WEB.mkdir(parents=True, exist_ok=True)

SEASON_FIRST = 1950
SEASON_LAST = 2024          # season 2024-25 ends 30 June 2025
START_MMDD = "10-01"
END_MMDD = "06-30"
MAGIC = b"ICEF1\0\0\0"       # 8 bytes so the uint16 payload is 4-byte aligned


def load(year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.load(CFDD / f"cfdd_{year}.npz")
    b = np.load(PDD / f"pdd_{year}.npz")
    if not np.array_equal(a["dates"], b["dates"]):
        raise RuntimeError(f"date mismatch between cfdd and pdd for {year}")
    return a["cfdd"].astype(np.uint16), b["pdd"].astype(np.uint16), a["dates"]


def season_frames(start: int):
    """Indices for 1 Oct `start` .. 30 Jun `start+1`, in playback order."""
    s0, p0, d0 = load(start)
    s1, p1, d1 = load(start + 1)
    dates = np.concatenate([d0, d1])
    s = np.concatenate([s0, s1], axis=0)
    p = np.concatenate([p0, p1], axis=0)
    lo = np.datetime64(f"{start}-{START_MMDD}")
    hi = np.datetime64(f"{start + 1}-{END_MMDD}")
    d = dates.astype("datetime64[D]")
    keep = (d >= lo) & (d <= hi)
    if not keep.any():
        raise RuntimeError(f"no frames in season {start}")
    # Chronological order; the concatenation is already sorted, but be explicit.
    order = np.argsort(dates[keep])
    return s[keep][order], p[keep][order], dates[keep][order]


def main() -> None:
    t0 = time.time()
    params = np.genfromtxt(PARAMS, delimiter=",", names=True, dtype=None,
                           encoding="utf-8")
    model = dict(
        k=float(params["k"]), C0=float(params["C0"]),
        C1=float(params["C1"]), m=float(params["m"]),
        formula="max(0, k*sqrt(max(0, S - C0 - C1*ln(d))) - m*P)",
        thaw_start=str(params["thaw_start"]).strip("'\""),
        thaw_end=str(params["thaw_end"]).strip("'\""),
        clamp_cm=20.32,
    )
    seasons = []
    total = 0
    for year in range(SEASON_FIRST, SEASON_LAST + 1):
        s, p, dates = season_frames(year)
        n, h, w = s.shape
        payload = MAGIC + struct.pack("<III", n, h, w)
        payload += s.astype("<u2").tobytes()
        payload += p.astype("<u2").tobytes()
        blob = gzip.compress(payload, compresslevel=9, mtime=0)
        path = WEB / f"season_{year}.bin.gz"
        path.write_bytes(blob)
        total += len(blob)
        seasons.append(dict(
            season=year, label=f"{year}-{str(year + 1)[-2:]}",
            frames=[str(x) for x in dates], n_frames=int(n),
            bytes=len(blob),
            S_max=int(s.max()), P_max=int(p.max()),
            file=path.name))
        print(f"  {year}-{str(year + 1)[-2:]}  {n:3d} frames  "
              f"{len(blob) / 1e6:5.2f} MB  S<={int(s.max())} P<={int(p.max())}")

    a = np.load(CFDD / f"cfdd_{SEASON_FIRST}.npz")
    manifest = dict(
        grid=dict(lat=[float(x) for x in a["lat"]],
                  lon=[float(x) for x in a["lon"]],
                  height=int(a["cfdd"].shape[1]),
                  width=int(a["cfdd"].shape[2]),
                  crs="EPSG:4326-like 0.1 deg, lat descending, lon ascending"),
        model=model,
        encoding=dict(magic=MAGIC.decode(), byte_order="little",
                      S="uint16, degree-day", P="uint16, degree-day",
                      layout="S then P, frame-major, row-major"),
        season_start=START_MMDD, season_end=END_MMDD,
        first_season=SEASON_FIRST, last_season=SEASON_LAST,
        total_bytes=int(total),
        seasons=seasons,
    )
    (WEB / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"seasons {len(seasons)}, total {total / 1e6:.1f} MB")
    print(f"wrote {WEB / 'manifest.json'}")
    print(f"elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
