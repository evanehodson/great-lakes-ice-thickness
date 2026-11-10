"""Download NOAA NCEI 3 arc-second bathymetry for Superior, Michigan, Huron.

Source: https://www.ngdc.noaa.gov/mgg/greatlakes/greatlakes.html
Each lake ships a GeoTIFF in a .tar.gz. Depth is metres below low-water datum,
stored as negative values (land positive), so depth = -elevation.

Output: data/raw/bathy/<lake>.tif
"""

from __future__ import annotations

import pathlib
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "bathy"
DEST.mkdir(parents=True, exist_ok=True)

BASE = "https://www.ngdc.noaa.gov/mgg/greatlakes/{lake}/data/geotiff/{lake}_lld.geotiff.tar.gz"
UA = {"User-Agent": "ice-research/1.0"}


def fetch(lake: str) -> None:
    tif = DEST / f"{lake}.tif"
    if tif.exists():
        print(f"{lake}: already have {tif.name}")
        return

    url = BASE.format(lake=lake)
    tgz = DEST / f"{lake}_lld.geotiff.tar.gz"
    print(f"{lake}: downloading {url}", flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, \
            open(tgz, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    print(f"{lake}: {tgz.stat().st_size / 1e6:.1f} MB, extracting", flush=True)

    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        # Archives also carry .tfw / .prj sidecars; take the GeoTIFF itself.
        tifs = [m for m in members if m.name.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise RuntimeError(f"no GeoTIFF in {tgz}: {[m.name for m in members]}")
        member = max(tifs, key=lambda m: m.size)
        member.name = tif.name
        tf.extract(member, DEST)

    tgz.unlink()
    print(f"{lake}: wrote {tif.name} ({tif.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> None:
    for lake in ("superior", "michigan", "huron"):
        fetch(lake)


if __name__ == "__main__":
    main()
