"""Compare the browser shader against an independent Python evaluation.

The viewer evaluates the v2 model per fragment from four textures. This script
rebuilds the expected colour from the same files on disk, with the same
sampling rules the shader uses (bilinear on mip 0, texel-centre convention,
clamp to edge), so a mismatch points at a decoding or formula bug rather than
at a modelling difference.

Input is a JSON file of samples produced from the browser: for each sample, the
geographic coordinate of one fragment centre and the RGB the GPU wrote there.

    python python/validate_shader.py --samples shader_samples.json

Two classes of sample are reported separately:

  flat   full coverage and a single depth code across the sample's 3x3 texel
         neighbourhood. Sub-texel placement and mip selection cannot move the
         result, so these carry a one-code-step tolerance.
  grad   the sample sits on a depth contour. Half a texel of sampling offset
         legitimately shifts the depth code, so the tolerance is ~2 cm.

Writes data/derived/diagnostics/shader_validation.txt and exits non-zero if any
sample breaks its tolerance.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
import struct
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYERS = ROOT / "web" / "layers"
FIELDS = ROOT / "web" / "fields"
OUT = ROOT / "data" / "derived" / "diagnostics" / "shader_validation.txt"

TOL = 1.5          # one code step on a flat sample
GRAD_TOL = 12.0    # ~2 cm where the sample sits on a depth contour
LAND = (0.047, 0.059, 0.078)
OTHER = (0.024, 0.055, 0.086)


def ramp(h: float) -> tuple[float, float, float]:
    """The shader's ramp, in linear 0..1 channel units."""
    x = min(max(h / 20.32, 0.0), 1.0)
    a = (0.043, 0.098, 0.196)
    b = (0.071, 0.208, 0.420)
    c = (0.129, 0.369, 0.659)
    d = (0.298, 0.620, 0.851)
    e = (0.678, 0.863, 0.961)
    w = (1.0, 1.0, 1.0)
    if x < 0.18:
        t, p, q = x / 0.18, a, b
    elif x < 0.42:
        t, p, q = (x - 0.18) / 0.24, b, c
    elif x < 0.68:
        t, p, q = (x - 0.42) / 0.26, c, d
    elif x < 0.86:
        t, p, q = (x - 0.68) / 0.18, d, e
    else:
        t, p, q = (x - 0.86) / 0.14, e, w
    return tuple(p[i] + (q[i] - p[i]) * t for i in range(3))


def mix(p: tuple, q: tuple, t: float) -> tuple[int, int, int]:
    return tuple(int(round(255.0 * (p[i] + (q[i] - p[i]) * t)))
                 for i in range(3))


def bilinear(arr: np.ndarray, x: float, y: float) -> float:
    """Texture LINEAR sampling in the texel-centre convention, clamped."""
    h, w = arr.shape
    x = min(max(x * w - 0.5, 0.0), w - 1.0)
    y = min(max(y * h - 0.5, 0.0), h - 1.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = x - x0, y - y0
    top = arr[y0, x0] * (1 - tx) + arr[y0, x1] * tx
    bot = arr[y1, x0] * (1 - tx) + arr[y1, x1] * tx
    return float(top * (1 - ty) + bot * ty)


def field_bilinear(arr: np.ndarray, fx: float, fy: float) -> float:
    """The shader's manual bilinear on the 0.1 degree field, clamped."""
    h, w = arr.shape
    fx = min(max(fx, 0.0), w - 1.0)
    fy = min(max(fy, 0.0), h - 1.0)
    x0, y0 = int(math.floor(fx)), int(math.floor(fy))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = fx - x0, fy - y0
    return float(
        (arr[y0, x0] * (1 - tx) + arr[y0, x1] * tx) * (1 - ty)
        + (arr[y1, x0] * (1 - tx) + arr[y1, x1] * tx) * ty)


def load_layers():
    man = json.loads((LAYERS / "manifest.json").read_text(encoding="utf-8"))
    out = {}
    for layer in man["layers"]:
        cov = np.asarray(Image.open(LAYERS / layer["cov"])).astype(np.float64)
        dep = np.asarray(Image.open(LAYERS / layer["dep"])).astype(np.float64)
        out[layer["name"]] = dict(meta=layer, cov=cov, dep=dep)
    return out


def load_season(meta: dict):
    raw = gzip.decompress((FIELDS / meta["file"]).read_bytes())
    if raw[:4] != b"ICEF":
        raise RuntimeError(f"bad magic in {meta['file']}")
    n, h, w = struct.unpack("<III", raw[8:20])
    count = n * h * w
    S = np.frombuffer(raw, dtype="<u2", count=count, offset=20)
    P = np.frombuffer(raw, dtype="<u2", count=count, offset=20 + 2 * count)
    return (S.reshape(n, h, w).astype(np.float64),
            P.reshape(n, h, w).astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--season", type=int, default=17)
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.samples).read_text(encoding="utf-8-sig"))
    frame = int(payload["frame"])
    layers = load_layers()
    fields = json.loads((FIELDS / "manifest.json").read_text(encoding="utf-8"))
    season = fields["seasons"][args.season]   # same zero-based index as the viewer
    S, P = load_season(season)
    if frame >= S.shape[0]:
        raise SystemExit(f"frame {frame} outside season {args.season}")

    lat0 = fields["grid"]["lat"][0]
    lon0 = fields["grid"]["lon"][0]
    step = fields["grid"]["lon"][1] - fields["grid"]["lon"][0]
    k = fields["model"]["k"]
    C1 = fields["model"]["C1"]
    m = fields["model"]["m"]

    results = []
    for s in payload["samples"]:
        lon, lat = float(s["lon"]), float(s["lat"])
        r, g, b = s["rgb"]
        cov = code = 0.0
        flat = False
        for name in ("gl", "inland"):
            L = layers[name]
            meta = L["meta"]
            u = (lon - meta["left"]) / (meta["width"] * meta["step"])
            v = (meta["top"] - lat) / (meta["height"] * meta["step"])
            if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
                continue
            c = bilinear(L["cov"], u, v) / 255.0
            d = bilinear(L["dep"], u, v) / 255.0
            if c > 0.002:
                # shader: code = dep.r * 255 / cov.r, with both in 0..1
                cov, code = c, d * 255.0 / c
            if cov >= 0.004:
                H, W = L["cov"].shape
                x0 = min(max(int(round(u * W - 0.5)) - 1, 0), W - 3)
                y0 = min(max(int(round(v * H - 0.5)) - 1, 0), H - 3)
                cw = L["cov"][y0:y0 + 3, x0:x0 + 3]
                dw = L["dep"][y0:y0 + 3, x0:x0 + 3]
                flat = bool((cw == 255).all() and (dw.max() - dw.min()) <= 1)
                break

        if cov < 0.004:
            exp = tuple(int(round(255.0 * c)) for c in LAND)
            h = depth = None
        elif code >= 245.0:
            exp = mix(LAND, OTHER, cov)
            h = depth = None
        else:
            depth = math.exp(code / 40.0)
            fx = (lon - lon0) / step
            fy = (lat0 - lat) / step
            Ss = field_bilinear(S[frame], fx, fy)
            Pp = field_bilinear(P[frame], fx, fy)
            arg = max(0.0, Ss - C1 * math.log(depth))
            h = max(0.0, k * math.sqrt(arg) - m * Pp)
            exp = mix(LAND, ramp(h), cov)

        delta = max(abs(exp[i] - (r, g, b)[i]) for i in range(3))
        tol = TOL if flat else GRAD_TOL
        results.append(dict(tag=s.get("tag", ""), lon=lon, lat=lat, cov=cov,
                            code=code, depth=depth, h=h, gpu=(r, g, b),
                            ref=exp, delta=delta, flat=flat, tol=tol,
                            ok=delta <= tol))

    flat = [x for x in results if x["flat"]]
    grad = [x for x in results if not x["flat"]]
    fails = sum(1 for x in results if not x["ok"])

    lines = ["=== Shader vs Python reference ===",
             f"season {season['label']}  frame {frame}  date {season['frames'][frame]}",
             f"model k={k:.4f} C1={C1:.4f} m={m:.4f}",
             f"samples {len(results)}: {len(flat)} flat, {len(grad)} gradient", "",
             "flat samples (strict, tolerance %.1f channel units):" % TOL]
    icey = [x for x in flat if x["h"] is not None and x["h"] >= 2.0]
    shown = sorted(icey, key=lambda y: -y["h"])[:8] + \
        sorted([x for x in flat if x not in icey], key=lambda y: -y["delta"])[:4]
    for x in shown:
        lines.append(
            f"  {x['tag'][:10]:<10} lon {x['lon']:9.5f} lat {x['lat']:8.5f}  "
            f"depth {('%.1f m' % x['depth']) if x['depth'] else '  -  '}  "
            f"h {('%6.2f cm' % x['h']) if x['h'] is not None else '   -   '}  "
            f"gpu {x['gpu'][0]:3d},{x['gpu'][1]:3d},{x['gpu'][2]:3d}  "
            f"ref {x['ref'][0]:3d},{x['ref'][1]:3d},{x['ref'][2]:3d}  "
            f"d {x['delta']:4.1f}  {'OK' if x['ok'] else 'FAIL'}")
    lines += ["", "gradient samples (looser, tolerance %.1f):" % GRAD_TOL]
    for x in sorted(grad, key=lambda y: -y["delta"])[:6]:
        lines.append(
            f"  {x['tag'][:10]:<10} lon {x['lon']:9.5f} lat {x['lat']:8.5f}  "
            f"code {x['code']:6.2f}  "
            f"h {('%6.2f cm' % x['h']) if x['h'] is not None else '   -   '}  "
            f"gpu {x['gpu'][0]:3d},{x['gpu'][1]:3d},{x['gpu'][2]:3d}  "
            f"ref {x['ref'][0]:3d},{x['ref'][1]:3d},{x['ref'][2]:3d}  "
            f"d {x['delta']:4.1f}  {'OK' if x['ok'] else 'FAIL'}")

    worst_flat = max((x["delta"] for x in flat), default=0.0)
    worst_grad = max((x["delta"] for x in grad), default=0.0)
    lines += ["",
              f"worst flat error {worst_flat:.2f} (tolerance {TOL})",
              f"worst gradient error {worst_grad:.2f} (tolerance {GRAD_TOL})",
              f"overall: {'PASS' if fails == 0 else 'FAIL'} ({fails} sample(s))"]
    text = "\n".join(lines) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
