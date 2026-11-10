"""v2 held-out error against per-month climatology, with a paired cluster interval.

The baseline is the one already used for v1: mean training thickness for the
observation's calendar month, falling back to the pooled training mean where a
month has no training rows (May). The paired difference is
RMSE(v2) - RMSE(baseline); the interval resamples the 15 held-out lakes.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/v2_vs_climatology.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_melt as F  # noqa: E402
import verify_pdd_fields as V  # noqa: E402

ROOT = HERE.parent
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
B = 2000


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    m["month"] = m.date.dt.month
    m["pdd"] = [V.sample("pdd", V.PDD_DIR, int(r.date.year), float(r.lat),
                         float(r.lon), np.datetime64(r.date.date()))
                for r in m.itertuples()]
    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)

    s_tr, d_tr, p_tr, y_tr = (tr.cfdd.to_numpy(), tr.V0.to_numpy(),
                              tr.pdd.to_numpy(), tr.total_cm.to_numpy())
    s_va, d_va, p_va, y_va = (va.cfdd.to_numpy(), va.V0.to_numpy(),
                              va.pdd.to_numpy(), va.total_cm.to_numpy())
    p, _ = F.fit_arm("A", s_tr, d_tr, p_tr, y_tr)
    pred = F.predict("A", p, s_va, d_va, p_va)

    tr_month = tr.groupby("month").total_cm.mean()
    base = va.month.map(tr_month).fillna(float(y_tr.mean())).to_numpy()

    model_rmse = F.rmse(y_va, pred)
    base_rmse = F.rmse(y_va, base)
    point = model_rmse - base_rmse

    val_lakes = np.array(sorted(va.lake_key.unique()))
    groups = {lk: np.flatnonzero(va.lake_key.to_numpy() == lk) for lk in val_lakes}
    rng = np.random.default_rng(F.SEED)
    draws = np.empty(B)
    for i in range(B):
        pick = rng.choice(val_lakes, size=len(val_lakes), replace=True)
        idx = np.concatenate([groups[lk] for lk in pick])
        draws[i] = (F.rmse(y_va[idx], pred[idx])
                    - F.rmse(y_va[idx], base[idx]))
    lo, hi = np.percentile(draws, [2.5, 97.5])

    rows = []
    for month, g in va.groupby("month"):
        pr = F.predict("A", p, g.cfdd.to_numpy(), g.V0.to_numpy(),
                       g.pdd.to_numpy())
        bm = g.month.map(tr_month).fillna(float(y_tr.mean())).to_numpy()
        rows.append(dict(month=int(month), n=len(g),
                         model_rmse=F.rmse(g.total_cm.to_numpy(), pr),
                         baseline_rmse=F.rmse(g.total_cm.to_numpy(), bm),
                         delta=F.rmse(g.total_cm.to_numpy(), bm)
                         - F.rmse(g.total_cm.to_numpy(), pr)))
    tab = pd.DataFrame(rows).round(2)
    tab.to_csv(DER / "v2_vs_climatology.csv", index=False)

    lines = []
    def say(t: str = "") -> None:
        print(t); lines.append(t)

    say("=== v2 held-out error against per-month climatology ===")
    say(f"n {len(y_va):,}, {len(val_lakes)} held-out lakes")
    say(f"v2 RMSE {model_rmse:.2f} cm; per-month baseline {base_rmse:.2f} cm")
    say(f"paired difference v2 - baseline {point:+.2f} cm, "
        f"lake-cluster 95% [{lo:+.2f}, {hi:+.2f}], {B} draws")
    say()
    say("by month (delta = baseline RMSE - v2 RMSE, positive favours v2):")
    say(tab.to_string(index=False))
    say()
    say("May has no training rows; the baseline there is the pooled training "
        "mean. November has 24 training rows.")
    (OUT / "v2_vs_climatology.txt").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    print(f"\nwrote {OUT / 'v2_vs_climatology.txt'}")


if __name__ == "__main__":
    main()
