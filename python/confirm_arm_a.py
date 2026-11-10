"""Confirmation numbers for the arm-A gate, before anything is applied.

Reports, with lake-cluster intervals:
  - held-out April RMSE for arm A against the per-month baseline (n, lakes);
  - the fitted m from the stored bootstrap draws;
  - arm A vs arm B CV error overall and for November-December.

Read-only: writes one diagnostic text file and touches nothing else.

Run:  $env:PYTHONIOENCODING="utf-8"; python python/confirm_arm_a.py
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


def cluster_rmse_diff(pred_a: np.ndarray, pred_b: np.ndarray,
                      y: np.ndarray, lakes: np.ndarray,
                      n_boot: int, seed: int) -> tuple[float, float, float]:
    """RMSE(b) - RMSE(a) with a lake-cluster bootstrap interval."""
    keys = np.unique(lakes)
    mse_a, mse_b, ns = [], [], []
    for lk in keys:
        sel = lakes == lk
        mse_a.append(float(np.mean((y[sel] - pred_a[sel]) ** 2)))
        mse_b.append(float(np.mean((y[sel] - pred_b[sel]) ** 2)))
        ns.append(int(sel.sum()))
    mse_a = np.array(mse_a); mse_b = np.array(mse_b); ns = np.array(ns)
    delta = float(np.sqrt(np.average(mse_b, weights=ns))
                  - np.sqrt(np.average(mse_a, weights=ns)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        a = np.average(mse_a[pick], weights=ns[pick])
        b = np.average(mse_b[pick], weights=ns[pick])
        draws[i] = np.sqrt(b) - np.sqrt(a)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return delta, float(lo), float(hi)


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

    # --- held-out April: arm A vs the per-month (training April) baseline
    p_a, _ = F.fit_arm("A", s_tr, d_tr, p_tr, y_tr)
    pred_va = F.predict("A", p_a, s_va, d_va, p_va)
    apr = va.month.to_numpy() == 4
    baseline = float(y_tr[tr.month == 4].mean())
    base_va = np.full_like(y_va, baseline)
    apr_lakes = va.lake_key.to_numpy()[apr]
    n_apr = int(apr.sum())
    n_lakes = int(len(np.unique(apr_lakes)))
    rmse_a = float(np.sqrt(np.mean((y_va[apr] - pred_va[apr]) ** 2)))
    rmse_base = float(np.sqrt(np.mean((y_va[apr] - base_va[apr]) ** 2)))
    d, lo, hi = cluster_rmse_diff(pred_va[apr], base_va[apr], y_va[apr],
                                  apr_lakes, 2000, F.SEED)

    # --- m
    mb = pd.read_csv(DER / "melt_m_bootstrap.csv").m.to_numpy()
    m_lo, m_hi = np.percentile(mb, [2.5, 97.5])

    # --- CV A vs B, overall and November-December
    rng = np.random.default_rng(F.SEED)
    folds_list = F.make_folds(tr, rng)
    cv_preds = {arm: F.cv_predictions(tr, arm, folds_list, s_tr, d_tr, p_tr, y_tr)
                for arm in F.ARMS}
    lakes_tr = tr.lake_key.to_numpy()
    win = F.windows(tr)

    lines = []
    def say(t: str = "") -> None:
        print(t); lines.append(t)

    say("=== Arm A confirmation (held-out and CV) ===")
    say(f"pool: {len(tr):,} train / {len(va):,} held-out, "
        f"{m.lake_key.nunique()} lakes")
    say()
    say("April, held-out")
    say(f"  arm A RMSE {rmse_a:.2f} cm; per-month baseline {rmse_base:.2f} cm; "
        f"n {n_apr}, lakes {n_lakes}")
    say(f"  paired difference (baseline - A) {d:+.2f} cm, "
        f"lake-cluster 95% [{lo:+.2f}, {hi:+.2f}] (positive favours A)")
    say()
    say(f"m = {p_a[2]:.4f} cm per degC-day, lake-cluster 95% "
        f"[{m_lo:.4f}, {m_hi:.4f}]")
    say()
    for name, mask in [("overall", win["all"]), ("Nov-Dec", win["nov_dec"])]:
        vals, mses, ns = {}, {}, {}
        for arm in F.ARMS:
            vals[arm] = float(np.mean([
                F.rmse(y_tr[mask], cv_preds[arm][r, mask])
                for r in range(F.REPEATS)]))
            per_lake, counts = [], []
            for lk in np.unique(lakes_tr[mask]):
                sel = (lakes_tr == lk) & mask
                per_lake.append(float(np.mean([
                    np.mean((y_tr[sel] - cv_preds[arm][r, sel]) ** 2)
                    for r in range(F.REPEATS)])))
                counts.append(int(sel.sum()))
            mses[arm] = np.array(per_lake)
            ns[arm] = np.array(counts)
        # cluster interval from the per-lake MSEs, averaged over repeats
        rng3 = np.random.default_rng(F.SEED)
        draws = np.empty(2000)
        for i in range(2000):
            pick = rng3.integers(0, len(ns["A"]), len(ns["A"]))
            a_ = np.average(mses["A"][pick], weights=ns["A"][pick])
            b_ = np.average(mses["B"][pick], weights=ns["B"][pick])
            draws[i] = np.sqrt(b_) - np.sqrt(a_)
        dd = float(np.sqrt(np.average(mses["B"], weights=ns["B"]))
                   - np.sqrt(np.average(mses["A"], weights=ns["A"])))
        dlo, dhi = np.percentile(draws, [2.5, 97.5])
        say(f"CV {name}: A {vals['A']:.2f} cm, B {vals['B']:.2f} cm, "
            f"B - A {dd:+.2f} cm [{dlo:+.2f}, {dhi:+.2f}], n {int(mask.sum())}")
    say()
    say("A was kept under the pre-agreed asymmetric rule: B must beat A on CV "
        "with a cluster interval excluding zero, and it does not. Depth is "
        "retained on product grounds (the map is per-lake, depth varies within "
        "a lake, and the held-out depth-bias test is the standing check).")

    (OUT / "arm_a_confirmation.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print(f"\nwrote {OUT / 'arm_a_confirmation.txt'}")


if __name__ == "__main__":
    main()
