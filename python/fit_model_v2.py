"""Canonical v2 model: arm A, fitted and verified.

v2 is v1 with the melt term adopted:

    h = max(0, k * sqrt(max(0, S - C1 * ln d)) - m * P)      C0 = 0

S is accumulated freezing degree-days since 1 October; P is the production H
field, cumulative positive degree-days from 1 March, exactly zero outside
1 March - 31 July; d is point water depth in metres; h is ice thickness in cm.

Arm A was retained at the gate under the pre-agreed asymmetric rule (B had to
beat A on lake-grouped CV with a cluster interval excluding zero; it did not).
Depth is retained on product grounds.

This script is the canonical v2 fit. It:
  1. refits arm A on the training lakes and asserts the parameters match the
     gate values;
  2. writes the canonical v2 outputs as new files (v1 is archived in
     data/derived/backups/pre_v2_melt/);
  3. repeats the count verification against v1 and the pre-snow backup;
  4. scores 8-inch (20.32 cm) and 4-inch (10.16 cm) classification accuracy on
     held-out rows with lake-cluster intervals.

Outputs
  data/derived/fit_results_v2.csv
  data/derived/validation_predictions_v2.csv
  data/derived/model_parameters.csv
  data/derived/v2_classification.csv
  data/derived/diagnostics/fit_report_v2.txt
  data/derived/diagnostics/v2_count_verification.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/fit_model_v2.py
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
BACKUP = DER / "backups" / "pre_g00803_snow"
V1_BACKUP = DER / "backups" / "pre_v2_melt"

GATE = dict(k=1.6865, C1=34.38, C0=0.0, m=0.3318)
THRESHOLDS = [("8 in", 8 * 2.54), ("4 in", 4 * 2.54)]


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

    p, train_rmse = F.fit_arm("A", s_tr, d_tr, p_tr, y_tr)
    assert abs(p[0] - GATE["k"]) < 5e-3, p[0]
    assert abs(p[1] - GATE["C1"]) < 5e-2, p[1]
    assert abs(p[2] - GATE["m"]) < 5e-4, p[2]

    pred_tr = F.predict("A", p, s_tr, d_tr, p_tr)
    pred_va = F.predict("A", p, s_va, d_va, p_va)
    met = F.metrics(y_va, pred_va)

    payload = dict(model="v2", arm="A", k=p[0], C0=0.0, C1=p[1], m=p[2],
                   thaw_start="03-01", thaw_end="07-31",
                   n_train=len(tr), n_val=len(va), lakes=int(m.lake_key.nunique()),
                   train_rmse=train_rmse, val_rmse=met["rmse"],
                   val_mae=met["mae"], val_bias=met["bias"], val_r2=met["r2"])
    pd.DataFrame([payload]).to_csv(DER / "fit_results_v2.csv", index=False)
    pd.DataFrame([payload]).to_csv(DER / "model_parameters.csv", index=False)

    out = pd.DataFrame({
        "lake_key": va.lake_key, "date": va.date, "region": va.region,
        "source": va.source, "total_cm": y_va,
        "pred_cm": np.round(pred_va, 2),
        "residual_cm": np.round(y_va - pred_va, 2),
        "pdd": p_va, "cfdd": s_va, "depth_m": d_va, "month": va.month,
    })
    out.to_csv(DER / "validation_predictions_v2.csv", index=False)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=== v2 canonical fit (arm A with melt) ===")
    say(f"k={p[0]:.4f}  C0=0 (pinned)  C1={p[1]:.2f}  m={p[2]:.4f}")
    say(f"train RMSE {train_rmse:.2f} cm; held-out RMSE {met['rmse']:.2f} cm, "
        f"MAE {met['mae']:.2f}, bias {met['bias']:+.2f}, R2 {met['r2']:.3f}")
    say(f"n train {len(tr):,}; n held-out {len(va):,}; "
        f"lakes {m.lake_key.nunique()}")
    say()

    # -------------------------------------------------- count verification
    obs = pd.read_csv(DER / "observations.csv", parse_dates=["date"])
    back = pd.read_csv(BACKUP / "training_matrix.csv", parse_dates=["date"])
    v1 = pd.read_csv(V1_BACKUP / "fit_results.csv")
    v1_pred = pd.read_csv(V1_BACKUP / "validation_predictions.csv",
                          parse_dates=["date"])
    checks = []
    checks.append(("observations rows", len(obs), 4785, len(obs) == 4785))
    checks.append(("observations lakes", obs.lake_key.nunique(), 74,
                   obs.lake_key.nunique() == 74))
    checks.append(("matrix rows", len(m), 4766, len(m) == 4766))
    checks.append(("train rows", len(tr), 3656, len(tr) == 3656))
    checks.append(("held-out rows", len(va), 1110, len(va) == 1110))
    checks.append(("v2 train rows == v1", len(tr),
                   int(v1.loc[v1.variant == "V0", "n_train"].iloc[0]),
                   len(tr) == int(v1.loc[v1.variant == "V0", "n_train"].iloc[0])))

    cur_keys = set(zip(m.lake_key, m.date))
    bak_keys = set(zip(back.lake_key, back.date))
    checks.append(("matrix keys == pre-snow backup", len(cur_keys),
                   len(bak_keys), cur_keys == bak_keys))

    shared = [c for c in back.columns if c not in ("lake_key", "date", "snow_cm")]
    bi = back.set_index(["lake_key", "date"]).sort_index()[shared]
    ci = m.set_index(["lake_key", "date"]).sort_index()[shared]
    same = bi.index.equals(ci.index)
    diffs = [] if same else ["<index mismatch>"]
    if same:
        for col in shared:
            if bi[col].dtype.kind in "fi":
                eq = bool(np.allclose(bi[col].to_numpy(dtype=float),
                                      ci[col].to_numpy(dtype=float),
                                      equal_nan=True))
            else:
                eq = bool((bi[col].astype(str) == ci[col].astype(str)).all())
            if not eq:
                same = False
                diffs.append(col)
    checks.append(("matrix columns except snow_cm unchanged",
                   len(shared) - len(diffs), len(shared), same))

    v1_keys = set(zip(v1_pred.lake_key, v1_pred.date))
    v2_keys = set(zip(out.lake_key, out.date))
    checks.append(("held-out keys == v1", len(v2_keys), len(v1_keys),
                   v2_keys == v1_keys))

    ver = []
    for label, got, want, ok in checks:
        ver.append(f"{'OK ' if ok else 'BAD'}  {label:42s} got {got}  want {want}")
    all_ok = all(c[3] for c in checks)
    ver.append(f"\nall checks passed: {all_ok}")
    if diffs:
        ver.append(f"columns that differ from backup: {diffs}")
    (OUT / "v2_count_verification.txt").write_text("\n".join(ver) + "\n",
                                                   encoding="utf-8")
    say("=== count verification ===")
    say("\n".join(ver))
    say()

    # ---------------------------------------------------- classification accuracy
    lakes = va.lake_key.to_numpy()
    rows = []
    for name, t in THRESHOLDS:
        correct = ((pred_va >= t) == (y_va >= t)).astype(float)
        acc = float(correct.mean())
        obs_above = int((y_va >= t).sum())
        pred_above = int((pred_va >= t).sum())
        fp = int(((pred_va >= t) & (y_va < t)).sum())
        fn = int(((pred_va < t) & (y_va >= t)).sum())
        rng = np.random.default_rng(F.SEED)
        groups = va.groupby("lake_key").indices
        keys = list(groups)
        draws = np.empty(2000)
        for i in range(2000):
            pick = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([groups[k] for k in pick])
            draws[i] = correct[idx].mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(threshold=name, threshold_cm=round(t, 2),
                         n=len(y_va), accuracy=round(acc, 4),
                         acc_lo=round(float(lo), 4), acc_hi=round(float(hi), 4),
                         observed_above=obs_above, predicted_above=pred_above,
                         false_above=fp, missed_above=fn,
                         n_lakes=len(keys)))
    cls = pd.DataFrame(rows)
    cls.to_csv(DER / "v2_classification.csv", index=False)
    say("=== held-out classification accuracy, lake-cluster 95% ===")
    say(cls.to_string(index=False))
    say()
    say("false_above = predicted at or above the threshold but observed below;")
    say("missed_above = predicted below but observed at or above.")
    say()
    say("v2 is canonical. v1 outputs are archived in "
        "data/derived/backups/pre_v2_melt/.")

    (OUT / "fit_report_v2.txt").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    print(f"\nwrote {OUT / 'fit_report_v2.txt'}")


if __name__ == "__main__":
    main()
