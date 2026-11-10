"""Gate D corrections: cluster bootstrap, C1 = 0 fit, per-month baseline.

Runs three analyses against the fitted V0 model in `fit_model.py`:

1. Lake-cluster bootstrap of the held-out error statistics. Resamples the 15
   held-out lakes with replacement (so a lake, not an observation, is the
   independent unit), predicts with the point-estimate parameters, and takes
   percentile intervals for RMSE, MAE, bias, the depth-bias slope and the snow
   slope. This interval is conditional on the fitted parameters; parameter
   uncertainty is reported separately by the training-lake bootstrap.

2. Training-lake bootstrap. Resamples the 59 training lakes with replacement,
   refits all three constants each time, and reports percentile intervals for
   k, C0 and C1. This is the cluster-robust replacement for the Gauss-Newton
   standard errors printed by `fit_model.py`, which assume independent
   observations and are therefore too narrow.

3. C1 = 0 fit (k and C0 free, C0 >= 0) and the per-month constant baseline
   (training mean thickness for each calendar month) evaluated on held-out
   observations. Residual is observed - predicted everywhere.

Outputs
  data/derived/diagnostics/bootstrap_report.txt
  data/derived/bootstrap_intervals.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/bootstrap_model.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_model as fm  # noqa: E402

ROOT = HERE.parent
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

B = 2000
SEED = 20261108

STATS = ["rmse", "mae", "bias", "depth_slope", "snow_slope"]


def refit(s: np.ndarray, d: np.ndarray, y: np.ndarray, x0: np.ndarray) -> np.ndarray | None:
    """Single-start fit from the point estimate; none if it does not converge."""
    try:
        r = least_squares(
            lambda p: fm.predict(p, s, d) - y, x0,
            bounds=([0.01, 0.0, -50.0], [20.0, 3000.0, 100.0]),
            max_nfev=2000,
        )
    except Exception:  # noqa: BLE001
        return None
    return r.x if np.all(np.isfinite(r.x)) else None


def fit_c1_zero(s: np.ndarray, d: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """h = k * sqrt(max(0, S - C0)) with C0 >= 0."""
    best = None
    for c0 in (0.0, 50.0, 100.0, 200.0):
        arg = np.maximum(0.0, s - c0)
        scale = np.sqrt(arg.mean())
        k0 = float(np.clip(y.mean() / scale, 0.1, 10.0)) if scale > 0 else 1.0
        try:
            r = least_squares(
                lambda p: p[0] * np.sqrt(np.maximum(0.0, s - p[1])) - y,
                [k0, c0], bounds=([0.01, 0.0], [20.0, 3000.0]), max_nfev=4000,
            )
        except Exception:  # noqa: BLE001
            continue
        rmse = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rmse < best[1]:
            best = (r.x, rmse)
    if best is None:
        raise RuntimeError("C1=0 fit failed")
    return best[0], best[1]


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan
    try:
        with np.errstate(all="ignore"):
            return fm.ols(x, y)[0]
    except Exception:  # noqa: BLE001
        return np.nan


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    tr = m[(m.split == "train") & m.V0.notna()].copy()
    va = m[(m.split == "val") & m.V0.notna()].copy()
    s_tr, d_tr, y_tr = tr.cfdd.to_numpy(), tr.V0.to_numpy(), tr.total_cm.to_numpy()
    s_va, d_va, y_va = va.cfdd.to_numpy(), va.V0.to_numpy(), va.total_cm.to_numpy()

    p_hat, _ = fm.fit_one(s_tr, d_tr, y_tr, 0.0)
    se_k, se_c0, se_c1 = fm.param_se(None, s_tr, d_tr, y_tr, p_hat)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=== V0 point fit ===")
    say(f"k  = {p_hat[0]:.4f}  (Gauss-Newton SE {se_k:.4f})")
    say(f"C0 = {p_hat[1]:.2f}    (Gauss-Newton SE {se_c0:.2f})")
    say(f"C1 = {p_hat[2]:.2f}  (Gauss-Newton SE {se_c1:.2f})")
    say(f"train RMSE {np.sqrt(np.mean((y_tr - fm.predict(p_hat, s_tr, d_tr)) ** 2)):.2f} cm")
    say()

    # ------------------------------------------------------------------
    # 1. Held-out lake-cluster bootstrap (parameters fixed)
    # ------------------------------------------------------------------
    say(f"=== held-out lake-cluster bootstrap ({B} replicates, seed {SEED}) ===")
    say("resampling unit = held-out lake (15 lakes); parameters fixed at the point fit")
    val_lakes = sorted(va.lake_key.unique())
    by_lake = {k: g for k, g in va.groupby("lake_key")}
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in STATS}
    for _ in range(B):
        picks = rng.choice(val_lakes, size=len(val_lakes), replace=True)
        g = pd.concat([by_lake[k] for k in picks], ignore_index=True)
        y = g.total_cm.to_numpy()
        pr = fm.predict(p_hat, g.cfdd.to_numpy(), g.V0.to_numpy())
        r = y - pr
        boot["rmse"].append(float(np.sqrt(np.mean(r ** 2))))
        boot["mae"].append(float(np.mean(np.abs(r))))
        boot["bias"].append(float(np.mean(r)))
        boot["depth_slope"].append(safe_slope(np.log(g.V0.to_numpy()), r))
        snow = g[g.snow_cm.notna()]
        boot["snow_slope"].append(
            safe_slope(snow.snow_cm.to_numpy(), (snow.total_cm - fm.predict(
                p_hat, snow.cfdd.to_numpy(), snow.V0.to_numpy())).to_numpy())
        )

    resid = y_va - fm.predict(p_hat, s_va, d_va)
    point = dict(
        rmse=float(np.sqrt(np.mean(resid ** 2))),
        mae=float(np.mean(np.abs(resid))),
        bias=float(np.mean(resid)),
        depth_slope=safe_slope(np.log(d_va), resid),
        snow_slope=safe_slope(va.loc[va.snow_cm.notna(), "snow_cm"].to_numpy(),
                              (va.loc[va.snow_cm.notna(), "total_cm"]
                               - fm.predict(p_hat, va.loc[va.snow_cm.notna(), "cfdd"].to_numpy(),
                                            va.loc[va.snow_cm.notna(), "V0"].to_numpy())).to_numpy()),
    )
    rows = []
    for stat in STATS:
        arr = np.asarray(boot[stat], float)
        arr = arr[np.isfinite(arr)]
        lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
        rows.append(dict(statistic=stat, point=point[stat], boot_median=med,
                         lo95=lo, hi95=hi, n=len(arr), method="held-out lake cluster"))
        say(f"{stat:12s} point {point[stat]:8.2f}  cluster 95% [{lo:8.2f}, {hi:8.2f}]  "
            f"(median {med:8.2f}, {len(arr)} usable replicates)")

    # ------------------------------------------------------------------
    # 2. Training-lake bootstrap, refit each replicate
    # ------------------------------------------------------------------
    say()
    say(f"=== training-lake bootstrap ({B} replicates, seed {SEED}) ===")
    say("resampling unit = training lake (59 lakes); refit k, C0, C1 each replicate")
    train_lakes = sorted(tr.lake_key.unique())
    tr_by_lake = {k: g for k, g in tr.groupby("lake_key")}
    param_boot = []
    for _ in range(B):
        picks = rng.choice(train_lakes, size=len(train_lakes), replace=True)
        g = pd.concat([tr_by_lake[k] for k in picks], ignore_index=True)
        pb = refit(g.cfdd.to_numpy(), g.V0.to_numpy(), g.total_cm.to_numpy(), p_hat)
        if pb is not None:
            param_boot.append(pb)
    param_boot = np.asarray(param_boot)
    say(f"usable replicates: {len(param_boot)}")
    for i, name in enumerate(("k", "C0", "C1")):
        lo, med, hi = np.percentile(param_boot[:, i], [2.5, 50, 97.5])
        say(f"{name:2s} point {p_hat[i]:9.3f}  cluster 95% [{lo:9.3f}, {hi:9.3f}]  "
            f"(median {med:9.3f})")
        rows.append(dict(statistic=f"param_{name}", point=p_hat[i], boot_median=med,
                         lo95=lo, hi95=hi, n=len(param_boot),
                         method="training lake cluster, refit"))
    frac_pos = float(np.mean(param_boot[:, 1] > 1e-6)) if len(param_boot) else np.nan
    say(f"replicates with C0 > 0: {frac_pos:.3f} (point fit is pinned at the C0 = 0 bound)")
    say()
    say("NOTE: the SEs printed in fit_report.txt (k +/- 0.0115, C1 +/- 4.49) are")
    say("Gauss-Newton, i.e. observation-independent. They are not cluster bootstrap")
    say("SEs and understate uncertainty because observations repeat within lakes.")
    say()

    # ------------------------------------------------------------------
    # 3. C1 = 0 fit
    # ------------------------------------------------------------------
    say("=== C1 = 0 fit (no depth term; k and C0 free, C0 >= 0) ===")
    p0, train_rmse0 = fit_c1_zero(s_tr, d_tr, y_tr)
    pred0 = p0[0] * np.sqrt(np.maximum(0.0, s_va - p0[1]))
    r0 = y_va - pred0
    met0 = fm.metrics(y_va, pred0)
    say(f"k = {p0[0]:.4f}  C0 = {p0[1]:.2f} degC-day  train RMSE {train_rmse0:.2f} cm")
    say(f"held-out: RMSE {met0['rmse']:.2f}  MAE {met0['mae']:.2f}  "
        f"bias {met0['bias']:+.2f}  R2 {met0['r2']:.3f}")
    say(f"held-out depth slope {safe_slope(np.log(d_va), r0):+.2f} cm per ln(m)")
    say(f"versus V0 (C1 = {p_hat[2]:.2f}): RMSE {point['rmse']:.2f}, "
        f"depth slope {point['depth_slope']:+.2f}")
    rows.append(dict(statistic="c1_zero_heldout_rmse", point=met0["rmse"],
                     boot_median=np.nan, lo95=np.nan, hi95=np.nan, n=len(y_va),
                     method="C1=0 fit"))
    say()

    # ------------------------------------------------------------------
    # 4. Per-month constant baseline
    # ------------------------------------------------------------------
    say("=== held-out error by month, model vs per-month training-mean baseline ===")
    say("residual = observed - predicted; baseline = mean training thickness for that")
    say("month, falling back to the pooled training mean where a month has no training")
    say("observations (May: zero training observations; November: 24)")
    tr_month = tr.groupby(tr.date.dt.month).total_cm.mean()
    va_month = va.date.dt.month
    base = va_month.map(tr_month)
    if base.isna().any():
        base = base.fillna(float(y_tr.mean()))
    va = va.assign(month=va_month, baseline=base,
                   pred=fm.predict(p_hat, s_va, d_va),
                   resid=resid)
    month_rows = []
    for month, g in va.groupby("month"):
        mr = fm.metrics(g.total_cm.to_numpy(), g.pred.to_numpy())
        br = fm.metrics(g.total_cm.to_numpy(), g.baseline.to_numpy())
        month_rows.append(dict(month=int(month), n=len(g),
                               model_rmse=mr["rmse"], model_bias=mr["bias"],
                               baseline_rmse=br["rmse"], baseline_bias=br["bias"],
                               delta_rmse=br["rmse"] - mr["rmse"]))
    say(pd.DataFrame(month_rows).round(2).to_string(index=False))
    overall_base = fm.metrics(y_va, base.to_numpy())
    say(f"overall per-month baseline RMSE {overall_base['rmse']:.2f} cm "
        f"(model {point['rmse']:.2f} cm)")
    say(f"overall single-mean baseline RMSE {fm.metrics(y_va, np.full(len(y_va), y_tr.mean()))['rmse']:.2f} cm")
    say()
    boot_month = []
    for _ in range(B):
        picks = rng.choice(val_lakes, size=len(val_lakes), replace=True)
        g = pd.concat([by_lake[k] for k in picks], ignore_index=True)
        y = g.total_cm.to_numpy()
        rm = float(np.sqrt(np.mean((y - fm.predict(p_hat, g.cfdd.to_numpy(),
                                                    g.V0.to_numpy())) ** 2)))
        gb = g.date.dt.month.map(tr_month).fillna(float(y_tr.mean())).to_numpy()
        rb = float(np.sqrt(np.mean((y - gb) ** 2)))
        boot_month.append(rm - rb)
    lo_m, med_m, hi_m = np.percentile(boot_month, [2.5, 50, 97.5])
    say(f"paired difference RMSE(model) - RMSE(per-month baseline): point "
        f"{point['rmse'] - overall_base['rmse']:+.2f} cm, "
        f"cluster 95% [{lo_m:+.2f}, {hi_m:+.2f}]")
    say("negative = model better; an interval spanning zero means no established")
    say("skill over seasonal climatology.")
    say()
    rows.append(dict(statistic="paired_rmse_diff_vs_month_baseline",
                     point=point["rmse"] - overall_base["rmse"], boot_median=med_m,
                     lo95=lo_m, hi95=hi_m, n=len(y_va),
                     method="held-out lake cluster, paired"))

    # ------------------------------------------------------------------
    # 5. Depth lever at C1 uncertainty
    # ------------------------------------------------------------------
    say("=== depth lever at C1 uncertainty (median held-out CFDD = "
        f"{np.median(s_va):.0f} degC-day) ===")
    c1_lo_gn = p_hat[2] - 2 * se_c1
    c1_hi_gn = p_hat[2] + 2 * se_c1
    c1_lo_b = float(np.percentile(param_boot[:, 2], 2.5))
    c1_hi_b = float(np.percentile(param_boot[:, 2], 97.5))
    say(f"C1 - 2 GN SE: {c1_lo_gn:.2f}   C1 + 2 GN SE: {c1_hi_gn:.2f}")
    say(f"C1 cluster 2.5%: {c1_lo_b:.2f}   C1 cluster 97.5%: {c1_hi_b:.2f}")
    med_s = float(np.median(s_va))
    lever = []
    for dd in (3, 10, 30):
        def h(c1: float) -> float:
            return float(fm.predict([p_hat[0], p_hat[1], c1],
                                    np.array([med_s]), np.array([dd]))[0])
        lever.append(dict(depth_m=dd, h_point=h(p_hat[2]),
                          h_gn_lo=h(c1_lo_gn), h_gn_hi=h(c1_hi_gn),
                          h_cluster_lo=h(c1_lo_b), h_cluster_hi=h(c1_hi_b)))
    say(pd.DataFrame(lever).round(2).to_string(index=False))
    say()

    # ------------------------------------------------------------------
    # 6. Post-hoc flowing-water subset, cluster bootstrap
    # ------------------------------------------------------------------
    say("=== POST HOC: held-out subset excluding flowing-water stations 127, 147, 314 ===")
    say("selection rule: stations 127 (Portage Lake-Keweenaw Waterway), 147 (East")
    say("Nipigon Bay) and 314 (DeTour-St. Marys River); all are validation-only")
    say("channel/river sites identified after inspecting the per-station residual table")
    flow = {"g00803:127", "g00803:147", "g00803:314"}
    sub = va[~va.lake_key.isin(flow)]
    sub_lakes = sorted(sub.lake_key.unique())
    sub_by_lake = {k: g for k, g in sub.groupby("lake_key")}
    boot_sub = {"rmse": [], "depth_slope": []}
    for _ in range(B):
        picks = rng.choice(sub_lakes, size=len(sub_lakes), replace=True)
        g = pd.concat([sub_by_lake[k] for k in picks], ignore_index=True)
        y = g.total_cm.to_numpy()
        pr = fm.predict(p_hat, g.cfdd.to_numpy(), g.V0.to_numpy())
        r = y - pr
        boot_sub["rmse"].append(float(np.sqrt(np.mean(r ** 2))))
        boot_sub["depth_slope"].append(safe_slope(np.log(g.V0.to_numpy()), r))
    r_sub = sub.total_cm.to_numpy() - fm.predict(p_hat, sub.cfdd.to_numpy(), sub.V0.to_numpy())
    sub_point = dict(rmse=float(np.sqrt(np.mean(r_sub ** 2))),
                     depth_slope=safe_slope(np.log(sub.V0.to_numpy()), r_sub))
    for stat, val in sub_point.items():
        arr = np.asarray(boot_sub[stat], float)
        arr = arr[np.isfinite(arr)]
        lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
        say(f"{stat:12s} point {val:8.2f}  cluster 95% [{lo:8.2f}, {hi:8.2f}]  "
            f"({len(sub_lakes)} lakes, {len(sub)} obs)")
        rows.append(dict(statistic=f"posthoc_{stat}", point=val, boot_median=med,
                         lo95=lo, hi95=hi, n=len(sub),
                         method="held-out lake cluster, post-hoc subset"))
    say()

    # ------------------------------------------------------------------
    # 7. Paired depth-term test: V0 minus C1=0 on the same lakes
    # ------------------------------------------------------------------
    say("=== is the depth term earning its place? paired held-out comparison ===")
    say("same fixed parameters per model; resample held-out lakes, take RMSE(V0) - RMSE(C1=0)")
    boot_pair = []
    for _ in range(B):
        picks = rng.choice(val_lakes, size=len(val_lakes), replace=True)
        g = pd.concat([by_lake[k] for k in picks], ignore_index=True)
        y = g.total_cm.to_numpy()
        r_v0 = y - fm.predict(p_hat, g.cfdd.to_numpy(), g.V0.to_numpy())
        pr0 = p0[0] * np.sqrt(np.maximum(0.0, g.cfdd.to_numpy() - p0[1]))
        r_c0 = y - pr0
        boot_pair.append(float(np.sqrt(np.mean(r_v0 ** 2)) - np.sqrt(np.mean(r_c0 ** 2))))
    lo, med, hi = np.percentile(boot_pair, [2.5, 50, 97.5])
    point_diff = point["rmse"] - met0["rmse"]
    say(f"point difference {point_diff:+.2f} cm (positive = V0 worse); "
        f"cluster 95% [{lo:+.2f}, {hi:+.2f}]")
    say("an interval that includes zero means the held-out data do not establish")
    say("that the depth term improves accuracy.")
    say()
    rows.append(dict(statistic="paired_rmse_diff_v0_minus_c1zero", point=point_diff,
                     boot_median=med, lo95=lo, hi95=hi, n=len(y_va),
                     method="held-out lake cluster, paired"))

    pd.DataFrame(rows).to_csv(DER / "bootstrap_intervals.csv", index=False)
    (OUT / "bootstrap_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'bootstrap_report.txt'}")
    print(f"wrote {DER / 'bootstrap_intervals.csv'}")


if __name__ == "__main__":
    main()
