"""Read-only skill diagnostics: anomalies, low-CFDD depth lever, clamped scale.

Three analyses that make no model change and touch no parameter. They run on
the frozen split and the fitted V0 parameters already in `fit_results.csv`.

  (a) Anomaly skill. The plain RMSE comparison against a per-month climatology
      is really a test of levels. This asks the harder question: does the model
      track *departures* from climatology? Anomalies are taken two ways -
      month-demeaned (removes the seasonal cycle) and lake+month-demeaned
      (removes between-lake differences as well), so the second isolates
      within-lake year-to-year variation. Reported: correlation, R^2, the OLS
      slope of observed anomaly on predicted anomaly with a lake-cluster
      bootstrap interval, and the anomaly RMSE against zero (i.e. against
      climatology).

  (b) Depth lever at low CFDD. The v1 report quotes the 3 m -> 30 m lever at
      the median held-out CFDD (618). Here it is quoted at S = 50, 100 and 200
      °C-day, where the depth term matters most because sqrt is steep, across
      the C1 lake-cluster interval.

  (c) V0 versus C1 = 0 on November-December and on the clamped display scale.
      The map clamps at 20.3 cm (8 in). Two clamped metrics are reported:
      error of min(prediction, 20.3) against raw observation (what a viewer
      comparing the map to a measured thickness would see), and error with
      both sides clamped (how well the displayed value matches the displayed
      truth).

Outputs
  data/derived/diagnostics/skill_diagnostics.txt
  data/derived/skill_diagnostics.csv

Run:  $env:PYTHONIOENCODING="utf-8"; python python/skill_diagnostics.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit_model as fm  # noqa: E402
import bootstrap_model as bm  # noqa: E402

ROOT = HERE.parent
DER = ROOT / "data" / "derived"
DIAG = DER / "diagnostics"
B = 2000
SEED = 20261108
CLAMP = 20.3


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    m = m[m.V0.notna()].copy()
    fit = pd.read_csv(DER / "fit_results.csv")
    v0 = fit[fit.variant == "V0"].iloc[0]
    p = np.array([v0.k, v0.C0, v0.C1])

    tr = m[m.split == "train"].copy()
    va = m[m.split == "val"].copy()
    va["pred"] = fm.predict(p, va.cfdd.to_numpy(), va.V0.to_numpy())
    va["resid"] = va.total_cm - va.pred

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    val_lakes = sorted(va.lake_key.unique())
    by_lake = {k: g for k, g in va.groupby("lake_key")}

    # ------------------------------------------------------------------
    # (a) anomaly skill
    # ------------------------------------------------------------------
    say(f"model: k={v0.k:.4f} C0={v0.C0:.2f} C1={v0.C1:.2f} "
        f"({len(va)} held-out obs, {len(val_lakes)} lakes)")
    say()

    tr_month = tr.groupby(tr.date.dt.month).total_cm.mean()
    va["month"] = va.date.dt.month
    va["c_month"] = va.month.map(tr_month).fillna(float(tr.total_cm.mean()))

    # Held-out lakes have no training observations at all (the split is by
    # lake), so a training lake-month mean cannot be used for them. For the
    # within-lake variant, centre both sides on the held-out lake-month cell
    # mean. That is descriptive and uses no fitted quantity.
    va["cell"] = va.lake_key + "|" + va.month.astype(str)
    cell_y = va.groupby("cell").total_cm.transform("mean")
    cell_p = va.groupby("cell").pred.transform("mean")
    va["y_within"] = va.total_cm - cell_y
    va["p_within"] = va.pred - cell_p

    say("=== (a) anomaly skill on held-out lakes ===")
    say("month-demeaned: anomaly = value - training per-month mean, applied to")
    say("both observed and predicted, so its residual equals the ordinary one.")
    say("within-lake-month: both sides centred on the held-out cell mean, which")
    say("removes between-lake and seasonal differences and leaves the within-cell")
    say("variation the model would have to get right.")
    say()

    pairs = [
        ("month-demeaned", va.total_cm - va.c_month, va.pred - va.c_month),
        ("within-lake-month", va.y_within, va.p_within),
    ]
    for label, y_series, p_series in pairs:
        y_an = y_series.to_numpy(dtype=float)
        p_an = p_series.to_numpy(dtype=float)
        good = np.isfinite(y_an) & np.isfinite(p_an)
        y_an, p_an = y_an[good], p_an[good]
        sub = va[good].copy()
        r = corr(p_an, y_an)
        r2 = float(np.mean((y_an - y_an.mean()) ** 2))
        mse_res = float(np.mean((y_an - p_an) ** 2))
        skill = 1 - mse_res / r2 if r2 > 0 else np.nan
        b = fm.ols(p_an, y_an)[0] if len(p_an) > 2 and np.ptp(p_an) > 0 else np.nan

        # cluster bootstrap of correlation and slope
        corr_boot, slope_boot = [], []
        idx_by_lake = {k: np.flatnonzero(sub.lake_key.to_numpy() == k)
                       for k in sub.lake_key.unique()}
        lakes = list(idx_by_lake)
        for _ in range(B):
            picks = rng.choice(lakes, size=len(lakes), replace=True)
            idx = np.concatenate([idx_by_lake[k] for k in picks])
            yy, pp = y_an[idx], p_an[idx]
            c = corr(pp, yy)
            if np.isfinite(c):
                corr_boot.append(c)
            if np.ptp(pp) > 0:
                slope_boot.append(float(fm.ols(pp, yy)[0]))
        c_lo, c_hi = np.percentile(corr_boot, [2.5, 97.5])
        s_lo, s_hi = np.percentile(slope_boot, [2.5, 97.5])
        say(f"{label}: n={len(y_an)}")
        say(f"  corr {r:+.3f}  cluster 95% [{c_lo:+.3f}, {c_hi:+.3f}]")
        say(f"  slope of observed on predicted anomaly {b:+.3f}  "
            f"cluster 95% [{s_lo:+.3f}, {s_hi:+.3f}]   (1.0 = perfect)")
        say(f"  anomaly RMSE {np.sqrt(mse_res):.2f} cm vs climatology-anomaly "
            f"spread {np.sqrt(r2):.2f} cm; skill score {skill:+.3f}")
        say("  skill score = 1 - MSE(model anomaly) / MSE(climatology); <= 0 means")
        say("  the model does not beat simply predicting the climatology anomaly (zero).")
        rows.append(dict(analysis="anomaly", group=label, n=len(y_an),
                         statistic="corr", point=r, lo95=c_lo, hi95=c_hi))
        rows.append(dict(analysis="anomaly", group=label, n=len(y_an),
                         statistic="slope", point=b, lo95=s_lo, hi95=s_hi))
        rows.append(dict(analysis="anomaly", group=label, n=len(y_an),
                         statistic="skill_score", point=skill, lo95=np.nan, hi95=np.nan))
        say()

    # January-March only, the window where the model beats climatology
    jfm = va[(va.month >= 1) & (va.month <= 3)].copy()
    y_an = (jfm.total_cm - jfm.c_month).to_numpy()
    p_an = (jfm.pred - jfm.c_month).to_numpy()
    say(f"January-March only: n={len(jfm)}  corr {corr(p_an, y_an):+.3f}  "
        f"slope {fm.ols(p_an, y_an)[0]:+.3f}")
    say()

    # ------------------------------------------------------------------
    # (b) depth lever at low CFDD
    # ------------------------------------------------------------------
    ci = pd.read_csv(DER / "bootstrap_intervals.csv")
    c1_lo = float(ci.loc[ci.statistic == "param_C1", "lo95"].iloc[0])
    c1_hi = float(ci.loc[ci.statistic == "param_C1", "hi95"].iloc[0])

    say("=== (b) depth lever at low CFDD, C1 lake-cluster interval ===")
    say(f"C1 point {v0.C1:.2f}; cluster 95% [{c1_lo:.2f}, {c1_hi:.2f}]; "
        f"k and C0 held at the point fit")
    say()
    lever_rows = []
    for s in (50.0, 100.0, 200.0, 618.0):
        for dd in (3.0, 10.0, 30.0):
            def h(c1: float) -> float:
                return float(fm.predict([v0.k, v0.C0, c1],
                                        np.array([s]), np.array([dd]))[0])
            lever_rows.append(dict(S=s, depth_m=dd, h_point=h(v0.C1),
                                   h_c1_lo=h(c1_lo), h_c1_hi=h(c1_hi)))
    lev = pd.DataFrame(lever_rows)
    wide = lev.pivot(index="S", columns="depth_m", values="h_point").round(2)
    wide.columns = [f"h({int(c)} m)" for c in wide.columns]
    say("predicted thickness, cm (point C1):")
    say(wide.to_string())
    say()
    span = lev.pivot(index="S", columns="depth_m", values=["h_c1_lo", "h_c1_hi"])
    say("3 m -> 30 m lever, cm:")
    for s in (50.0, 100.0, 200.0, 618.0):
        r = lev[lev.S == s].set_index("depth_m")
        say(f"  S={s:5.0f}: point {r.loc[30,'h_point']-r.loc[3,'h_point']:5.2f}   "
            f"C1 lo {r.loc[30,'h_c1_lo']-r.loc[3,'h_c1_lo']:5.2f}   "
            f"C1 hi {r.loc[30,'h_c1_hi']-r.loc[3,'h_c1_hi']:5.2f}")
    say()
    for _, r in lev.iterrows():
        rows.append(dict(analysis="depth_lever", group=f"S={r.S:.0f}",
                         n=3, statistic=f"h_{int(r.depth_m)}m",
                         point=r.h_point, lo95=r.h_c1_lo, hi95=r.h_c1_hi))

    # ------------------------------------------------------------------
    # (c) V0 vs C1=0, November-December and clamped scale
    # ------------------------------------------------------------------
    p0, _ = bm.fit_c1_zero(tr.cfdd.to_numpy(), tr.V0.to_numpy(), tr.total_cm.to_numpy())
    va["pred_c1zero"] = p0[0] * np.sqrt(np.maximum(0.0, va.cfdd.to_numpy() - p0[1]))

    say("=== (c) V0 versus C1 = 0 ===")
    say(f"C1 = 0 refit on training lakes: k={p0[0]:.4f} C0={p0[1]:.2f}")
    frac_above = float((va.total_cm > CLAMP).mean())
    say(f"{frac_above:.1%} of held-out observations exceed the {CLAMP} cm display clamp,")
    say("so the both-sides-clamped metric is compressed: for those rows the truth and")
    say("the prediction are both exactly the clamp. It measures agreement on the")
    say("displayed colour scale, not model accuracy. The prediction-only clamp is")
    say("what a viewer comparing the map to a measured thickness would see, and its")
    say("large positive bias is the clamp itself, not a model defect.")
    say()

    def block(label: str, sel: pd.Series, clamp_pred: bool, clamp_truth: bool) -> None:
        g = va[sel]
        if len(g) == 0:
            say(f"{label}: no observations")
            return
        y = g.total_cm.to_numpy()
        if clamp_truth:
            y = np.minimum(y, CLAMP)
        preds = {}
        for name, col in (("V0", "pred"), ("C1=0", "pred_c1zero")):
            pr = g[col].to_numpy()
            preds[name] = np.minimum(pr, CLAMP) if clamp_pred else pr
        m0 = fm.metrics(y, preds["V0"])
        m1 = fm.metrics(y, preds["C1=0"])
        say(f"{label}  n={len(g)}")
        say(f"  V0   RMSE {m0['rmse']:6.2f}  MAE {m0['mae']:6.2f}  bias {m0['bias']:+6.2f}")
        say(f"  C1=0 RMSE {m1['rmse']:6.2f}  MAE {m1['mae']:6.2f}  bias {m1['bias']:+6.2f}")
        say(f"  difference RMSE(V0) - RMSE(C1=0) = {m0['rmse']-m1['rmse']:+.2f} cm")
        rows.append(dict(analysis="v0_vs_c1zero", group=label, n=len(g),
                         statistic="rmse_v0", point=m0["rmse"],
                         lo95=np.nan, hi95=np.nan))
        rows.append(dict(analysis="v0_vs_c1zero", group=label, n=len(g),
                         statistic="rmse_c1zero", point=m1["rmse"],
                         lo95=np.nan, hi95=np.nan))

    block("all held-out, raw scale", pd.Series(True, index=va.index), False, False)
    say()
    block("January-March, raw scale", va.month.isin([1, 2, 3]), False, False)
    say()
    block("November-December, raw scale", va.month.isin([11, 12]), False, False)
    say()
    block(f"all held-out, prediction clamped at {CLAMP} cm, truth raw",
          pd.Series(True, index=va.index), True, False)
    say()
    block(f"all held-out, both sides clamped at {CLAMP} cm",
          pd.Series(True, index=va.index), True, True)
    say()

    # paired cluster interval for the November-December difference
    nd = va[va.month.isin([11, 12])]
    nd_lakes = sorted(nd.lake_key.unique())
    nd_by = {k: g for k, g in nd.groupby("lake_key")}
    diffs = []
    for _ in range(B):
        picks = rng.choice(nd_lakes, size=len(nd_lakes), replace=True)
        g = pd.concat([nd_by[k] for k in picks], ignore_index=True)
        y = g.total_cm.to_numpy()
        r0 = np.sqrt(np.mean((y - g.pred.to_numpy()) ** 2))
        r1 = np.sqrt(np.mean((y - g.pred_c1zero.to_numpy()) ** 2))
        diffs.append(r0 - r1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    say(f"November-December paired cluster difference: "
        f"{np.mean(diffs):+.2f} cm, 95% [{lo:+.2f}, {hi:+.2f}] "
        f"({len(nd_lakes)} lakes, {len(nd)} obs)")
    say()

    pd.DataFrame(rows).to_csv(DER / "skill_diagnostics.csv", index=False)
    (DIAG / "skill_diagnostics.txt").write_text("\n".join(lines) + "\n",
                                                encoding="utf-8")
    print(f"\nwrote {DER / 'skill_diagnostics.csv'}")
    print(f"wrote {DIAG / 'skill_diagnostics.txt'}")


if __name__ == "__main__":
    main()
