"""Fit the three-constant ice model and validate on held-out lakes.

ARCHIVED 2026-11-09: this is the v1 growth-only model, kept as the frozen
comparison. Canonical v2 adds the melt term (arm A) and lives in
`python/fit_model_v2.py`; its parameters are in `model_parameters.csv`.
The v1 outputs here are also archived in `data/derived/backups/pre_v2_melt/`.

Model (v1, no snow):

    h = k * sqrt(max(0, S - C0 - C1 * ln(d)))

h is ice thickness in cm, S is accumulated freezing degree-days since 1 October
at the observation point, d is water depth in metres at the station. k sets the
growth scale, C0 is the offset before ice forms, C1 is the depth lever.

Fitting uses training lakes only. Held-out lakes are used exactly once, to
score the fitted model, and never to choose parameters. The variants differ
only in the depth convention (depth_variants.csv):

    V0   point depth
    V1   1.5 km median, four shallow stations only
    V2   drop the four shallow stations
    V3   1.5 km median, all Great Lakes stations
    V3a  3 km median, all Great Lakes stations
    V3b  max(point, 1.5 km median), all Great Lakes stations

Comparison protocol: V3/V3a/V3b against V0 on the full held-out set; V1/V2
against V0 on the common subset that excludes the held-out observations of the
three shallow validation stations (147, 306, 314).

Outputs
  data/derived/fit_results.csv
  data/derived/validation_predictions.csv
  data/derived/diagnostics/fit_report.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/fit_model.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

# (label, depth column, C0 lower bound). C0 >= 0 is a physical constraint:
# with no freezing degree-days there is no ice. The unconstrained fit puts C0
# near -111, which predicts ~15 cm on a 1.2 m lake at S = 0, i.e. ice on
# 1 October. V0u keeps the unconstrained parameters for comparison only.
FITS = [(v, v, 0.0) for v in ("V0", "V1", "V2", "V3", "V3a", "V3b")]
FITS.append(("V0u", "V0", -500.0))
VARIANTS = [f[0] for f in FITS]
SHALLOW_SIDS = {147, 306, 314}  # held-out shallow stations; 214 is training


def predict(h: np.ndarray, s: np.ndarray, d: np.ndarray) -> np.ndarray:
    k, c0, c1 = h
    return k * np.sqrt(np.maximum(0.0, s - c0 - c1 * np.log(d)))


def fit_one(s: np.ndarray, d: np.ndarray, y: np.ndarray,
            c0_floor: float = 0.0) -> tuple[np.ndarray, float]:
    """Multi-start least squares; returns parameters and training RMSE."""
    best = None
    for c0 in (max(0.0, c0_floor), 100.0, 200.0, 300.0):
        for c1 in (0.0, 5.0, 10.0, 20.0, 30.0):
            arg = np.maximum(0.0, s - c0 - c1 * np.log(d))
            scale = np.sqrt(arg.mean())
            k0 = float(np.clip(y.mean() / scale, 0.1, 10.0)) if scale > 0 else 1.0
            try:
                res = least_squares(
                    lambda p: predict(p, s, d) - y, [k0, c0, c1],
                    bounds=([0.01, c0_floor, -50.0], [20.0, 3000.0, 100.0]),
                    max_nfev=4000,
                )
            except Exception:  # noqa: BLE001
                continue
            rmse = float(np.sqrt(np.mean(res.fun ** 2)))
            if best is None or rmse < best[1]:
                best = (res.x, rmse)
    if best is None:
        raise RuntimeError("fit failed")
    return best


def param_se(res: np.ndarray, s: np.ndarray, d: np.ndarray, y: np.ndarray,
             p: np.ndarray) -> tuple[float, float, float]:
    """Standard error of k, C0, C1 from the Gauss-Newton covariance.

    Rows whose CFDD is below the threshold contribute h = 0 and have zero
    derivative with respect to every parameter; they must not enter the
    Jacobian as 1/sqrt(0).
    """
    k, c0, c1 = p
    arg = s - c0 - c1 * np.log(d)
    live = arg > 0
    root = np.sqrt(np.where(live, arg, 1.0))
    n, m = len(y), 3
    pred = k * np.sqrt(np.maximum(arg, 0.0))
    rss = float(np.sum((y - pred) ** 2))
    s2 = rss / max(n - m, 1)
    jac = np.column_stack([
        np.where(live, root, 0.0),                          # d/dk
        np.where(live, -k / (2 * root), 0.0),               # d/dC0
        np.where(live, -k * np.log(d) / (2 * root), 0.0),   # d/dC1
    ])
    jtj = jac.T @ jac
    try:
        cov = np.linalg.inv(jtj) * s2
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan
    se = np.sqrt(np.diag(cov))
    return float(se[0]), float(se[1]), float(se[2])


def ols2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope on x1 controlling for x2; returns slope, SE, R^2."""
    X = np.column_stack([np.ones(len(y)), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - 3, 1)
    s2 = float(resid @ resid) / dof
    cov = np.linalg.inv(X.T @ X) * s2
    se = float(np.sqrt(cov[1, 1]))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return float(beta[1]), se, r2


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    r = y - p
    ss_res = float(np.sum(r ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(r ** 2))),
                mae=float(np.mean(np.abs(r))), bias=float(np.mean(r)),
                r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, its standard error, and R^2 for y = a + b x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = np.linalg.inv(X.T @ X) * s2
    se = float(np.sqrt(cov[1, 1]))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return float(beta[1]), se, r2


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say(f"training matrix: {len(m):,} observations, {m.lake_key.nunique()} lakes")
    say(f"date range: {m.date.min().date()} to {m.date.max().date()}")
    say()

    results = []
    predictions = {}
    for var, depth_col, floor in FITS:
        sub = m[m[depth_col].notna()].copy()
        tr = sub[sub.split == "train"]
        va = sub[sub.split == "val"]
        s_tr, d_tr, y_tr = tr.cfdd.to_numpy(), tr[depth_col].to_numpy(), tr.total_cm.to_numpy()
        p, train_rmse = fit_one(s_tr, d_tr, y_tr, floor)
        se_k, se_c0, se_c1 = param_se(None, s_tr, d_tr, y_tr, p)

        # Held-out predictions under this variant.
        s_va, d_va, y_va = va.cfdd.to_numpy(), va[depth_col].to_numpy(), va.total_cm.to_numpy()
        pred_va = predict(p, s_va, d_va)
        wk = va.lake_key.str.replace("g00803:", "", regex=False)
        common = ~va.lake_key.isin([f"g00803:{s}" for s in SHALLOW_SIDS])
        # V2 has no validation rows for the dropped lakes anyway; keep the same
        # common mask for every variant so the comparison is like-for-like.
        common = ~wk.isin([str(s) for s in SHALLOW_SIDS])

        met_full = metrics(y_va, pred_va)
        met_common = metrics(y_va[common.to_numpy()], pred_va[common.to_numpy()])

        predictions[var] = pd.DataFrame({
            "lake_key": va.lake_key.to_numpy(), "date": va.date.to_numpy(),
            "region": va.region.to_numpy(), "split": "val",
            "total_cm": y_va, "pred_cm": np.round(pred_va, 2),
            "residual_cm": np.round(y_va - pred_va, 2),
            "depth_m": d_va, "cfdd": s_va,
            "common_subset": common.to_numpy(),
        })

        # Baselines from the training mean and median.
        base_mean = float(y_tr.mean())
        base_median = float(np.median(y_tr))
        bm = metrics(y_va, np.full_like(y_va, base_mean))
        bmd = metrics(y_va, np.full_like(y_va, base_median))

        results.append(dict(
            variant=var, depth_column=depth_col, c0_floor=floor,
            k=p[0], C0=p[1], C1=p[2],
            se_k=se_k, se_C0=se_c0, se_C1=se_c1,
            n_train=len(y_tr), n_val=len(y_va),
            train_rmse=train_rmse,
            val_rmse=met_full["rmse"], val_mae=met_full["mae"],
            val_bias=met_full["bias"], val_r2=met_full["r2"],
            common_n=met_common["n"], common_rmse=met_common["rmse"],
            common_mae=met_common["mae"], common_bias=met_common["bias"],
            baseline_mean_rmse=bm["rmse"], baseline_median_rmse=bmd["rmse"],
            base_mean=base_mean, base_median=base_median,
        ))
        say(f"{var}: k={p[0]:.3f} C0={p[1]:.1f} C1={p[2]:.1f}  "
            f"train RMSE {train_rmse:.1f} cm  val RMSE {met_full['rmse']:.1f} cm  "
            f"common {met_common['rmse']:.1f} cm")

    res = pd.DataFrame(results)
    res.to_csv(DER / "fit_results.csv", index=False)
    predictions["V0"].to_csv(DER / "validation_predictions.csv", index=False)

    say()
    say("=== variant comparison (held-out lakes only) ===")
    show = res[["variant", "k", "C0", "C1", "se_C1", "train_rmse",
                "val_rmse", "val_mae", "val_bias", "val_r2",
                "common_n", "common_rmse", "baseline_mean_rmse"]].copy()
    say(show.round(2).to_string(index=False))
    say()

    # Detailed report for V0 (the frozen baseline convention).
    v0 = res[res.variant == "V0"].iloc[0]
    pred = predictions["V0"]
    say("=== V0 fit ===")
    say(f"k  = {v0.k:.4f} +/- {v0.se_k:.4f}")
    say(f"C0 = {v0.C0:.1f} +/- {v0.se_C0:.1f} degC-day")
    say(f"C1 = {v0.C1:.2f} +/- {v0.se_C1:.2f} degC-day per ln(m)")
    say(f"C1 +/- 2 SE = [{v0.C1 - 2*v0.se_C1:.2f}, {v0.C1 + 2*v0.se_C1:.2f}]")
    say()
    y = pred.total_cm.to_numpy()
    p = pred.pred_cm.to_numpy()

    say("=== held-out error by region ===")
    rows = []
    for region, g in pred.groupby("region"):
        mm = metrics(g.total_cm.to_numpy(), g.pred_cm.to_numpy())
        base = metrics(g.total_cm.to_numpy(),
                       np.full(len(g), v0.base_mean))
        rows.append(dict(region=region, n=mm["n"], rmse=mm["rmse"], mae=mm["mae"],
                         bias=mm["bias"], r2=mm["r2"],
                         baseline_rmse=base["rmse"]))
    say(pd.DataFrame(rows).round(2).to_string(index=False))
    say()

    say("=== held-out error by month ===")
    pred["month"] = pd.to_datetime(pred.date).dt.month
    rows = []
    for month, g in pred.groupby("month"):
        mm = metrics(g.total_cm.to_numpy(), g.pred_cm.to_numpy())
        rows.append(dict(month=month, n=mm["n"], rmse=mm["rmse"], mae=mm["mae"],
                         bias=mm["bias"]))
    say(pd.DataFrame(rows).round(2).to_string(index=False))
    say()

    say("=== held-out error by depth band (V0) ===")
    pred["depth_band"] = pd.cut(pred.depth_m, [0, 3, 6, 10, 20, 40, 200],
                                labels=["0-3", "3-6", "6-10", "10-20", "20-40", "40+"])
    rows = []
    for band, g in pred.groupby("depth_band", observed=True):
        mm = metrics(g.total_cm.to_numpy(), g.pred_cm.to_numpy())
        rows.append(dict(depth_m=band, n=mm["n"], rmse=mm["rmse"], bias=mm["bias"]))
    say(pd.DataFrame(rows).round(2).to_string(index=False))
    say()

    say("=== depth-bias check (residual vs ln depth), held-out V0 ===")
    b, se, r2 = ols(np.log(pred.depth_m.to_numpy()), pred.residual_cm.to_numpy())
    say(f"slope {b:.1f} cm per ln(m), SE {se:.1f}, R2 {r2:.3f}, n {len(pred)}")
    month_num = pd.to_datetime(pred.date).dt.month.to_numpy().astype(float)
    b2, se2, r22 = ols2(np.log(pred.depth_m.to_numpy()), month_num,
                        pred.residual_cm.to_numpy())
    say(f"month-adjusted slope {b2:.1f} cm per ln(m), SE {se2:.1f}, R2 {r22:.3f}")
    say("no systematic depth bias is the gate; a slope whose 2-SE interval "
        "excludes zero is reported as a real residual depth effect")
    say()
    say("depth slope by region (V0, held-out):")
    for region, g in pred.groupby("region"):
        if g.depth_m.nunique() < 3:
            say(f"  {region:14s} not estimable: one station / constant depth")
            continue
        bb, ss, _ = ols(np.log(g.depth_m.to_numpy()), g.residual_cm.to_numpy())
        say(f"  {region:14s} n={len(g):4d}  slope {bb:+6.2f} +/- {ss:.2f}")
    say()
    say("sensitivity: drop the three flowing-water stations "
        "(127 Portage Waterway, 147 East Nipigon, 314 St. Marys River):")
    keep = ~pred.lake_key.isin(["g00803:127", "g00803:147", "g00803:314"])
    kp = pred[keep]
    bb, ss, _ = ols(np.log(kp.depth_m.to_numpy()), kp.residual_cm.to_numpy())
    say(f"  n={len(kp)}  depth slope {bb:+.2f} +/- {ss:.2f}  "
        f"RMSE {metrics(kp.total_cm.to_numpy(), kp.pred_cm.to_numpy())['rmse']:.2f}")
    say()
    say("implied onset threshold C0 + C1*ln(depth), degC-day:")
    for dd in (1.2, 3, 5, 10, 20, 50, 100):
        say(f"  depth {dd:5.1f} m -> {v0.C0 + v0.C1 * np.log(dd):7.1f}")
    say()
    say("depth lever at C1 +/- 2 SE (predicted thickness, cm, at median held-out CFDD):")
    med_s = float(np.median(pred.cfdd))
    for dd in (3, 10, 30):
        base = predict([v0.k, v0.C0, v0.C1], np.array([med_s]), np.array([dd]))[0]
        lo = predict([v0.k, v0.C0, v0.C1 - 2 * v0.se_C1], np.array([med_s]), np.array([dd]))[0]
        hi = predict([v0.k, v0.C0, v0.C1 + 2 * v0.se_C1], np.array([med_s]), np.array([dd]))[0]
        say(f"  depth {dd:2.0f} m: {base:5.1f} cm  (C1-2SE {lo:5.1f}, C1+2SE {hi:5.1f}, "
            f"spread {abs(hi - lo):4.1f})")
    say()

    say("=== error by lake type ===")
    inland = pred.region.isin(["inland_mi", "inland_ntl", "inland_eccc"])
    for label, sel in [("Michigan inland", pred.region == "inland_mi"),
                       ("NTL + ECCC inland", pred.region.isin(["inland_ntl", "inland_eccc"])),
                       ("Great Lakes", ~inland)]:
        mm = metrics(pred.total_cm[sel].to_numpy(), pred.pred_cm[sel].to_numpy())
        say(f"{label:18s} n={mm['n']:4d}  RMSE {mm['rmse']:5.1f}  "
            f"MAE {mm['mae']:5.1f}  bias {mm['bias']:+5.1f}  R2 {mm['r2']:.3f}")
    say()

    say("=== snow residual diagnostic (V0, observations that record snow) ===")
    tr = m[(m.split == "train") & m.snow_cm.notna()].copy()
    va = m[(m.split == "val") & m.snow_cm.notna()].copy()
    for label, g in [("train", tr), ("held-out", va)]:
        if len(g) < 10:
            say(f"{label}: n={len(g)} - too few")
            continue
        g = g.copy()
        g["pred"] = predict([v0.k, v0.C0, v0.C1], g.cfdd.to_numpy(), g.V0.to_numpy())
        g["resid"] = g.total_cm - g.pred
        b, se, r2 = ols(g.snow_cm.to_numpy(), g.resid.to_numpy())
        say(f"{label:9s} n={len(g):4d}  slope {b:+.2f} cm per cm snow, "
            f"SE {se:.2f}, R2 {r2:.3f}")
    say()
    say("slope < 0 means more snow -> thinner ice than the model, i.e. the "
        "insulation the v1 model deliberately omits")
    say()

    say("=== five largest held-out residuals (V0) ===")
    top = pred.reindex(pred.residual_cm.abs().sort_values(ascending=False).index).head(5)
    say(top[["lake_key", "date", "region", "total_cm", "pred_cm", "residual_cm",
             "depth_m", "cfdd"]].to_string(index=False))

    (OUT / "fit_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'fit_report.txt'}")


if __name__ == "__main__":
    main()
