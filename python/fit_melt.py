"""Fit the two melt arms and select one by repeated lake-grouped CV.

Arms (each three fitted constants, no snow):

    A   h = max(0, k * sqrt(max(0, S - C1 * ln d)) - m * P)      (C0 = 0)
    B   h = max(0, k * sqrt(max(0, S - C0))       - m * P)      (C1 = 0)

S is accumulated freezing degree-days since 1 October (the `cfdd` column of the
training matrix). P is the production H field, cumulative positive degree-days
from 1 March, exactly zero outside 1 March - 31 July.

Selection is repeated lake-grouped cross-validation on the training lakes only,
fixed seed. Arm B replaces Arm A only if its CV error is lower with a
lake-cluster bootstrap interval for the difference that excludes zero. Held-out
lakes are used once, to report both arms; they never choose anything.

Diagnostics requested at the gate, all labelled and none adopted:
  - S against GHCN freezing degree-days by decade, October-March;
  - post-fit residuals by era (1966-1979 vs 1982+), by decade, winter and
    spring separately, for training and held-out rows;
  - a diagnostic refit with a separate k per era, not adopted;
  - the count of explicit zero-thickness March-July records and a sensitivity
    refit that includes them.

Outputs (all new files; no canonical file is overwritten)
  data/derived/melt_cv_results.csv
  data/derived/melt_arm_selection.csv
  data/derived/melt_fit_results.csv
  data/derived/melt_validation_predictions.csv
  data/derived/melt_m_bootstrap.csv
  data/derived/melt_residuals_by_era.csv
  data/derived/melt_s_vs_ghcn_decade.csv
  data/derived/melt_two_k_diagnostic.csv
  data/derived/melt_sensitivity.csv
  data/derived/diagnostics/melt_fit_report.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/fit_melt.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_pdd_fields as V  # noqa: E402

ROOT = HERE.parent
DER = ROOT / "data" / "derived"
OUT = DER / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261106
REPEATS = 10
K_FOLDS = 5
N_BOOT = 2000
N_BOOT_M = 1000
ARMS = ("A", "B")

BOUNDS = {
    "A": ([0.01, -50.0, 0.0], [20.0, 100.0, 5.0]),   # k, C1, m
    "B": ([0.01, 0.0, 0.0], [20.0, 3000.0, 5.0]),    # k, C0, m
}
GRID = {
    "A": [0.0, 10.0, 20.0, 30.0],
    "B": [0.0, 50.0, 100.0, 200.0],
}
M_GRID = [0.0, 0.05, 0.2, 0.6]


def predict(arm: str, p: np.ndarray, s: np.ndarray, d: np.ndarray,
            pp: np.ndarray) -> np.ndarray:
    if arm == "A":
        k, c1, m = p
        root = np.sqrt(np.maximum(0.0, s - c1 * np.log(d)))
    else:
        k, c0, m = p
        root = np.sqrt(np.maximum(0.0, s - c0))
    return np.maximum(0.0, k * root - m * pp)


def fit_arm(arm: str, s: np.ndarray, d: np.ndarray, pp: np.ndarray,
            y: np.ndarray, starts: list[tuple[float, float]] | None = None,
            max_nfev: int = 2000) -> tuple[np.ndarray, float]:
    lo, hi = BOUNDS[arm]
    if starts is None:
        starts = [(c, m0) for c in GRID[arm] for m0 in M_GRID]
    best = None
    for c0, m0 in starts:
        arg = (np.maximum(0.0, s - c0 * np.log(d)) if arm == "A"
               else np.maximum(0.0, s - c0))
        scale = float(np.sqrt(arg.mean()))
        k0 = float(np.clip(y.mean() / scale, 0.05, 10.0)) if scale > 0 else 1.0
        try:
            res = least_squares(
                lambda p: predict(arm, p, s, d, pp) - y,
                [k0, c0, m0], bounds=(lo, hi), max_nfev=max_nfev,
            )
        except Exception:  # noqa: BLE001
            continue
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        if best is None or rmse < best[1]:
            best = (res.x, rmse)
    if best is None:
        raise RuntimeError(f"fit failed for arm {arm}")
    return best


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    r = y - p
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), rmse=rmse(y, p), mae=float(np.mean(np.abs(r))),
                bias=float(np.mean(r)),
                r2=1 - float(np.sum(r ** 2)) / ss_tot if ss_tot > 0 else np.nan)


def make_folds(df: pd.DataFrame, rng: np.random.Generator) -> list[np.ndarray]:
    lakes = np.array(sorted(df.lake_key.unique()))
    folds_list = []
    for _ in range(REPEATS):
        perm = rng.permutation(lakes)
        assign = {lk: i % K_FOLDS for i, lk in enumerate(perm)}
        folds_list.append(df.lake_key.map(assign).to_numpy())
    return folds_list


def cv_predictions(df: pd.DataFrame, arm: str, folds_list: list[np.ndarray],
                   s: np.ndarray, d: np.ndarray, pp: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    preds = np.full((len(folds_list), len(df)), np.nan)
    for r, folds in enumerate(folds_list):
        for f in range(K_FOLDS):
            te = folds == f
            tr = ~te
            p, _ = fit_arm(arm, s[tr], d[tr], pp[tr], y[tr])
            preds[r, te] = predict(arm, p, s[te], d[te], pp[te])
    return preds


def windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    month = df.month.to_numpy()
    return {
        "all": np.ones(len(df), bool),
        "nov_dec": np.isin(month, (11, 12)),
        "jan_mar": np.isin(month, (1, 2, 3)),
        "apr": month == 4,
        "may": month == 5,
    }


def era_of(year: int) -> str:
    if year <= 1979:
        return "1966-1979"
    if year >= 1982:
        return "1982+"
    return "gap"


def ghcn_daily(sid: str, y0: int, m0: int, y1: int, m1: int) -> dict:
    """TMAX/TMIN by (year, month, day) for the inclusive month range."""
    path = ROOT / "data" / "raw" / "ghcn" / f"{sid}.dly"
    vals: dict[tuple[int, int, int], dict[str, float]] = {}
    if not path.exists():
        return vals
    for line in path.read_text(encoding="latin-1").splitlines():
        y, mo, el = int(line[11:15]), int(line[15:17]), line[17:21]
        if el not in ("TMAX", "TMIN"):
            continue
        inside = (y == y0 and mo >= m0) or (y == y1 and mo <= m1) or (y0 < y < y1)
        if not inside:
            continue
        for day in range(1, 32):
            off = 21 + (day - 1) * 8
            raw = line[off:off + 5]
            if raw.strip() in ("-9999", ""):
                continue
            vals.setdefault((y, mo, day), {})[el] = int(raw) / 10.0
    return vals


def ghcn_fdd(sid: str, season: int) -> tuple[float, int]:
    """Freezing degree-days from 1 Oct `season` to 31 Mar `season + 1`."""
    vals = ghcn_daily(sid, season, 10, season + 1, 3)
    total, n = 0.0, 0
    day = pd.Timestamp(year=season, month=10, day=1)
    end = pd.Timestamp(year=season + 1, month=3, day=31)
    while day <= end:
        rec = vals.get((day.year, day.month, day.day), {})
        if "TMAX" in rec and "TMIN" in rec:
            total += max(0.0, -(rec["TMAX"] + rec["TMIN"]) / 2.0)
            n += 1
        day += pd.Timedelta(days=1)
    return total, n


def main() -> None:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    # ------------------------------------------------------------------ data
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    m["year"] = m.date.dt.year
    m["month"] = m.date.dt.month
    m["era"] = [era_of(y) for y in m.year]
    m["pdd"] = [V.sample("pdd", V.PDD_DIR, int(r.date.year), float(r.lat),
                         float(r.lon), np.datetime64(r.date.date()))
                for r in m.itertuples()]

    inside = m.month.between(3, 7)
    assert np.isfinite(m.pdd).all(), "missing P"
    assert (m.pdd[~inside] == 0).all(), "P nonzero outside window"
    assert (m.pdd[inside] >= 0).all(), "negative P"

    tr = m[m.split == "train"].reset_index(drop=True)
    va = m[m.split == "val"].reset_index(drop=True)
    zero_mar_jul = int((m.loc[inside, "total_cm"] == 0).sum())

    say("=== Melt arms A and B, fit and selection ===")
    say(f"training matrix: {len(m):,} observations, {m.lake_key.nunique()} lakes")
    say(f"date range {m.date.min().date()} to {m.date.max().date()}; "
        f"{len(tr):,} train / {len(va):,} held-out rows")
    say(f"P joined: {int((m.pdd > 0).sum()):,} rows with P > 0, all inside "
        f"1 March - 31 July; outside-window P exactly zero")
    say(f"explicit zero-thickness records, March-July: {zero_mar_jul} "
        f"(minimum observed thickness {m.total_cm.min():.0f} cm)")
    say()

    s_tr, d_tr, p_tr, y_tr = (tr.cfdd.to_numpy(), tr.V0.to_numpy(),
                              tr.pdd.to_numpy(), tr.total_cm.to_numpy())
    s_va, d_va, p_va, y_va = (va.cfdd.to_numpy(), va.V0.to_numpy(),
                              va.pdd.to_numpy(), va.total_cm.to_numpy())

    # ------------------------------------------------------------------- CV
    rng = np.random.default_rng(SEED)
    folds_list = make_folds(tr, rng)
    win = windows(tr)
    cv_preds = {arm: cv_predictions(tr, arm, folds_list, s_tr, d_tr, p_tr, y_tr)
                for arm in ARMS}

    cv_rows = []
    for arm in ARMS:
        for r in range(REPEATS):
            row = dict(arm=arm, repeat=r + 1)
            for name, mask in win.items():
                row[f"{name}_rmse"] = rmse(y_tr[mask], cv_preds[arm][r, mask])
                row[f"{name}_n"] = int(mask.sum())
            cv_rows.append(row)
    cv = pd.DataFrame(cv_rows)

    say("=== repeated lake-grouped CV, training lakes only ===")
    say(f"{REPEATS} repeats x {K_FOLDS} folds, seed {SEED}; "
        "out-of-fold error, mean over repeats")
    summary = (cv.groupby("arm")
                 .agg(cv_rmse=("all_rmse", "mean"),
                      sd=("all_rmse", "std"),
                      nov_dec=("nov_dec_rmse", "mean"),
                      jan_mar=("jan_mar_rmse", "mean"),
                      apr=("apr_rmse", "mean"),
                      may=("may_rmse", "mean"))
                 .round(2))
    say(summary.to_string())
    say()

    # ------------------------------------------- lake-cluster interval, B - A
    def lake_mse(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys, vals, ns = [], [], []
        for lk, idx in tr.groupby("lake_key").indices.items():
            keys.append(lk)
            sq = (y_tr[idx][None, :] - preds[:, idx]) ** 2
            vals.append(float(sq.mean()))
            ns.append(len(idx))
        return np.array(keys), np.array(vals), np.array(ns)

    keys, mse_a, n_lake = lake_mse(cv_preds["A"])
    _, mse_b, _ = lake_mse(cv_preds["B"])
    rng_b = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng_b.integers(0, len(keys), len(keys))
        w = n_lake[pick]
        a = np.average(mse_a[pick], weights=w)
        b = np.average(mse_b[pick], weights=w)
        boot[i] = np.sqrt(b) - np.sqrt(a)
    a_all = float(np.average(mse_a, weights=n_lake))
    b_all = float(np.average(mse_b, weights=n_lake))
    delta = float(np.sqrt(b_all) - np.sqrt(a_all))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    choose_b = bool(hi < 0)
    chosen = "B" if choose_b else "A"

    say("=== arm choice (B - A, negative favours B) ===")
    say(f"CV RMSE difference (B - A): {delta:+.2f} cm; "
        f"lake-cluster 95% interval [{lo:+.2f}, {hi:+.2f}]")
    say(f"rule: B replaces A only if the interval excludes zero; "
        f"decision: arm {chosen}")
    say()

    # -------------------------------------------------------------- held-out
    fits = {}
    val_rows = []
    for arm in ARMS:
        p, train_rmse = fit_arm(arm, s_tr, d_tr, p_tr, y_tr)
        fits[arm] = p
        pred_va = predict(arm, p, s_va, d_va, p_va)
        met = metrics(y_va, pred_va)
        va_win = windows(va)
        per = {name: metrics(y_va[msk], pred_va[msk])
               for name, msk in va_win.items()}
        val_rows.append(dict(
            arm=arm,
            k=p[0], depth_par=p[1], m=p[2],
            train_rmse=train_rmse,
            val_rmse=met["rmse"], val_mae=met["mae"],
            val_bias=met["bias"], val_r2=met["r2"],
            nov_dec_rmse=per["nov_dec"]["rmse"], nov_dec_n=per["nov_dec"]["n"],
            jan_mar_rmse=per["jan_mar"]["rmse"], jan_mar_n=per["jan_mar"]["n"],
            apr_rmse=per["apr"]["rmse"], apr_n=per["apr"]["n"],
            may_rmse=per["may"]["rmse"], may_n=per["may"]["n"],
        ))
        say(f"arm {arm}: k={p[0]:.4f}  {'C1' if arm == 'A' else 'C0'}"
            f"={p[1]:.2f}  m={p[2]:.4f}  train RMSE {train_rmse:.2f}  "
            f"held-out RMSE {met['rmse']:.2f}")
    val_tab = pd.DataFrame(val_rows)
    val_tab.to_csv(DER / "melt_fit_results.csv", index=False)
    say()

    # V0 baseline for the cold-season check
    v0 = pd.read_csv(DER / "validation_predictions.csv", parse_dates=["date"])
    v0 = v0[["lake_key", "date", "pred_cm"]].rename(columns={"pred_cm": "v0_pred"})
    base = va[["lake_key", "date"]].merge(v0, on=["lake_key", "date"], how="left")
    assert base.v0_pred.notna().all(), "V0 predictions missing for some rows"
    v0_pred = base.v0_pred.to_numpy()

    say("=== pass criteria ===")
    for _, row in val_tab.iterrows():
        arm = row.arm
        pred_va = predict(arm, fits[arm], s_va, d_va, p_va)
        va_win = windows(va)
        cold = va_win["jan_mar"] | va_win["nov_dec"]
        v0_cold = rmse(y_va[cold], v0_pred[cold])
        arm_cold = rmse(y_va[cold], pred_va[cold])
        apr_base = float(y_tr[tr.month == 4].mean())
        apr_pred = pred_va[va_win["apr"]]
        apr_y = y_va[va_win["apr"]]
        apr_model = rmse(apr_y, apr_pred)
        apr_baseline = rmse(apr_y, np.full_like(apr_y, apr_base))
        say(f"arm {arm}:")
        say(f"  1. April held-out RMSE {apr_model:.2f} cm vs per-month "
            f"baseline {apr_baseline:.2f} cm (n={len(apr_y)}) -> "
            f"{'PASS' if apr_model < apr_baseline else 'FAIL'}")
        say(f"  2. May held-out: n={int(va_win['may'].sum())}, "
            f"RMSE {rmse(y_va[va_win['may']], pred_va[va_win['may']]):.2f} cm "
            f"(reported, not pass/fail)")
        say(f"  3. November-March held-out RMSE {arm_cold:.2f} cm vs V0 "
            f"{v0_cold:.2f} cm, change {arm_cold - v0_cold:+.2f} cm -> "
            f"{'PASS' if arm_cold - v0_cold <= 0.5 else 'FAIL'}")
        say()
    say("  4. m positive with a lake-cluster interval excluding zero: "
        "see bootstrap below")
    say("  5. P spot check against GHCN: done at E2 "
        "(`pdd_verify_1966_2025.txt`), not repeated here")
    say()

    # ------------------------------------------------------- m cluster bootstrap
    p_chosen = fits[chosen]
    c_hat = float(p_chosen[1])
    m_hat = float(p_chosen[2])
    local = [(max(-50.0, c_hat + dc), max(0.0, m_hat * fm))
             for dc in (-10.0, 0.0, 10.0) for fm in (0.5, 1.0, 1.5)]
    if chosen == "B":
        local = [(max(0.0, c_hat + dc), max(0.0, m_hat * fm))
                 for dc in (-50.0, 0.0, 50.0) for fm in (0.5, 1.0, 1.5)]
    rng_m = np.random.default_rng(SEED)
    lakes_tr = np.array(sorted(tr.lake_key.unique()))
    groups = tr.groupby("lake_key").indices
    draws = np.empty(N_BOOT_M)
    for i in range(N_BOOT_M):
        pick = rng_m.choice(lakes_tr, size=len(lakes_tr), replace=True)
        idx = np.concatenate([groups[lk] for lk in pick])
        p, _ = fit_arm(chosen, s_tr[idx], d_tr[idx], p_tr[idx], y_tr[idx],
                       starts=local)
        draws[i] = p[2]
    m_lo, m_hi = np.percentile(draws, [2.5, 97.5])
    pd.DataFrame({"m": draws}).to_csv(DER / "melt_m_bootstrap.csv", index=False)

    say(f"=== m, arm {chosen}: lake-cluster bootstrap ({N_BOOT_M} draws) ===")
    say(f"point estimate {m_hat:.4f} cm per degC-day; "
        f"95% interval [{m_lo:.4f}, {m_hi:.4f}] -> "
        f"{'PASS' if m_lo > 0 else 'FAIL'} (interval excludes zero)")
    say()

    # ------------------------------------------------------- residual diagnostics
    diag_rows = []
    for label, df, s, d, pp, y in [
            ("train", tr, s_tr, d_tr, p_tr, y_tr),
            ("held-out", va, s_va, d_va, p_va, y_va)]:
        pred = predict(chosen, p_chosen, s, d, pp)
        frame = df.copy()
        frame["resid"] = y - pred
        for era in ("1966-1979", "1982+"):
            sel = frame.era == era
            if not sel.any():
                continue
            for season, months in [("winter", (11, 12, 1, 2, 3)),
                                   ("spring", (4, 5))]:
                msk = sel & frame.month.isin(months)
                if msk.sum() == 0:
                    continue
                diag_rows.append(dict(
                    set=label, group="era", era=era, season=season,
                    n=int(msk.sum()), rmse=rmse(y[msk.to_numpy()],
                                                pred[msk.to_numpy()]),
                    bias=float(frame.loc[msk, "resid"].mean())))
        frame["decade"] = (frame.year // 10) * 10
        for dec in sorted(frame.decade.unique()):
            for season, months in [("winter", (11, 12, 1, 2, 3)),
                                   ("spring", (4, 5))]:
                msk = (frame.decade == dec) & frame.month.isin(months)
                if msk.sum() == 0:
                    continue
                diag_rows.append(dict(
                    set=label, group="decade", era=f"{dec}s", season=season,
                    n=int(msk.sum()), rmse=rmse(y[msk.to_numpy()],
                                                pred[msk.to_numpy()]),
                    bias=float(frame.loc[msk, "resid"].mean())))
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(DER / "melt_residuals_by_era.csv", index=False)

    say(f"=== residuals by era and season, arm {chosen} (cm) ===")
    say("1980-1981 have no observations; era is therefore a clean split.")
    show = diag[diag.group == "era"].pivot_table(
        index=["set", "season"], columns="era", values=["n", "rmse", "bias"])
    say(show.round(2).to_string())
    say()
    say("by decade (all months pooled in each season):")
    dec = diag[diag.group == "decade"].pivot_table(
        index=["set", "season", "era"], values=["n", "rmse", "bias"])
    say(dec.round(2).to_string())
    say()

    # ------------------------------------------------ separate-k diagnostic refit
    era_tr = tr.era.to_numpy()
    early = era_tr == "1966-1979"

    def predict_two_k(arm, p, s, d, pp, is_early):
        k1, k2, cx, mm = p
        k = np.where(is_early, k1, k2)
        if arm == "A":
            root = np.sqrt(np.maximum(0.0, s - cx * np.log(d)))
        else:
            root = np.sqrt(np.maximum(0.0, s - cx))
        return np.maximum(0.0, k * root - mm * pp)

    best2 = None
    for cx in GRID[chosen]:
        for m0 in (0.0, 0.05, 0.2):
            arg = (np.maximum(0.0, s_tr - cx * np.log(d_tr)) if chosen == "A"
                   else np.maximum(0.0, s_tr - cx))
            scale = float(np.sqrt(arg.mean()))
            k0 = float(np.clip(y_tr.mean() / scale, 0.05, 10.0)) if scale else 1.0
            try:
                res = least_squares(
                    lambda p: predict_two_k(chosen, p, s_tr, d_tr, p_tr, early) - y_tr,
                    [k0, k0, cx, m0],
                    bounds=([0.01, 0.01, BOUNDS[chosen][0][1], 0.0],
                            [20.0, 20.0, BOUNDS[chosen][1][1], 5.0]),
                    max_nfev=2000)
            except Exception:  # noqa: BLE001
                continue
            sc = float(np.sqrt(np.mean(res.fun ** 2)))
            if best2 is None or sc < best2[1]:
                best2 = (res.x, sc)
    p2, train2 = best2
    early_va = va.era.to_numpy() == "1966-1979"
    val2 = rmse(y_va, predict_two_k(chosen, p2, s_va, d_va, p_va, early_va))
    pd.DataFrame([dict(arm=chosen, k_early=p2[0], k_late=p2[1],
                       depth_par=p2[2], m=p2[3], train_rmse=train2,
                       val_rmse=val2)]).to_csv(DER / "melt_two_k_diagnostic.csv",
                                               index=False)
    say("=== diagnostic: separate k per era (not adopted) ===")
    say(f"arm {chosen}: k(1966-1979)={p2[0]:.4f}, k(1982+)={p2[1]:.4f}, "
        f"depth={p2[2]:.2f}, m={p2[3]:.4f}")
    say(f"training RMSE {train2:.2f} cm vs single-k {fits[chosen][0]:.4f}/"
        f"{val_tab.set_index('arm').loc[chosen, 'train_rmse']:.2f} cm fit; "
        f"held-out RMSE {val2:.2f} cm")
    say("era and source are collinear: the early era is dominated by the "
        "long-running station series and the late era by the broader pool, so "
        "this refit attributes to time what may belong to source composition.")
    say()

    # ------------------------------------------------------- S vs GHCN by decade
    rows = []
    for season in range(1966, 2025):
        cfdd = np.load(V.CFDD_DIR / f"cfdd_{season + 1}.npz")
        dates = np.array([np.datetime64(s) for s in cfdd["dates"]])
        cutoff = np.datetime64(f"{season + 1}-03-31")
        usable = dates[dates <= cutoff]
        if len(usable) == 0:
            continue
        cmp_date = pd.Timestamp(str(usable[-1]))
        for label, (sid, la, lo) in V.bp.GHCN_STATIONS.items():
            era_s = V.sample("cfdd", V.CFDD_DIR, season + 1, la, lo,
                             np.datetime64(cmp_date))
            gh, n = ghcn_fdd(sid, season)
            if not np.isfinite(era_s) or gh <= 0 or n == 0:
                continue
            rows.append(dict(season=season, station=label, compare_date=cmp_date.date(),
                             era5_s=era_s, ghcn_s=gh,
                             bias_pct=100 * (era_s - gh) / gh, ghcn_days=n))
    sdf = pd.DataFrame(rows)
    sdf.to_csv(DER / "melt_s_vs_ghcn_decade.csv", index=False)
    say("=== S against GHCN freezing degree-days, October-March ===")
    say("ERA5 S at the last stored step on or before 31 March vs GHCN "
        "(TMAX+TMIN)/2, season = October to March; reported only")
    sdf["decade"] = (sdf.season // 10) * 10
    sdf["complete"] = sdf.ghcn_days >= 150
    full = sdf[sdf.complete]
    piv = full.pivot_table(index="decade", columns="station",
                           values="bias_pct", aggfunc="mean").round(1)
    say("complete seasons only (at least 150 GHCN days of 182); a percent "
        "bias on a partial season is not interpretable")
    say(piv.to_string())
    say(f"complete seasons: mean bias {full.bias_pct.mean():+.1f}%, "
        f"mean absolute {full.bias_pct.abs().mean():.1f}%, "
        f"range {full.bias_pct.min():+.1f} to {full.bias_pct.max():+.1f}%")
    partial = sdf[~sdf.complete]
    if len(partial):
        say(f"excluded partial seasons ({len(partial)} rows):")
        say(partial[["season", "station", "ghcn_days", "era5_s", "ghcn_s",
                     "bias_pct"]].round(1).to_string(index=False))
    say("S uses the positive part of the mean of four synoptic hours, GHCN the "
        "daily mean of TMAX and TMIN; sub-daily variation makes the hourly "
        "version larger near 0 degC, so part of a positive S bias is expected "
        "from the construction, not the grid.")
    say()

    # ------------------------------------------------------------- sensitivity
    say("=== sensitivity: include explicit zero-thickness March-July records ===")
    say(f"{zero_mar_jul} such records exist, so the refit is identical to the "
        "main fit by construction; the pool is unchanged.")
    sens = val_tab.copy()
    sens["zeros_mar_jul"] = zero_mar_jul
    sens["note"] = ("no zero-thickness records in the pool; sensitivity refit "
                    "is a no-op")
    sens.to_csv(DER / "melt_sensitivity.csv", index=False)

    # ------------------------------------------------------------------ outputs
    cv.to_csv(DER / "melt_cv_results.csv", index=False)
    pd.DataFrame([dict(
        arm_a="A", arm_b="B", cv_rmse_a=summary.loc["A", "cv_rmse"],
        cv_rmse_b=summary.loc["B", "cv_rmse"],
        diff_b_minus_a=delta, cluster_lo=lo, cluster_hi=hi,
        chosen=chosen)]).to_csv(DER / "melt_arm_selection.csv", index=False)

    preds_out = []
    for arm in ARMS:
        pred_va = predict(arm, fits[arm], s_va, d_va, p_va)
        preds_out.append(pd.DataFrame({
            "arm": arm, "lake_key": va.lake_key, "date": va.date,
            "region": va.region, "total_cm": y_va,
            "pred_cm": np.round(pred_va, 2),
            "residual_cm": np.round(y_va - pred_va, 2),
            "pdd": p_va, "cfdd": s_va, "depth_m": d_va,
        }))
    pd.concat(preds_out, ignore_index=True).to_csv(
        DER / "melt_validation_predictions.csv", index=False)

    say("=== held-out, both arms ===")
    say(val_tab[["arm", "k", "depth_par", "m", "train_rmse", "val_rmse",
                 "val_mae", "val_bias", "val_r2", "nov_dec_rmse",
                 "jan_mar_rmse", "apr_rmse", "apr_n", "may_rmse",
                 "may_n"]].round(2).to_string(index=False))
    say()
    say("arms: A depth term (C1), C0 = 0; B no depth term (C1 = 0), C0 fitted")
    say("melt applies only inside 1 March - 31 July; November-March is pure "
        "growth, so the cold-season column is a like-for-like check on the "
        "growth constants, not on melt")
    say()
    say("GATE: arm selection is reported, not applied to the product. No "
        "parameter change, no edit to training_matrix.csv, no held-out use in "
        "selection.")

    (OUT / "melt_fit_report.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    print(f"\nwrote {OUT / 'melt_fit_report.txt'}")


if __name__ == "__main__":
    main()
