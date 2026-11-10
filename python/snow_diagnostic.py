"""Snow residual diagnostic by source and pooled, with lake-cluster intervals.

Uses the frozen V0 fit from `fit_results.csv` (no refit, no model change).
Residual is observed - predicted. For each source and for all sources pooled:
ordinary slope of residual on observed snow depth, its Gauss-Newton SE, and a
lake-cluster bootstrap interval (lakes resampled with replacement, 2,000
replicates, seed 20261108).

A missingness comparison shows how much of each source records snow and
whether observations with and without snow differ in thickness or residual.

Sources: G00803 (Great Lakes observers), NTL (Wisconsin lakes), ECCC (Little
Turkey). ECCC has one lake, so no cluster interval is reported for it.

Outputs
  data/derived/snow_diagnostic.csv
  data/derived/snow_missingness.csv
  data/derived/diagnostics/snow_diagnostic.txt

Run:  $env:PYTHONIOENCODING="utf-8"; python python/snow_diagnostic.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
DIAG = DER / "diagnostics"

B = 2000
SEED = 20261108


def predict(p: np.ndarray, s: np.ndarray, d: np.ndarray) -> np.ndarray:
    k, c0, c1 = p
    return k * np.sqrt(np.maximum(0.0, s - c0 - c1 * np.log(d)))


def slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Slope, Gauss-Newton SE, R^2 for y = a + b x."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(x) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = np.linalg.inv(X.T @ X) * s2
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(r2)


def main() -> None:
    m = pd.read_csv(DER / "training_matrix.csv", parse_dates=["date"])
    fit = pd.read_csv(DER / "fit_results.csv")
    v0 = fit[fit.variant == "V0"].iloc[0]
    p = np.array([v0.k, v0.C0, v0.C1])
    m["pred"] = predict(p, m.cfdd.to_numpy(), m.V0.to_numpy())
    m["resid"] = m.total_cm - m.pred

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    rng = np.random.default_rng(SEED)
    say("=== snow residual diagnostic, V0, residual = observed - predicted ===")
    say(f"fit from fit_results.csv: k={v0.k:.4f} C0={v0.C0:.1f} C1={v0.C1:.2f}")
    say(f"bootstrap: {B} replicates resampling lakes, seed {SEED}")
    say("slope units: cm of ice residual per cm of snow depth")
    say()

    rows = []
    for label, sel in [("g00803", m.source == "g00803"),
                       ("ntl", m.source == "ntl"),
                       ("eccc", m.source == "eccc"),
                       ("pooled", np.ones(len(m), bool))]:
        g = m[sel]
        s = g[g.snow_cm.notna()].copy()
        if len(s) < 5 or s.snow_cm.nunique() < 2:
            say(f"{label}: n={len(s)} - too few or constant")
            continue
        b, se, r2 = slope(s.snow_cm.to_numpy(), s.resid.to_numpy())
        lakes = sorted(s.lake_key.unique())
        by = {k: h for k, h in s.groupby("lake_key")}
        boot = []
        for _ in range(B):
            picks = rng.choice(lakes, size=len(lakes), replace=True)
            h = pd.concat([by[k] for k in picks], ignore_index=True)
            if h.snow_cm.nunique() < 2:
                continue
            try:
                boot.append(slope(h.snow_cm.to_numpy(), h.resid.to_numpy())[0])
            except np.linalg.LinAlgError:
                continue
        if len(lakes) > 1:
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            note = f"{len(lakes)} lakes"
        else:
            lo = hi = np.nan
            note = "one lake; no cluster interval"
        rows.append(dict(group=label, n_obs=len(s), n_lakes=len(lakes),
                         slope=b, gn_se=se, cluster_lo=lo, cluster_hi=hi,
                         r2=r2, note=note))
        say(f"{label:8s} n={len(s):4d} lakes={len(lakes):3d}  slope {b:+.3f} "
            f"(GN SE {se:.3f})  95% [{lo:+.3f}, {hi:+.3f}]  R2 {r2:.3f}  {note}")
    say()

    out = pd.DataFrame(rows)
    out.to_csv(DER / "snow_diagnostic.csv", index=False)

    # ---- missingness ------------------------------------------------------
    say("=== missingness comparison ===")
    overall = pd.DataFrame([dict(
        group="all", n=len(m), snow_n=int(m.snow_cm.notna().sum()),
        snow_pct=100 * m.snow_cm.notna().mean(),
        zero_n=int((m.snow_cm == 0).sum()),
        mean_total_snow=float(m.loc[m.snow_cm.notna(), "total_cm"].mean()),
        mean_total_nosnow=float(m.loc[m.snow_cm.isna(), "total_cm"].mean()),
        mean_resid_snow=float(m.loc[m.snow_cm.notna(), "resid"].mean()),
        mean_resid_nosnow=float(m.loc[m.snow_cm.isna(), "resid"].mean()),
    )])
    by_src = (m.groupby("source")
              .apply(lambda g: pd.Series(dict(
                  n=len(g), snow_n=int(g.snow_cm.notna().sum()),
                  snow_pct=100 * g.snow_cm.notna().mean(),
                  zero_n=int((g.snow_cm == 0).sum()),
                  mean_total_snow=float(g.loc[g.snow_cm.notna(), "total_cm"].mean()),
                  mean_total_nosnow=float(g.loc[g.snow_cm.isna(), "total_cm"].mean()),
                  mean_resid_snow=float(g.loc[g.snow_cm.notna(), "resid"].mean()),
                  mean_resid_nosnow=float(g.loc[g.snow_cm.isna(), "resid"].mean()),
              )), include_groups=False)
              .reset_index().rename(columns={"source": "group"}))
    miss = pd.concat([overall, by_src], ignore_index=True)
    miss.to_csv(DER / "snow_missingness.csv", index=False)
    say(miss.round(2).to_string(index=False))
    say()
    say("by month:")
    mon = m.groupby(m.date.dt.month).agg(
        n=("total_cm", "size"),
        snow_n=("snow_cm", "count"),
        snow_pct=("snow_cm", lambda s: 100 * s.notna().mean()),
        mean_total=("total_cm", "mean"),
    ).round(2)
    say(mon.to_string())
    say()
    say("note: G00803 snow depth is now read from columns [36:38]; blank is")
    say("NaN, an explicit 0 is kept as 0. The earlier negative pooled slope was")
    say("a composition effect: it came from NTL/ECCC only.")

    (DIAG / "snow_diagnostic.txt").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    print(f"\nwrote {DER / 'snow_diagnostic.csv'}")
    print(f"wrote {DER / 'snow_missingness.csv'}")
    print(f"wrote {DIAG / 'snow_diagnostic.txt'}")


if __name__ == "__main__":
    main()
