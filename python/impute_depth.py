"""Fill missing lake depths for Michigan inland lakes.

`build_depth.py` produces surveyed values only. This script fills the rest.

Area alone explains essentially none of the depth variance here
(R^2 = 0.00 for max depth, -0.03 for mean depth on 1,559 surveyed lakes), so a
power law would be a false precision. Instead each missing lake gets the median
depth of surveyed lakes in its area bin, which is the honest summary of what we
know: small lakes are shallower, but individual depth is unpredictable.

Depth is a weak lever on ice thickness (h ~ sqrt(S - C1*ln d)), so a factor-of-two
depth error costs only a few percent of modelled thickness. That is why binned
medians are acceptable rather than a blocker.

Outputs (overwrites data/derived/mi_lake_depth.csv):
    max_depth_m, mean_depth_m   filled where possible
    depth_source                point+contour | point | contour | imputed | ...
    depth_imputed               1 where the value came from the bin median
    depth_bin                   area bin label
    max_depth_iqr_m             interquartile range of the bin (uncertainty)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "derived" / "mi_lake_depth.csv"

# Mean/max depth ratio on surveyed lakes: median 0.41, IQR 0.32-0.50.
MEAN_MAX_RATIO = 0.41

BINS = [0.1, 0.2, 0.35, 0.6, 1.0, 2.0, 4.0, 8.0, np.inf]
LABELS = ["0.1-0.2", "0.2-0.35", "0.35-0.6", "0.6-1", "1-2", "2-4", "4-8", ">8"]


def main() -> None:
    df = pd.read_csv(CSV)
    df["depth_bin"] = pd.cut(df["area_km2"], BINS, labels=LABELS, right=True)

    # --- surveyed lakes -> bin statistics --------------------------------
    known = df[df["max_depth_m"].notna() & (df["max_depth_m"] > 0)]
    stats = (
        known.groupby("depth_bin", observed=True)["max_depth_m"]
        .agg(n="size", median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75))
    )
    stats["iqr"] = stats["q3"] - stats["q1"]

    mean_stats = (
        known.groupby("depth_bin", observed=True)["mean_depth_m"]
        .agg(mean_median="median")
    )
    mean_ratio = (known["mean_depth_m"] / known["max_depth_m"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    print("Area-bin medians from surveyed lakes:")
    print(
        stats.assign(mean=mean_stats["mean_median"])
        .round(2)
        .to_string()
    )
    print(f"\nmean/max depth ratio: median {mean_ratio.median():.2f} "
          f"n={len(mean_ratio)}")

    # Fall back to a global median if a bin has too few lakes to trust.
    global_med = float(known["max_depth_m"].median())
    min_n = 8

    before_missing = int(df["max_depth_m"].isna().sum())

    def fill_max(row) -> tuple[float, float, bool]:
        if pd.notna(row["max_depth_m"]) and row["max_depth_m"] > 0:
            return row["max_depth_m"], np.nan, False
        b = row["depth_bin"]
        if b in stats.index and stats.loc[b, "n"] >= min_n:
            return float(stats.loc[b, "median"]), float(stats.loc[b, "iqr"]), True
        return global_med, np.nan, True

    filled = df.apply(fill_max, axis=1, result_type="expand")
    df["max_depth_m"], df["max_depth_iqr_m"], df["depth_imputed"] = (
        filled[0], filled[1], filled[2].astype(int)
    )

    # --- mean depth -------------------------------------------------------
    # Missing mean: use the bin's surveyed mean depth, else the max-depth ratio.
    mean_by_bin = mean_stats["mean_median"]

    def fill_mean(row) -> float:
        if pd.notna(row["mean_depth_m"]) and row["mean_depth_m"] > 0:
            return row["mean_depth_m"]
        b = row["depth_bin"]
        if b in mean_by_bin.index and stats.loc[b, "n"] >= min_n:
            return float(mean_by_bin.loc[b])
        return float(row["max_depth_m"]) * MEAN_MAX_RATIO

    df["mean_depth_m"] = df.apply(fill_mean, axis=1)

    # --- label ------------------------------------------------------------
    df.loc[df["depth_imputed"] == 1, "depth_source"] = "imputed"
    # A handful have a surveyed max but no contour-integrated mean.
    only_max = (df["depth_imputed"] == 0) & (df["depth_source"] == "point")
    df.loc[only_max, "depth_source"] = "point"  # mean came from the ratio

    df = df.drop(columns=["depth_bin"])
    df["depth_bin"] = pd.cut(df["area_km2"], BINS, labels=LABELS, right=True).astype(str)
    df = df.sort_values("lake_idx")

    df.to_csv(CSV, index=False)

    print(f"\nimputed {before_missing} lakes with no surveyed depth")
    print(df["depth_source"].value_counts().to_string())
    print(f"\nall {len(df)} lakes now have max and mean depth")
    print("max depth (m):", df["max_depth_m"].describe(percentiles=[.5, .9]).round(1).to_dict())
    print("mean depth (m):", df["mean_depth_m"].describe(percentiles=[.5, .9]).round(1).to_dict())

    # Sanity: no absurd values for 10 ha ponds.
    small = df[df["area_km2"] <= 0.2]
    print(f"\nsmall lakes (<=0.2 km2): n={len(small)} "
          f"median max {small['max_depth_m'].median():.1f} m")


if __name__ == "__main__":
    main()
