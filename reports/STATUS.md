# STATUS — lake ice thickness map

**Last updated:** 2026-11-10 (web viewer shipped; default opening view fixed)
**Read this first.** Single source of truth for project state.
`reports/data-audit.md` holds the detailed source inventory.
`reports/artifacts.md` holds the cross-source artifact audit and the
pre-fit cleanup decisions.

---

## 1. What this is

Animated historical/current lake ice-thickness map for:

- **Lake Superior, Michigan, Huron**
- **All Michigan inland lakes ≥10 ha**

Only ice thickness is displayed, plus city and lake names. Dark basemap,
navy→white ramp, white above 8 inches. Plays **October 1950 → spring 2025** at
3-day steps.

It is **model guidance, not a measurement**, and must not be presented as a
safety tool.

---

## 2. Model

```
h = k * sqrt(max(0, S(x,t) - C0 - C1 * ln(d(x))))
```

| Symbol | Meaning |
|---|---|
| `h` | ice thickness (cm) |
| `S` | cumulative freezing degree-days since 1 October (°C·day) |
| `d` | water depth at the pixel (m) |
| `k, C0, C1` | three fitted constants |

Fitted on point observations. Ice thickness is computed **per pixel in a shader**
from a static depth raster plus a small CFDD field — never stored as a raster stack.

**Snow excluded from v1.** Residual diagnostic against observed snow after fitting.

### Depth is a weak lever (measured)

At S = 600 °C·day with the fitted C1 = 26.5, going from 1.5 m to 6 m mean depth
moves ice thickness 1.2 cm on ~38 cm (3.2%). Even the top of the cluster
interval (C1 = 51.6) gives only ~4.3 cm across 3 m → 30 m. This is why the
56% of Michigan lakes without a survey can be imputed without harming the map.

**Fitted (V0, C0 ≥ 0): k = 1.6049, C1 = 26.52, C0 = 0.** Held-out RMSE
12.16 cm on 1,110 observations; **no established skill over a per-month
climatology** (12.60 cm; paired 95% interval −1.94 to +0.65 cm). The
depth-bias gate fails on the full held-out set. Details in
`reports/model-v1.md`.

---

## 3. Decisions log

| Date | Decision |
|---|---|
| 2026-11-05 | Project lives in `Map_Art\ice`; old `currents` project untouched |
| 2026-11-05 | Temperature: ERA5-Land (9 km) via **raw hourly dataset**, 4 synoptic hours/day → daily mean (daily-statistics dataset is broken) |
| 2026-11-05 | Snow excluded from v1 |
| 2026-11-05 | Validation gate: **no hard numeric threshold.** No depth bias, report held-out error by lake type and season, beat a constant-thickness baseline |
| 2026-11-05 | Depth at a station = **point depth at that station**, not lake mean |
| 2026-11-05 | Display clamp 0–8 in (20.3 cm); 1 m depth quantisation; uint8 depth PNGs; uint16 CFDD; 64×64 CFDD grid at 3-day steps |
| 2026-11-05 | 1950–65 flagged as unvalidated extrapolation (observations start 1966) |
| 2026-11-05 | Michigan DNR `MaxDepth` and contour `MinDepth` are in **feet** → ×0.3048 |
| 2026-11-06 | Michigan depth table: 1,559 surveyed + 1,989 imputed by area-bin median |
| 2026-11-06 | Great Lakes bathymetry: NOAA NCEI 3 arc-second GeoTIFF, depth = −elevation |
| 2026-11-06 | **80/20 split by lake, stratified by region.** Michigan inland lakes split 50/50 because they are the target population |
| 2026-11-06 | Stations/lakes without a clean depth are **dropped**, not patched |

---

## 4. Data on disk

| Path | Contents | Counts |
|---|---|---|
| `data/raw/g00803/extract/data2.dat` | Station ice thickness, cm | 7,547 rows, 130 stations, 1966–79 |
| `data/raw/g00803/extract/ice.loc` | Station names + lat/lon | 121 of 130 |
| `data/raw/eccc/ECCC_LI-TC_v6.csv` | ECCC Lake Ice DB (header on row 58) | 2,870 rows, 16 sites |
| `data/raw/eccc/External_LI-TC_v2.csv` | ECCC external DB (header on row 21) | 4,784 rows, 17 sites |
| `data/raw/ntl/ntl34_v10.csv` | NTL snow & ice depth, cm | 1,015 rows, 11 lakes, 1982–2026 |
| `data/raw/dnr/lakes_10ha.geojson` | DNR lake polygons ≥10 ha | 3,548 |
| `data/raw/dnr/deep_points.geojson` | DNR surveyed deep points, ft | 2,651 rows → 2,175 lakes |
| `data/raw/dnr/contours.geojson` | DNR depth contours, `MinDepth` ft | 25,492 lines, 2,166 lakes |
| `data/raw/bathy/{superior,michigan,huron}.tif` | NOAA 3" bathymetry, EPSG:4269 | 165 / 90 / 97 MB |
| `data/derived/mi_lake_depth.csv` | Per-lake area, max/mean depth, source | 3,548 rows |
| `data/derived/training_stations.csv` | G00803 station region classification | 121 rows |
| `data/derived/observations.csv` | **All normalised thickness measurements** | 4,785 lake-dates |
| `data/derived/training_lakes.csv` | **One row per lake, with frozen split** | 74 lakes |
| `data/derived/split_frozen.csv` | Frozen lake → split, never recomputed | 89 lake keys |
| `data/derived/station_lake_match.csv` | Inland station → DNR polygon audit | 22 stations |
| `data/derived/era5/cfdd_YYYY.npz` | CFDD, 3-day steps, 81×171 uint16 | one per year |
| `data/derived/era5/carry_YYYY.npy` | CFDD at 31 Dec (seed for next year) | one per year |

### Michigan inland lakes — size distribution

3,548 polygons, **3,018 km²** total, median 0.24 km².

| Size (km²) | Lakes | % of lakes | Area (km²) | % of area | Surveyed |
|---|---|---|---|---|---|
| 0.1–0.2 | 1,451 | 40.9% | 202 | 6.7% | 31% |
| 0.2–0.3 | 623 | 17.6% | 152 | 5.0% | 42% |
| 0.3–0.5 | 567 | 16.0% | 217 | 7.2% | 51% |
| 0.5–1 | 464 | 13.1% | 327 | 10.8% | 59% |
| 1–2 | 224 | 6.3% | 310 | 10.3% | 63% |
| 2–5 | 147 | 4.1% | 439 | 14.5% | 63% |
| 5–10 | 35 | 1.0% | 255 | 8.4% | 71% |
| 10–50 | 31 | 0.9% | 709 | 23.5% | 71% |
| >50 | 6 | 0.2% | 408 | 13.5% | 100% |

Surveyed bathymetry covers 1,559 lakes but **67% of total lake area**.

---

## 5. Observation table and split

`python/build_observations.py` produces `observations.csv` and
`training_lakes.csv`. It normalises every source to **cm**, attaches a lake
depth, and applies the split frozen on 2026-11-06 (`split_frozen.csv`, seed
**20261106**).

**The pool below is after the 2026-11-07 cleanup:** 11 harbour stations (H1) and
4 wrong-lake stations (M1 plus the Warners Lake name check) are excluded, and
inland depth is joined on the DNR polygon id (H2). Pre-cleanup copies are
`*_2026-11-07_pre-cleanup.csv`.

### Sources normalised

All sources are converted to **cm**. Multi-point surveys are collapsed to one
value per lake-date (see below).

| Source | Raw rows | Lake-dates kept | Units in | Conversion |
|---|---|---|---|---|
| G00803 | 7,547 | 3,754 | cm | none |
| NTL | 1,015 | 1,009 | cm | none |
| ECCC | 385 | 22 | m | ×100 |
| **Total** | | **4,785** | | |

### One row per lake-date

ECCC is a multi-point survey: Little Turkey Lake has 189 probes across only 15
dates, Crystal Lake 196 probes across 7 dates. A dozen holes on the same
afternoon describe within-lake variability, not independent evidence about a
lake-scale model, so they are averaged to one value per lake-date. The spread is
retained as `sd_cm` (median 4.1 cm, max 17.7 cm) which is a realistic
measurement-error floor of roughly 10% of typical thickness.

G00803 and NTL were already one row per lake-date.

Crystal Lake appears in both NTL (127 dates) and ECCC (7 dates) with no overlap;
it is treated as one lake (`ntl:CR`).

### Lakes in the training pool: 74

| Region | Train lakes | Train obs | Val lakes | Val obs |
|---|---|---|---|---|
| Lake Superior | 29 | 1,773 | 3 | 446 |
| Lake Michigan | 6 | 388 | 1 | 96 |
| Lake Huron | 5 | 393 | 1 | 97 |
| **Michigan inland** | **10** | **281** | **7** | **280** |
| NTL Wisconsin | 8 | 821 | 3 | 195 |
| Little Turkey ON | 1 | 15 | 0 | 0 |
| **Total** | **59** | **3,671** | **15** | **1,114** |

19 January–April 2026 observations (15 train, 4 held-out) have no ERA5 file yet
and are excluded from the fitting matrix, leaving **3,656 train / 1,110
held-out**. The table above is the full observation pool before that exclusion.

**Validation share: 23.3% of observations, 15 of 74 lakes.** Michigan inland
lakes are split 50/50; everything else 20% per region. Whole-lake granularity
pushes the overall figure a little above 20%. Date range 1966-01-18 to
2026-04-01. No lake appears on both sides.

### Split rules

- Split is **by lake**, never by observation — no lake is on both sides.
- **Frozen on 2026-11-06** in `data/derived/split_frozen.csv` (seed 20261106).
  `build_observations.py` reads it and errors on any unknown lake; it never
  re-allocates. Re-running the original greedy allocator after the cleanup
  reassigns three surviving Michigan inland lakes (Bodi 116, Independence 154,
  Chain 305), so recomputation is forbidden.
- Michigan inland 50/50 by lake; every other multi-lake region 20%. Single-lake
  regions (Little Turkey) are all training.

### Dropped, not patched

43 of 141 candidate lakes have no clean depth and are dropped:

| Reason | Lakes |
|---|---|
| Ontario inland stations — no authoritative morphometry in hand | 16 |
| Minnesota inland stations — same | 7 |
| Wisconsin inland (Europe Lake) — same | 1 |
| Erie / Ontario / St Clair stations — out of scope | 17 |
| No coordinates in `ice.loc` (121 of 130 stations only) | 9 |
| Michigan inland: Lake Antoine, no DNR polygon match | 1 |
| Unclassified: Long Lake, Iron County | 1 |

Separately, **15 lakes are excluded by policy, not by depth**: the 11 harbour
stations (H1, 295 obs, shoreline-edge depth artifact) and the 4 stations whose
name and coordinate disagree (M1 + name audit, 117 obs — Pine, Gibson, Big Island
and Warners Lake). Both sets are listed in `reports/artifacts.md` §2–§4 and
`data/derived/diagnostics/station_name_check.csv`.

This leaves 4,785 lake-date observations: 3,754 G00803, 1,009 NTL, 22 ECCC.

G00803 fixed-width offsets, verified by hand: station `[0:3]`, year `[4:6]`,
month `[6:8]`, day `[8:10]`, flag `[10]`, total `[11:14]`, black `[14:17]`,
white `[17:20]`. Total = black + white on every row.

The Fortran layout (`display_ice2.f`, format 100) also puts **snow depth at
`[36:38]`**, water level at `[33:35]`, snow condition at `[39:42]` and ice
condition at `[44:47]`. Snow depth is populated on 3,758 of 7,547 records and
agrees with the English file (`data1.dat`, same columns, inches × 2.54) to a
median 0.3 cm. **It is now parsed** as `snow_cm` (blank → NaN, explicit 0 → 0):
2,545 of 3,754 G00803 observations carry snow. Diagnostic only; see
`reports/model-v1.md` §8.

### Depth attachment

| Region | Depth source |
|---|---|
| Michigan inland | DNR deep point / contour integration (`mi_lake_depth.csv`), joined on `lake_idx` |
| NTL | published NTL morphometry |
| Little Turkey | ECCC bathymetry page (13 m) |
| Great Lakes | NOAA raster sampled at station, nearest water pixel if on shore |

Depth range across the pool: median 9.1 m, 90th percentile 27.0 m, max 109.3 m
(deep Lake Superior station). The H2 join corrected four surviving inland lakes
(Bear 10.36→1.22, Indian 7.62→19.96, Crystal 16.76→53.34, Twin 7.62→12.19 m;
123 observations).

### Corrected outlier

Earlier notes recorded a Batchawana Bay 1968-03-02 value of **201 cm**. The raw
record is `1040680302 120102 18` — total **120 cm**, black 102, white 18. The
201 was a misread. 120 cm is still the largest value in the pool (99th percentile
is 84 cm) and remains the one flagged outlier.

---

## 6. Scripts

Run with `$env:PYTHONIOENCODING="utf-8"` in PowerShell.

| Script | Does | Output |
|---|---|---|
| `fetch_dnr.py` | Lakes ≥10 ha, deep points, contour key list | `data/raw/dnr/*` |
| `fetch_dnr_contours.py` | Contour geometry + per-lake index | `contours.geojson`, `contour_index.csv` |
| `fetch_era5.py` | Raw ERA5-Land 4 h/day → daily mean → CFDD | `data/derived/era5/cfdd_YYYY.npz` |
| `fetch_greatlakes_bathy.py` | NOAA 3" GeoTIFF per lake | `data/raw/bathy/*.tif` |
| `build_depth.py` | DNR depth table + hypsographic mean depth | `mi_lake_depth.csv` |
| `impute_depth.py` | Area-bin medians for unsampled lakes | `mi_lake_depth.csv` (filled) |
| `build_observations.py` | Normalise all sources, attach depth, freeze split | `observations.csv`, `training_lakes.csv` |
| `audit_artifacts.py` | Re-run the H1/H2/M1/L4 cross-source checks | console tables |
| `check_station_names.py` | Name-agreement audit of the spatial station match | `diagnostics/station_name_check.csv`, `diagnostics/inland_depth_provenance.csv` |
| `shallow_bay_check.py` | Point vs neighbourhood depth for every GL station; depth variants | `diagnostics/gl_depth_neighbourhoods.csv`, `depth_variants.csv` |
| `sample_cfdd.py` | Bilinear/time-interpolated CFDD at every observation; GHCN 1952 check | `training_matrix.csv`, `diagnostics/cfdd_missing_2026.csv` |
| `fit_model.py` | Fit the three constants, validate, run variants and diagnostics | `fit_results.csv`, `validation_predictions.csv`, `diagnostics/fit_report.txt` |
| `bootstrap_model.py` | Lake-cluster bootstrap, C1 = 0 fit, per-month baseline, post-hoc subset | `bootstrap_intervals.csv`, `diagnostics/bootstrap_report.txt` |
| `build_pdd.py` | 1966 P pilot: unpack CDS ZIP → daily mean → P on the CFDD grid; sample obs; GHCN Mar–May check | `pdd/pdd_1966.npz`, `diagnostics/pdd_pilot_1966.txt` |
| `fetch_pdd_trial.py` | One hourly t2m request, 1 Mar–31 Jul 1966; wall time and size | `raw/era5/pdd_trial_1966_hourly.zip`, `diagnostics/pdd_trial_1966_timing.*` |
| `verify_snow_column.py` | Prove the G00803 snow column changed nothing else | `diagnostics/snow_column_verification.txt` |
| `snow_diagnostic.py` | Snow residual by source and pooled, cluster intervals, missingness | `snow_diagnostic.csv`, `snow_missingness.csv` |
| `skill_diagnostics.py` | Anomaly skill, low-CFDD depth lever, V0 vs C1 = 0 on Nov-Dec and clamp | `skill_diagnostics.csv`, `diagnostics/skill_diagnostics.txt` |
| `compare_pdd_definitions.py` | Hourly trial -> H, P4, M4 vs GHCN 1966; pre-written selection rule | `pdd_trial/*.npz`, `pdd_definition_comparison_1966.csv` |
| `pull_pdd.py` | Resumable sequential hourly pull, 1966-1970 then 1971-2025 then 1950-1965; retries, status, no silent P=0 | `raw/era5/pdd_YYYY_hourly.zip`, `diagnostics/pdd_pull_*` |
| `apply_pdd_rule.py` | Build H, P4, M4 for 1966-1970 and apply the pre-written GHCN rule | `pdd_definition_rule_*.csv`, `diagnostics/pdd_rule_1966_1970.txt` |

`fetch_era5.py` is resumable: existing `cfdd_YYYY.npz` are skipped and
`carry_YYYY.npy` makes an interrupted run safe to restart.

`audit_artifacts.py` is the reproduction for `reports/artifacts.md`; it reads
the derived tables and the three NOAA GeoTIFFs and prints the same tables.

---

## 7. Temperature route

CDS `derived-era5-land-daily-statistics` is unusable — requests stay `accepted`
forever. Raw `reanalysis-era5-land` processes in ~30 s.

- 4 synoptic hours (00/06/12/18 UTC), basin `[49.5, -93, 41.5, -76]`, 0.1° grid.
- Daily mean = mean of those four hours.
- CFDD resets 1 October, 3-day steps, uint16.
- ~5 min and 37 MB per year; raw zip deleted after processing.
- **Submit sequentially.** CDS anti-abuse policy (22 Oct 2026) can disable the
  account for parallel requests.
- **Complete: 1950–2025**, 76 files, log clean at 2025-12-31.
### CFDD validation vs GHCN stations (Oct–Dec 1952)

Reproducible from `sample_cfdd.py` (GHCN `.dly` cached in `data/raw/ghcn`).
FDD = sum of max(0, −(TMAX+TMIN)/2) over the 92 days, same definition as the
ERA5 daily mean.

| Station | GHCN ID | GHCN FDD | ERA5 FDD | Diff |
|---|---|---|---:|---:|---:|
| Duluth MN | USW00014913 | 288 | 216 | −25% |
| Marquette MI | USW00014838 | 98 | 87 | −11% |
| Ann Arbor MI | USC00200230 | 50 | 47 | −6% |

The earlier ad-hoc table (Marquette 131, Ann Arbor 78) was not tied to named
station IDs and does not reproduce; the values above supersede it.

Point station vs 9 km grid cell, plus lake moderation and elevation. The **mean**
bias is absorbed by the fitted `C0`; the station-to-station spread is a real error
source and must be reported in validation.

---

## 8. Current state

| Component | State |
|---|---|
| Source audit | ✅ `reports/data-audit.md` |
| DNR polygons / points / contours | ✅ |
| Contour → polygon matching | ✅ |
| Michigan depth table | ✅ 3,548 / 3,548 |
| Great Lakes bathymetry (NOAA 3") | ✅ 3 GeoTIFFs |
| Station classification | ✅ `training_stations.csv` (+ `lake_idx`) |
| Observation table | ✅ 4,785 lake-dates, 74 lakes |
| Train/validation split | ✅ frozen in `split_frozen.csv`, 0 lakes shared, 0 drift |
| Cross-source artifact audit | ✅ `reports/artifacts.md` — 9.5% of obs carried a depth/identity problem |
| Pre-fit cleanup (H2 join, H1 + name-check drops) | ✅ applied 2026-11-07, Gate B verified |
| ERA5 CFDD | ✅ 1950–2025 complete, 76 files |
| CFDD lookup + training matrix | ✅ 4,766 obs joined; 19 2026 obs flagged |
| Model fit + validation | ✅ `reports/model-v1.md` — held-out RMSE 12.16 cm; no established skill over per-month climatology; depth-bias gate fails |
| Cluster bootstrap + corrections | ✅ `bootstrap_model.py`; `bootstrap_intervals.csv`, `diagnostics/bootstrap_report.txt` |
| G00803 snow column | ✅ parsed as `snow_cm`; all other columns, row order, split and RMSE unchanged |
| Snow diagnostic by source | ✅ `snow_diagnostic.py`; pooled interval includes zero |
| P pilot 1966 | ✅ `pdd/pdd_1966.npz`; GHCN Mar–May check done |
| P timing trial | DONE 668 s, 94.9 MB; 76-year linear estimate 14.1 h, 7.2 GB |
| P definition rule 1966-1970 | DONE: H selected, mean abs bias 19.47 vs P4 21.09, M4 24.16; production P awaits gate |
| Melt term | ✅ v2 canonical: arm A kept, `k = 1.6865`, `C1 = 34.38`, `m = 0.3318` |
| Production P 1950–2025 | ✅ 76/76 `data/derived/pdd/pdd_YYYY.npz`, E2 verified |
| Shoreline mask gate | ✅ HydroLAKES + ESI St Marys + DNR-wins; 33,015,374 px; `build_shoreline_mask.py` |
| Per-pixel depth rasters | ✅ 3″ depth + mask, 24 GL / 14 inland tiles; `verify_rasters.py` PASS |
| Per-season S/P web fields | ✅ 75 seasons 1950/51–2024/25, 89.3 MB; `build_web_fields.py` |
| Web layers (6″ cov + depth) | ✅ `gl_cov/dep`, `inland_cov/dep`, 2.3 MB total; `build_web_layers.py` |
| Viewer + local server | ✅ `web/` WebGL2 app, `serve_web.py`, shader validated vs Python |

---

## 9. Next steps, in order

1. ~~Finish the ERA5 pull~~ — done, 1950–2025.
2. ~~Apply the artifact-audit fixes~~ — done 2026-11-07 (Gate B), verified:
   4,785 obs, 74 lakes, 15 val lakes, 0 split drift.
3. ~~Write the CFDD sampler and join~~ — done 2026-11-08. 4,766 obs,
   19 January–April 2026 obs flagged (`diagnostics/cfdd_missing_2026.csv`).
4. ~~Fit the three constants and validate~~ — done 2026-11-08.
   `reports/model-v1.md`: held-out RMSE 12.16 cm vs 16.90 baseline, R² 0.48.
5. ~~Gate D decisions~~ — done 2026-11-08: `C0 >= 0`, depth variant V0, and
   the three flowing-water stations reported separately (not deleted).
6. **Melt term** (design approved; hourly pull approved and running):
   - pull `t2m` hourly, 1 Mar-31 Jul, one CDS request per year, sequential;
     order 1966-1970, then 1971-2025, then 1950-1965; resumable,
     disk checked first, every raw file kept;
   - retry failed years; never silently `P = 0`; list unrecoverable years;
   - when 1966-1970 are complete, apply the pre-written rule to H, P4
     and M4 (mean absolute station bias, then spread; three stations;
     temperature and GHCN only), report, then stop at a gate before
     deriving the production P for all years. Do not pause the pull.
   - build the 3-day `P` texture (uint16, same grid) from 1 March, zero
     outside 1 Mar–31 Jul;
   - fit two arms on training lakes only: A (`k`, `C1`, `m`), B (`k`, `C0`,
     `m`); choose by repeated lake-grouped CV, fixed seed; switch to B only if
     its CV error beats A with a lake-cluster interval excluding zero;
     include November-December error as a reported column in the CV output;
   - report both arms on held-out for reporting only;
   - pass criteria: **April** held-out RMSE beats the per-month baseline (n
     reported); May reported but not pass/fail; Nov–Mar worsens ≤0.5 cm; `m`
     positive with cluster interval excluding zero; `P` checked against GHCN
     Mar–May at Duluth, Marquette, Ann Arbor.
7. ~~**Per-pixel depth rasters**~~ — done 2026-11-10. 3″ log-depth + lake-code
   mask, contour interpolation for surveyed lakes and bowls elsewhere, DNR
   wins over the Great Lakes mask, mask matches the gate exactly.
8. ~~**Per-pixel CFDD and P fields for the animation**~~ — done 2026-11-10.
   75 season files of S/P on the 0.1° grid, 1 Oct → 30 Jun.
9. ~~**Pack web assets** and build the app~~ — done 2026-11-10. 6″ coverage +
   depth textures, WebGL2 viewer, local server, shader validated against a
   Python reference.
10. **Visual QA and hand-over** (in progress): melt-frame ramp, inland zoom,
    Georgian Bay islands, labels — checked. Remaining: confirm `web/README.md`
    and the report sections stay current, and give the user the serving
    command.

---

## 10. Open questions for the user

**Model — remaining questions (detail in `reports/model-v1.md`):**

- **Authorize the `P` pull?** The 1966 pilot is complete and clean. On
  approval: 1 Mar–31 Jul, 1966–2025 then 1950–1965, sequential, resumable,
  disk checked first, failed years retried and listed. No silent `P = 0`.
- **V0 vs C1 = 0?** The depth term is kept on physical grounds, but C1 = 0 is
  marginally better on held-out data (11.89 vs 12.16 cm; paired interval
  +0.03 to +0.51) and has a smaller depth-bias slope. The melt step resolves
  this by fitting both arms and selecting on training-lake CV only.
- **P definition.** The 1966 hourly trial is done: 668 s, 94.9 MB; 76-year
  linear estimate 14.1 h, 7.2 GB. Mean absolute station bias is H 16.7,
  P4 18.7, M4 23.5; H's gain is Duluth only. The rule, written before the
  comparison: over 1966–1970, pick the definition with the smallest mean
  absolute station bias; tie-break on the smallest spread across stations. No
  ice observations or held-out data enter the choice. Gate: hourly pull (H
  available) or 4-sample pull storing P4 and M4.
- Melt limitations to state when fitted: sparse nearshore spring observations,
  `m` rests on ~54 seasons and few lakes, May has no training observations,
  no ice-off validation for the big lakes, no wind-driven breakup. ERA5 P is
  18–34% below GHCN at Marquette and Ann Arbor under every definition
  (32–38% at Marquette), which would bias `m` upward; Duluth is unbiased under H.

**Map and product:**

- Great Lakes shoreline polygons for clipping: Natural Earth 10 m, or derive from
  the NOAA bathymetry mask?
- Basemap style and tile provider.
- City label set, and how many lake labels at each zoom.
- Whether the 1950–65 extrapolation is visually distinguished or captioned.
- Whether to keep or drop the single 120 cm Batchawana Bay outlier in the fit
  (currently kept).

---

## 11. Environment

- Python 3.11.0, Windows, project at `C:\Users\evane\OneDrive\Desktop\CREATE\Map_Art\ice`
- pandas 2.3.0, numpy 2.2.6, geopandas 1.1.2, shapely 2.1.2, rasterio 1.4.4,
  pyproj 3.7.2, netCDF4 1.7.5.1, requests 2.32.4, fiona 1.10.1, cdsapi, fitz
- `C:\Users\evane\.cdsapirc` present and working
- PowerShell/Python console needs `$env:PYTHONIOENCODING="utf-8"`
- `con.py` is a reserved Windows device name

---

## 12. Progress log

Newest last. Kept so the project can be resumed from this file alone.

**2026-11-05**
- Source audit written (`reports/data-audit.md`).
- DNR layers downloaded: 3,548 lakes ≥10 ha, 2,651 deep points, 25,492 contours.
- Contour→polygon spatial matching built (1,562 lakes covered).
- ERA5 raw pipeline fixed: ERA5-Land starts 01:00 UTC on 1950-01-01, so the first
  day has three hours, not four. Daily mean now groups by calendar date.
- CFDD verified on 1950–51: resets to 0 on 1 Oct, carries 703 °C·day into 1951.

**2026-11-06 morning**
- Contour validation: `MinDepth` is the contour depth in feet; max contour depth
  matches the DNR deep point exactly on 1,541 lakes.
- Hypsographic bug fixed: zero-area fragment rings were collapsing mean depth.
  Filter rings below 0.5% of lake area before integration.
- `mi_lake_depth.csv` built: 1,559 surveyed lakes, plausible means (Houghton 2.7 m,
  Burt 8.0 m, Mullett 10.6 m, Higgins 16.4 m).
- Area→depth regression tested: R² = 0.00 (max), −0.03 (mean). Power law rejected.

**2026-11-06 midday**
- 1,989 unsampled lakes imputed with area-bin medians, flagged `depth_imputed`.
- NOAA NCEI 3 arc-second GeoTIFF downloaded for all three lakes. First extraction
  grabbed a sidecar file; fixed to pick the largest `.tif` member.
- Bathymetry sanity check: max depths 406 / 281 / 229 m, matching published values.
- Station classification: 121 located stations → 45 on Great Lakes water,
  22 Michigan inland lakes with 710 observations. This was the key find — the
  observation pool for inland lakes is real, not a proxy.
- CFDD spot check vs GHCN: Duluth −26%, Marquette +0.2%, Ann Arbor −40%.
  Point-vs-grid effect; mean absorbed by `C0`.

**2026-11-06 afternoon**
- `build_observations.py`: 7,547 G00803 + 1,015 NTL + 385 ECCC rows normalised to cm.
- ECCC found to be multi-point surveys (12–16 probes/date); collapsed to one value
  per lake-date, retaining `sd_cm` as a measurement-error term.
- ECCC Crystal Lake merged into NTL `CR` (same waterbody).
- 52 of 141 candidate lakes dropped for missing depth or coordinates.
- Final pool: 89 lakes, 5,197 lake-dates, 71/18 by lake (80/20) and **23.9% by observation**.
- Outlier corrected: Batchawana Bay 1968-03-02 is 120 cm, not the 201 cm previously
  recorded.

**2026-11-06 evening — cross-source artifact audit**
- Wrote `reports/artifacts.md`. Thickness values are consistent across sources;
  the artifacts are in the **depth covariate** and **three lake-identity matches**.
- **H1 harbours:** 267 of 295 harbour obs (91%) sit on a depth 2–100× shallower than
  the 3–4 km neighbourhood median. Cause: shore coordinate → nearest water pixel is
  the shallowest edge pixel. Recommended: drop the 11 harbour stations.
- **H2 duplicate names:** `drop_duplicates("name")` picks an arbitrary polygon.
  On 5 inland stations the depth is wrong by 0.9–36.6 m (Crystal 16.8 vs 53.3,
  Indian 7.6 vs 20.0, Bear 10.4 vs 1.2). 140 obs. Fix: join on `lake_idx`.
- **M1 wrong-lake:** Pine→Second, Gibson→Lake Ellen, Big Island→Triangle. 74 obs.
  Recommended: drop.
- **M2 coarse coordinates:** 22 stations / 1,198 obs (23.1%) at ≤2 dp; 755 at 1 dp
  (~11 km). Six GL stations sample depth 5–20× too shallow. Recommended: drop
  <3-dp GL stations or use uncertainty-scaled depth. **Withdrawn 2026-11-07 —
  see the entry below.**
- Cleared: Two Harbors trio (distinct probes), Fanny Hooe/Copper Harbor (different
  lakes, shared town geocode), Crystal Lake merge (correct), digit preference
  (resolution not error), zero/trace values (0.1%).
- No change to `mi_lake_depth.csv` or the map rasters: those are per polygon.

**2026-11-06 late — M2 correction and drop scenarios — WITHDRAWN 2026-11-07**
- **M2 was measured wrong.** Precision must be the coarser of latitude and
  longitude, not latitude alone (Duluth Harbor: 15-dp lat, 1-dp lon). Correct
  figure is **2,148 obs (41.3%)** at ≤2 dp and **1,472 obs (28.3%)** at 1 dp,
  not 1,198 / 755. Report corrected in `artifacts.md` §5, §11, §13.
- Consequence: "drop the 1-dp GL stations" removes **1,108 obs**, not ~620, and
  **leaves Lake Huron with zero validation observations** (DeTour is 1-dp).
  Recommendation flipped to **keep** the 1-dp stations and report the uncertainty.
- Drop scenarios priced (`artifacts.md` §13): current 5,197/89; drop harbours
  4,902/78; **harbours + wrong-lake 4,828/75, 24.0% val (recommended)**;
  + 1-dp GL 3,720/61 with Huron val = 0; + 2 bad-depth 4,689/73.
- ERA5 running: 1960 done, 1961 downloading. Fit-ready when 1966 lands (~30 min
  from 12:26).

**2026-11-07 — M2 withdrawn; shallow-bay question isolated**
- **M2 is withdrawn entirely.** The 2026-11-06 entries above counted decimal
  digits after converting degrees and minutes to decimal degrees. All 121
  `ice.loc` records are whole arc-minutes, so a minute that is a multiple of 6
  becomes one decimal place (06′ = 0.1°) and a multiple of 3 becomes two.
  Verified: 121/121 parse, no seconds, 242 minute values across 44 distinct
  values, 40 of them multiples of 6. The digit count measured the conversion,
  not the source.
- **Real precision is uniform at ±0.5′** (~0.9 km N–S, ~0.65 km E–W at 46 °N),
  under a sixth of an ERA5-Land cell. The 1,472 / 2,148 counts and the
  3,720-observation scenario are void. `artifacts.md` §5 now carries the
  retraction and no recommendation.
- The four shallow Great Lakes stations (East Nipigon 147, Little Traverse 214,
  Mackinaw City 306, DeTour 314) are **not** a coordinate problem: all four
  coordinates fall on water and are sampled at the exact pixel (0-pixel snap).
  They are a depth-convention question, tested as variants V1–V3.
- ERA5 finished: 1950–2025, 76 files, log clean at 2025-12-31. The fit is no
  longer blocked.

**2026-11-07 — Gate B: scenario B applied and verified**
- Backups: `observations_2026-11-07_pre-cleanup.csv`,
  `training_lakes_..._pre-cleanup.csv`, `training_stations_..._pre-cleanup.csv`.
- New `data/derived/split_frozen.csv` (89 lake keys). `build_observations.py`
  now requires it and errors on unknown lakes; the greedy allocator is removed
  because it drifts three surviving Michigan inland lakes.
- H2 fixed: inland depth joins on `lake_idx`, matched spatially (inside, else
  nearest within 5 km). `station_lake_match.csv` records the match. Four
  surviving lakes corrected, 123 obs: Bear 10.36->1.22, Indian 7.62->19.96,
  Crystal 16.76->53.34, Twin 7.62->12.19 m. No other station moved >0.5 m.
- H1 (11 harbour sids, 295 obs) and M1 (sids 122/204/210, 74 obs) dropped; the
  g00803 station join is now inner, so policy drops no longer masquerade as
  "no depth".
- Verified: **4,828 obs, 75 lakes, 3,671 train / 1,157 val (23.96%), 16 val
  lakes, 0 split drift, 0 lakes on both sides.** Michigan inland 604 obs /
  18 lakes (10 train, 8 val). Removed 369 obs (282 train, 87 val). Row-level
  identity of the 4,828 survivors exact except 123 depth values.
- Validation coverage: Huron 97 obs / 1 lake, Michigan 96 / 1, Superior 446 / 3,
  inland_mi 323 / 8, inland_ntl 195 / 3. `inland_eccc` is a single-lake region
  and cannot have a held-out lake.

**2026-11-07 - Gate B follow-up: name audit and depth provenance**
- New `python/check_station_names.py` (exits 0) writes
  `data/derived/diagnostics/station_name_check.csv` and
  `inland_depth_provenance.csv`. It reads only derived tables plus the DNR
  polygons; it changes nothing.
- **Name agreement on the 22 spatial matches: 17 exact, 5 MISMATCH.** Four of
  the five are already-dropped M1 stations (Pine->Second, Gibson->Lake Ellen,
  Big Island->Triangle) or the unmatched Antoine. The one surviving mismatch is
  **WARNERS LAKE -> Andrus Lake** (sid 115, 43 obs, all validation, 0.35 km).
  No polygon named Warner/Warners exists within 473 km; the nearest is 473.55 km
  away and the next 506 km. Held for a user decision; not changed.
- **Duplicate-name safety:** no inland station has another polygon with the
  same name closer than 0.9 km. Smallest margin is Twin Lakes at 1.93 km
  (next: Michigamme 36.5, Triangle 58.9, Second 107.2, Bear 116.6,
  Crystal 117.3, Warner 179.4/..., Chain 182.6). Coordinate uncertainty
  (+/-0.9 km) cannot flip any of the 22 matches.
- Station names that differ only by a place prefix still resolve correctly:
  BEAVER ISLAND-LAKE BARNEY -> Barneys Lake (0.30 km),
  S. MANITOU IS.-LAKE FLORENCE -> Florence Lake (0.18 km),
  MUSKEGON LAKE-SNUG HARBOR -> Muskegon Lake (0.17 km).
- **Bear Lake (1517): surveyed, `depth_source=point+contour`, `depth_imputed=0`,
  4.0 ft = 1.22 m, 9 contours, 0.402 km2, mean 0.81 m.** It is the only Bear
  Lake within 116.6 km; 13 Bear Lakes exist statewide, the next at 116.6 km.
- **Crystal Lake (3523): surveyed, `point+contour`, `depth_imputed=0`,
  175 ft = 53.34 m, 18 contours, 39.9 km2, mean 19.84 m,** next Crystal Lake
  117.3 km away.
- Of the 123 H2-corrected observations (43.8% of the 281 Michigan inland
  training observations), **61 are surveyed** (Bear 33 + Crystal 28) and
  **62 are imputed** (Indian 34 at 19.96 +/- 21.9 m, Twin 28 at
  12.19 +/- 9.8 m).

**2026-11-07 - Gate B follow-up: Warners Lake dropped**
- The name audit flagged **WARNERS LAKE -> Andrus Lake** as the only surviving
  station whose name and coordinate disagree. No Michigan lake is named
  Warners: the MDNR statewide named-lakes list has zero occurrences, the two
  Warner Lake polygons are 473.5 and 506.3 km away, and the coordinate is
  0.35 km from Andrus Lake with nothing else within 3.8 km. Decision: **drop.**
- Applied by adding sid 115 to the M1 set in `build_observations.py`.
  **New pool: 4,785 obs, 74 lakes, 3,671 train / 1,114 val (23.28%), 15 val
  lakes.** Michigan inland becomes 561 obs / 17 lakes (10 train, 7 val).
  Removed 43 val obs; no training row changed.
- The two remaining mismatch stations in the audit table (Pine, Gibson, Big
  Island) were already dropped, and Antoine has no polygon within 5 km, so the
  name audit is now clean for every station in the pool: **17 stations, all
  exact name matches.**

**2026-11-08 - Gate C and the model fit**
- `shallow_bay_check.py` (exits 0) computed point vs 1.5 km vs 3 km water-depth
  medians for all 49 Great Lakes stations. Confirms the four flagged stations
  are genuine shallow points (ratios 0.09-0.58); Grand Portage Bay is the only
  other station below 0.5 (0.44). Depth variants V0-V3b written to
  `data/derived/depth_variants.csv`.
- `sample_cfdd.py` (exits 0) joins CFDD bilinearly in space and linearly in time
  between 3-day steps. **4,766 observations joined; 19 January-April 2026
  observations (15 train, 4 held-out) flagged** in
  `diagnostics/cfdd_missing_2026.csv`, per D2.
- GHCN 1952 spot check is now tied to named station IDs and reproducible:
  Duluth 288 vs 216 (-25%), Marquette 98 vs 87 (-11%), Ann Arbor 50 vs 47
  (-6%). The earlier ad-hoc table (Marquette 131, Ann Arbor 78) did not
  reproduce and is superseded.
- `fit_model.py` (exits 0) fits V0-V3b plus an unconstrained V0u sensitivity.
  **Adopted fit (V0, C0 >= 0): k = 1.6049 +/- 0.0115, C0 = 0 (bound),
  C1 = 26.52 +/- 4.49.** Training RMSE 13.73 cm on 3,656 obs.
- **Held-out (1,110 obs): RMSE 12.16 cm, MAE 9.11, bias +1.24, R2 0.479.**
  Constant-mean baseline RMSE 16.90 cm (28% better). Note: the per-month
  baseline in the next entry is the stronger comparison and the model does not
  establish skill over it.
  Michigan inland: RMSE 11.96 cm, R2 0.46 (280 obs, 8 lakes).
- **Variants are indistinguishable:** common-subset RMSE 11.68-11.81 cm across
  V0-V3b (0.13 cm spread). Recommend keeping V0.
- **Depth-bias gate:** all held-out slope +2.73 +/- 0.38 cm per ln(m) - fails.
  Excluding the three validation-only flowing-water stations (127 Portage
  Waterway, 147 East Nipigon, 314 St. Marys River) gives **+0.23 +/- 0.54,
  which passes**, and changes no fitted parameter. Reported as a separate line.
- **Season:** growth-only model. April RMSE 19.72 cm (bias -6.79), May 38.19 cm
  (4 obs). Needs a melt rule before the animation.
- **Snow:** held-out residual slope -0.14 cm per cm snow (SE 0.07), the expected
  insulation sign; small (R2 0.025), so no fourth parameter in v1.
- Unconstrained sensitivity (V0u): C0 = -111.3, C1 = 70.2, held-out RMSE
  12.39, and it predicts 15.6 cm at S = 0 on Bear Lake (1.2 m). Rejected on
  physical grounds as well as error.
- Full write-up: `reports/model-v1.md`. Outputs: `training_matrix.csv`,
  `fit_results.csv`, `validation_predictions.csv`, `diagnostics/fit_report.txt`.

**2026-11-08 - Gate D corrections and cluster bootstrap**
- `bootstrap_model.py` (exits 0) added. Held-out lake-cluster bootstrap (2,000
  replicates, 15 lakes, seed 20261108): RMSE 95% 10.80-13.12 cm, MAE 7.91-10.09,
  bias -2.83 to +5.52, depth-bias slope **-2.01 to +6.63**, snow slope
  **-0.17 to -0.07** cm per cm snow.
- Training-lake refit bootstrap (2,000 replicates, 59 lakes): k 1.494-1.742,
  C0 0.000-0.017, C1 **5.95-51.64**. The Gauss-Newton SEs in `fit_report.txt`
  (k +/- 0.0115, C1 +/- 4.49) are **not** cluster-robust and understate
  uncertainty. C0 was above zero in 15.5% of replicates.
- **Per-month baseline is the honest headline:** 12.60 cm vs model 12.16 cm;
  paired cluster difference -0.44 cm, 95% [-1.94, +0.65]. The model beats
  climatology in Jan-Mar only and loses in Nov, Dec, Apr, May. No established
  skill over seasonal climatology.
- **C1 = 0 fit** (k 1.5436, C0 4.56): held-out RMSE **11.89** cm, bias +0.83,
  depth slope +1.54. Paired difference RMSE(V0) - RMSE(C1=0) = +0.27 cm,
  95% [+0.03, +0.51]. The depth term costs ~0.3 cm of held-out accuracy; it is
  kept on physical grounds, not on measured skill.
- **Depth-bias gate still fails** on the full held-out set (+2.73, GN SE 0.38).
  The post-hoc subset excluding 127/147/314 (12 lakes, 759 obs) gives +0.23,
  cluster 95% [-3.68, +4.44], RMSE 11.74 [9.95, 13.09]. Clearly labelled post
  hoc; all three stations remain in every headline number.
- **G00803 snow depth is parseable at [36:38]** (Fortran format 100): 3,758 of
  7,547 records populated, and metric values match the English file x 2.54 to
  a median 0.3 cm. The pipeline reads only [11:20], so **0 of 3,754 G00803 obs
  carry snow_cm**; the snow diagnostic rests on 992 NTL/ECCC obs. No pipeline
  change made.
- **ERA5 melt feasibility:** the two files in `data/raw/era5` are CDS ZIPs;
  `year_1966_4h.nc` contains a full year of 4-hourly t2m on the 81x171 grid.
  The rest of the raw archive was deleted, so `P` needs a new sequential pull:
  76 yearly requests, Mar-Jun, ~1 GB. 1966 supports a pipeline pilot only
  (6 spring observations).
- Report rewritten: `reports/model-v1.md` (residual = observed - predicted,
  per-month baseline headline, C1 = 0, cluster intervals, G00803 finding,
  melt feasibility).

**2026-11-08 - P pilot and melt brief amendments**
- `build_pdd.py` (exits 0) unpacked `year_1966_4h.nc` (a CDS ZIP) to 365 daily
  means on the exact 81x171 CFDD grid and wrote `data/derived/pdd/pdd_1966.npz`
  (123 dates, uint16, P in degC-day). P is zero on all 20 pre-March and all 52
  post-July steps; max inside 1 Mar-31 Jul is 2,215 degC-day.
- 18 observations fall in 1966, 6 in March; P sampled at each. Pipeline works.
- GHCN March-May 1966 positive degree-days vs ERA5 at the same points:
  Duluth 339 vs 302 (-10.9%), Marquette 416 vs 257 (-38.1%), Ann Arbor 815 vs
  639 (-21.5%), 92 days each. ERA5 P is low; a low P biases `m` upward. To be
  re-checked after the full pull.
- Melt brief amended per instruction: criterion 1 is April only; May is
  reported but not pass/fail. Two arms, A (k, C1, m) and B (k, C0, m),
  selected by repeated lake-grouped CV on the 59 training lakes with a fixed
  seed; B only if its CV error beats A with a lake-cluster interval excluding
  zero. Held-out is reporting only.
- Pull plan recorded: 1 Mar-31 Jul, sequential, 1966-2025 then 1950-1965,
  resumable, disk checked first, failed years retried and listed, never a
  silent P = 0. Not started; awaiting approval.
- `reports/model-v1.md` now leads with the climatology finding ("At a glance").

**2026-11-08 - G00803 snow_cm added (diagnostic only)**
- `build_observations.py` now reads snow depth from `[36:38]` and no longer
  forces G00803 `snow_cm` to NaN. Backups in
  `data/derived/backups/pre_g00803_snow/`.
- `verify_snow_column.py` (exits 0): observations and training matrix match the
  backups in row order and every other column; counts 4,785 / 74 / 3,671 /
  1,114; split identical; `fit_results.csv` and `validation_predictions.csv`
  identical, so held-out RMSE is unchanged at 12.16 cm. All 2,545 G00803 snow
  values match a direct parse of `data2.dat`.
- Snow coverage: G00803 2,545/3,754 (67.8%), NTL 972/991, ECCC 20/21; 145
  explicit zeros, all NTL. Storage: 83% in January, 87% in February, 41% in
  April, 0% in May - not missing at random.
- `snow_diagnostic.py` (exits 0), lake-cluster intervals: G00803
  **+0.150 [0.019, 0.282]**, NTL **-0.097 [-0.144, -0.054]**, ECCC -0.208
  [-0.469, 0.188], pooled **+0.086 [-0.018, 0.183]**. The earlier negative
  pooled slope was a source-composition effect and is withdrawn. No snow term.
- Parameter fitting, CV and validation do not read `snow_cm`; only diagnostic
  blocks in `fit_model.py` and `bootstrap_model.py` do.

**2026-11-08 - Read-only skill diagnostics (no model change)**
- `skill_diagnostics.py` (exits 0) added. Split, parameters and fitted outputs
  untouched; it reads `fit_results.csv` and `bootstrap_intervals.csv`.
- **Anomaly skill.** Month-demeaned: corr +0.377 [0.226, 0.569], slope of
  observed on predicted +0.593 [0.374, 0.978], skill over zero anomaly +0.065.
  Within held-out lake-month cells: corr +0.407 [0.343, 0.464], slope +0.834
  [0.713, 0.986], skill +0.159. The model does track winter severity
  (about 16% of within-cell variance); the RMSE failure against climatology is
  cross-lake level error, not failure to respond to cold.
- **Low-CFDD depth lever.** At S = 50 the model gives 7.3 cm at 3 m and 0 ice at
  10 m and 30 m; at S = 100 the 3 m -> 30 m lever is 8.5 cm. Across the C1
  cluster interval the sign can flip (C1 2.5% gives near-zero lever). This is
  the model's least-supported claim and an argument for the no-depth arm B.
- **V0 vs C1 = 0.** Nov-Dec: V0 8.19 vs C1=0 6.74 cm; paired cluster
  difference +1.63 cm [0.31, 3.22], excludes zero, so the depth term measurably
  hurts early season. Jan-Mar difference only +0.09 cm. 82.3% of held-out
  observations exceed the 20.3 cm clamp, so the both-clamped metric (3.23 vs
  2.83) is compressed and measures display saturation, not skill.
- `reports/model-v1.md` updated: anomaly-skill subsection, low-CFDD lever table,
  Nov-Dec/clamp table, limitations, scripts and outputs.

**2026-11-08 - Hourly P timing trial and 1966 definition comparison**
- `fetch_pdd_trial.py` (exits 0): one CDS request, 1 Mar-31 Jul 1966, all 24
  hours. Wall time **668 s (11.1 min)**, compressed **94.9 MB**, 3,672 steps
  (153 days x 24 h) on the exact 81x171 grid. Linear 76-year estimate
  **14.1 h, 7.2 GB**. Raw file kept at
  `data/raw/era5/pdd_trial_1966_hourly.zip`. Timing in
  `diagnostics/pdd_trial_1966_timing.{txt,json}`.
- `compare_pdd_definitions.py` (exits 0) built three definitions on the CFDD
  grid and wrote `data/derived/pdd_trial/pdd_1966_{H,P4,M4}.npz`. All three are
  zero before 1 March and after 31 July; maxima 2,233 / 2,230 / 2,215 degC-day.
- 31 May cumulative P vs GHCN 1966: Duluth 340/331/302 vs 339; Marquette
  281/273/257 vs 416; Ann Arbor 671/659/640 vs 815.
- Mean absolute station bias: **H 16.7**, P4 18.7, M4 23.5. Spread: H 32.5,
  P4 31.8, M4 27.2. The H gain is one station (Duluth); Marquette stays 32% low
  under every definition, so that deficit is station/grid representativeness,
  not the daily-mean definition.
- Pre-written rule (1966-1970, smallest mean absolute bias then smallest
  spread, temperature and GHCN only) would pick **H on 1966**, but one year and
  three stations do not settle it. Gate open: choose hourly pull (~14 h,
  7.2 GB) or 4-sample pull storing both P4 and M4 (~1.2 GB, ~2.5-4 h inferred).

**2026-11-08 - Hourly pull approved and launched**
- Decision: hourly pull, 1 Mar-31 Jul, sequential, 1966-1970 first, then
  1971-2025, then 1950-1965. Resumable (50 MB completeness threshold), four
  attempts per year, status written after every year, raw files kept, disk
  checked per year (20 GB floor). Unresolved years listed in
  `diagnostics/pdd_pull_unresolved.txt`; never `P = 0`.
- `pull_pdd.py` runs `apply_pdd_rule.py` automatically when 1966-1970 are on
  disk, then continues. The rule output is the gate; production P for all
  years is not derived.
- `apply_pdd_rule.py` smoke-tested on 1966 alone (partial outputs suffixed
  `_partial`, since deleted). Reproduces the trial numbers exactly.
- Report statements fixed: Marquette bias is definition-independent
  (32-38% under all three); S and P may use different daily-mean definitions;
  the 1966 ranking rested on Duluth alone.
- Melt brief: arm A vs B CV output must report November-December error as a
  column. No model change.

**2026-11-08 - P-definition rule over 1966-1970: H selected**
- 1966-1970 block completed: 1967 101.5 MB, 1968 97.3 MB, 1969 102.0 MB,
  1970 101.5 MB; all ~656-666 s each. Pull continued into 1971 automatically.
- `apply_pdd_rule.py` ran automatically and wrote
  `pdd_definition_rule_1966_1970.csv`, `pdd_definition_rule_summary.csv`,
  `diagnostics/pdd_rule_1966_1970.txt`.
- Mean absolute station bias over 15 year-station pairs: **H 19.47**, P4 21.09,
  M4 24.16. Mean per-year spread: H 23.21, P4 23.06, M4 21.66. Rule is bias
  first, so **H wins**; the spread tie-break is not reached.
- H wins at all three stations (Duluth 10.3, Marquette 32.0, Ann Arbor 16.1)
  and in all five years, so the 1966 "Duluth alone" concern is resolved.
- Marquette remains 32.0 mean absolute bias under H (33.6 P4, 36.4 M4): station
  or grid representativeness, not definition. H is still 19.5% low overall,
  which biases `m` upward; to be flagged when the melt fit is reported.
- Selected definition: H, hourly-integrated, cumulative from 1 March, zero
  outside 1 Mar-31 Jul. Production P for 1950-2025 NOT derived; gate.

**2026-11-08 - H builder launched; E2 will fire at 1966-2025 complete**
- 1966-1970 built with definition H, checked against the rule table at the
  31 May step: Duluth 339.9 / Marquette 281.7 / Ann Arbor 670.7 for 1966 and
  436.1 / 314.1 / 726.1 for 1969, matching `apply_pdd_rule.py` exactly.
  Per-year maxima 2233/2192/2318/2192/2309 degC-day; zero before 1 Mar and
  after 31 Jul; monotonic inside the window.
- Stale M4 `pdd_1966.npz` backed up to
  `data/derived/backups/pre_pdd_H/pdd_1966.npz` before overwrite.
- Builder test on the 5 years: 1.4-2.0 s per year, `is_valid` skips on rerun
  (no rewrite when relaunched at 14:00).
- `verify_pdd_fields.py` run in partial mode against 1966-1970, then the
  `_partial` outputs were deleted. Fixes made: missing-year sampling returns
  NaN instead of raising, outside-window zero check ignores NaN, and the
  builder checks `is_valid` before the 20 s stability probe so already-built
  years cost nothing.
- Builder launched 14:00:17 (background) and a watcher waits for
  `diagnostics/pdd_verify_1966_2025.txt`. Expected ~03:00 on 2026-11-09.
- Partial-run numbers to be re-read from the full run, not quoted: GHCN
  1966-1970 mean bias -19.5%, Marquette -32.1%, Duluth -10.3%, Ann Arbor -16.1%;
  training matrix 4,766 obs / 74 lakes, 1,762 in the Mar-Jul window.
- No fit, no parameter change, no edit to `training_matrix.csv`.

**2026-11-09 - server restart; pull and builder resumed, nothing lost**
- The harness restarted overnight and cancelled the pull, the builder and the
  watcher. State on disk was clean: 59 raw years (1966-2024) and 59 P fields.
  2025 was mid-request and left no partial file; 1950-1965 not started.
- Restarted at 11:36. Pull skipped the 54 completed years in the main block and
  re-requested 2025 only, then continues to 1950-1965.
- Builder restarted at 11:36; it revalidates the 59 existing H files (no
  rewrite) and will build 2025 as soon as it lands, then fire E2.
- Watcher re-armed for `diagnostics/pdd_verify_1966_2025.txt`.
- Disk 95.0 GB free; 17 years remain (~1.7 GB).

**2026-11-09 - E2 verification passed; at the melt-fit gate**
- 2025 landed in 27 s after the restart (CDS cache hit), 100.1 MB. Production P
  now exists for all 60 years 1966-2025 and the builder fired E2.
- Integrity: 60/60 ok, 123 steps, exact CFDD dates/grid, uint16, zero before
  1 Mar and after 31 Jul, monotonic inside; maxima 2,095 (1984) to 2,908 (2012),
  median 2,345.
- Pull health: no failed years, no retries, nothing unresolved.
- GHCN all years: Ann Arbor -7.7% (n=60), Duluth -9.9% (60), Marquette -21.7%
  (59; 1979 absent), all -13.1% mean / 13.2% mean abs, year means -27.3% (1972)
  to -4.1% (2018). Bias is non-stationary: Marquette -33.7% in the 1960s to
  -9.8% in the 2020s; Ann Arbor -16.4% to -2.1%; Duluth no trend.
- Join: 4,766 obs / 74 lakes, 1,762 in Mar-Jul, 3,004 outside; outside-window P
  exactly 0; 0 NaN and 0 negative in window; S unchanged at 2.3e-13.
- DEFECT FOUND AND FIXED: temporal interpolation crossed 1 Mar and gave February
  observations P up to 2.0. `sample()` now returns exactly 0 outside the window;
  canonical outputs are from the fixed run.
- S/P construction note and the revised `m` caveat are in reports/model-v1.md
  section 9. `m` is calibrated to the same biased P the map displays; the
  station-to-station spread (Marquette vs Ann Arbor) is the real error source,
  and because the bias drifts, a pooled `m` absorbs a moving average.
- No fit run. `training_matrix.csv` untouched. 1950-1965 backfill in progress.

**2026-11-09 - melt arms A/B fitted; arm A retained at the gate**
- `fit_melt.py` written and run (exit 0). New outputs only:
  `melt_cv_results.csv`, `melt_arm_selection.csv`, `melt_fit_results.csv`,
  `melt_validation_predictions.csv`, `melt_m_bootstrap.csv`,
  `melt_residuals_by_era.csv`, `melt_s_vs_ghcn_decade.csv`,
  `melt_two_k_diagnostic.csv`, `melt_sensitivity.csv`,
  `diagnostics/melt_fit_report.txt`.
- Pool unchanged: 3,656 train / 1,110 held-out; 1,417 rows with P > 0.
- CV (10 repeats x 5 folds, lake-grouped, seed 20261106): A 13.66 (sd 0.09),
  B 13.78 (0.06). Paired cluster difference B-A +0.12 cm [-0.09, +0.31],
  includes zero, so **arm A is retained** (depth term kept, C0 = 0).
- Held-out, reporting only: A 11.23 cm, B 10.95 cm. Held-out prefers B by
  0.28 cm while CV prefers A; recorded, not used for selection. B is also
  better in Nov-Dec (7.13 vs 8.73).
- Arm A: k 1.6865, C1 34.38, m 0.3318 (95% cluster [0.2082, 0.4744]).
- Pass criteria for A: April 15.24 vs baseline 18.22 cm (n=102) PASS; May 22.19
  (n=4, reported); Nov-Mar 10.67 vs V0 10.87, -0.21 cm PASS; m interval
  excludes zero PASS; P spot check done at E2.
- Era diagnostics: train winter RMSE 14.51 (1966-79) vs 8.39 (1982+), spring
  14.78 vs 10.08; early era is worse in variance, not bias. Separate-k refit
  returns k 1.6864 vs 1.6866 (a no-op), so the era gap is not a scale effect;
  era and source are collinear. Not adopted.
- S vs GHCN Oct-Mar: complete seasons mean +11.1%, mean abs 17.5%; Duluth low
  (-6 to -12%), Ann Arbor and Marquette high (up to +45% in the 1960s). Two
  partial Marquette seasons (1978, 2013) excluded from the percent summary and
  kept in the CSV. Construction contributes: S is the positive part of a
  4-hour mean, GHCN a daily mean.
- Zero-thickness records Mar-Jul: **0** (min thickness 1 cm), so the labelled
  sensitivity refit is identical by construction.
- No parameter change applied, no edit to `training_matrix.csv`, no held-out
  use in selection. Awaiting the arm-choice gate.

**2026-11-09 - v2 adopted: arm A canonical, v1 archived, counts and classes verified**
- Confirmation before applying (`confirm_arm_a.py`):
  April held-out 15.24 vs baseline 18.22 cm, n=102, 9 lakes, paired
  (baseline - A) +2.97 cm [+1.55, +3.60]; m 0.3318 [0.2082, 0.4744];
  CV overall A 13.66 / B 13.78, B-A +0.12 [-0.09, +0.31]; CV Nov-Dec
  A 8.65 / B 8.28, B-A -0.37 [-1.55, +0.67].
- Arm A kept under the pre-agreed asymmetric rule (B did not beat A with an
  interval excluding zero). Depth retained on product grounds.
- v1 archived to `data/derived/backups/pre_v2_melt/` (fit_results.csv,
  validation_predictions.csv, fit_report.txt, fit_model.py). v1 canonical files
  stay in place untouched as the frozen comparison; `fit_model.py` header now
  points to v2.
- Canonical v2 outputs (new files): `fit_results_v2.csv`,
  `validation_predictions_v2.csv`, `model_parameters.csv`
  (k 1.6865, C0 0, C1 34.38, m 0.3318; train RMSE 13.42, held-out 11.23,
  MAE 8.69, bias +1.37, R2 0.556).
- Count verification (`v2_count_verification.txt`): all 9 checks OK. 4,785 obs /
  74 lakes; matrix 4,766; 3,656 / 1,110; matrix keys and all columns except
  snow_cm identical to the pre-snow backup; held-out keys identical to v1.
- Held-out classification (`v2_classification.csv`): 8 in 0.893
  [0.851, 0.932], 914 observed / 961 predicted above, 83 false above, 36
  missed; 4 in 0.947 [0.920, 0.972], 1,032 / 1,045, 36 false above, 23 missed.
  Class imbalance: a constant above rule scores 0.823 and 0.930.
- `reports/model-v1.md` header, limitations and section 9 updated; the report
  documents both v1 and v2.

**2026-11-09 - raster gate checks: v2 clears climatology; log depth proposed**
- `v2_vs_climatology.py` (new): held-out 1,110 / 15 lakes; v2 RMSE 11.23 cm vs
  per-month baseline 12.60 cm; paired difference **-1.38 cm [-2.61, -0.66]**,
  2,000 cluster draws. Unlike v1 (-0.44 [-1.94, +0.65]) the interval excludes
  zero. Monthly delta (baseline - v2): Jan +1.01, Feb +1.48, Mar +1.65,
  Apr +2.97, May -1.88 (n=4), Nov -4.23 (n=6), Dec -1.20 (n=110).
- v2 held-out depth slope recomputed for the report: +3.04 cm per ln(m)
  [-1.55, +6.41] (v1 was +2.73 [-2.01, +6.63]); both intervals include zero.
- `depth_encoding_proposal.py` (new, proposal only): current uint8 1 m scheme
  clips 1,098,097 Superior pixels (7.68% of the lake, 6,354 km2) and 77,859
  Michigan pixels (0.83%, 483 km2) at the 255 m cap; Huron none. Lake mask is
  bathy elevation below the surface; 3 arc-second cells.
- Proposed v = clip(round(ln d * 40), 0, 255), floor 1 m: 2.53% per code step,
  cap 587 m (0 pixels clipped), half-step error 0.430 degC-day at every depth
  vs the 1 m scheme's C1*ln(d/(d-0.5)) (13.9 at 1.2 m, 9.9 at 2 m).
- Thickness error at S=100: log 0.04-0.08 cm vs 1 m scheme 1.21 cm at 1.2 m,
  0.96 at 2 m, 0.67 at 3 m. Worst near onset ~1.11 cm before clipping.
- Floor: 0.31% of Superior, 0.46% Michigan, 1.07% Huron lake pixels below 1 m;
  10 of 3,548 MI inland lakes (0.28%) have max depth < 1 m.
- Report updated: At a glance, depth-bias paragraph, depth-term paragraph,
  limitation 1, and a new "Raster gate checks" subsection in section 9.
- Nothing applied: no raster, encoder or product file changed. Stopped at the
  raster gate.

**2026-11-09 - raster output proposal; station depth by source reported**
- `station_depth_by_source.py` (new): V0 of the training matrix, observation
  weighted. g00803 n=3,754 / 62 lakes, min 0.92, p5 2.23, p25 3.25,
  median 7.70, p75 11.41, p95 21.34, max 109.31 m. ntl n=991 / 11,
  min 2.50, median 20.00, max 35.70. eccc n=21 / 2, min 13.00, max 20.40.
  Pooled n=4,766 / 74, min 0.92, p5 2.23, p25 3.31, median 8.30,
  p75 12.34, p95 29.36, max 109.31.
- `raster_grid_proposal.py` (new, proposal only): common grid EPSG:4269,
  3 arc-second, pixel-is-area, same lattice as NOAA.
- Great Lakes: one combined mosaic 9,457 x 15,025 (142.1 M cells),
  33,489,600 water pixels, lake codes 1/2/3. Depth PNG 5.18 MB full /
  4.74 MB in 26 non-empty 2048 tiles; mask 0.47 / 0.42 MB.
- Inland: 7,736 x 9,114 (70.5 M cells), 493,108 water pixels (0.70%),
  codes lake_idx+1 (1..3548). Mask 0.61 MB full / 0.52 MB in 15 tiles;
  depth flat lower bound 0.29 MB. Collar adds 339,814 flagged land pixels.
- Total depth+mask download 5.98 MB (GL 5.17, inland 0.81), 82 PNG tiles.
  Inland depth will rise with contour detail: budget 6.0-6.3 MB total.
  Per-frame S/P is separate, 81x171 uint16 ~28 KB each; no raster stack.
- Mask encoding: 0 = none, low bits = lake code, high flag = land collar
  (128 uint8 / 32768 uint16). Collar is for filtering, not drawn as water.
- Straits: all three tiles share the lattice; offsets Superior->Michigan
  (+5040,+4092), Superior->Huron (+9240,+3600), Michigan->Huron
  (+4200,-492). No resampling. Water overlap 100,446 px: 34,908
  Superior-Michigan (northern Lake Michigan), 65,202 Superior-Huron
  (St Marys / North Channel), 336 Michigan-Huron (one-pixel seam at
  -84.5004). Build rule must be nearest lake body, not tile order. The
  0.0008 deg Michigan-Huron gap is one pixel and is collar-filled.
- GATE NOTE: the bathy mask (elevation < 0) includes the St Marys River /
  North Channel, which is outside the three-lake scope. Options: (a) clip
  with a Great Lakes shoreline polygon at the NOAA 0 m contour; (b) exclude
  connecting channels by corridor. Recommendation (a). Not applied.
- Nothing generated: no raster, tile, mask or encoder output written.
  Stopped at the raster gate.

**2026-11-10 - shoreline gate: HydroLAKES + St Marys ESI + DNR wins**
- `build_shoreline_mask.py` (new, exits 0, no product raster). One 3″ grid,
  EPSG:4269, 9,457 x 15,025 = 142.1 M cells.
- Lake-body polygons: HydroLAKES v1.0 (CC-BY 4.0), Superior/Michigan/Huron
  kept separate; Huron includes Georgian Bay and the North Channel; the
  Straits of Mackinac are in. `mask_detail` (ESI 2021) is not used.
- Connecting rivers: NOAA ESI `GL_St_Marys_River_2021` HYDROP (CC0-1.0),
  538 water polygons unioned and dilated two cells, clears **60,952 Huron px
  (0.624%)**. The ESI study area stops at the documented river mouth
  (46.05556 N); it does not cut the North Channel.
- Mask = lake-body polygon ∩ bathy water (elevation < 0). The 164,117 px
  (973.1 km²) that the polygon labels as lake but the bathy has no depth for
  are **left out** (one-pixel shoreline ring plus the Straits seam); no seam
  fill.
- DNR inland polygons win inside their boundary: **1,443 Great Lakes px
  cleared** where a Michigan inland lake overlaps a lake body.
- Approved counts: **GL 33,015,374 px / 196,739.5 km²**
  (Superior 14,050,667; Michigan 9,307,694; Huron 9,658,456), **inland
  493,108** water px. Gate report
  `data/derived/diagnostics/shoreline_mask_gate.txt`; previews
  `gate_stmarys/straits/greenbay/saginaw/dnr_overlap.png`.
- Nothing else consumed the mask until the raster build below.

**2026-11-10 - product rasters (3″) and read-back verification**
- `build_rasters.py` (new, exits 0, 49.6 s): depth PNG tiles 2048², 24 for the
  Great Lakes and 14 for the inland layer; empty tiles skipped.
- Encoding preserved as approved: `v = clip(round(ln(max(d,1))*40), 0, 255)`,
  separate uint8 GL mask (lake codes 1/2/3) and uint16 inland mask
  (lake_idx+1, high flag 32768 = land collar). Two passes of collar fill give
  the shoreline one land pixel of the nearest lake code so the 3″ product
  antialiases; the collar is flagged and never counted as water.
- Inland depth: 1,552 lakes from contour interpolation, 1,996 from a bowl
  (max/mean) shape; island holes in 919 polygons are honoured, 64
  MultiLineString contours are supported, and the collar shift no longer
  wraps at the grid edge (three real bugs found and fixed before the run).
- `verify_rasters.py` (new, exits 0): **all checks PASS** read back from the
  PNGs — GL 33,015,374 px, inland 493,108 px, 3,548 distinct lake codes,
  decoded depth ≥ 1 m on every water cell, St Marys mouth clear (0 Huron cells
  in a 17x17 box). Previews `rasters_gl_depth.png`, `rasters_inland_zoom.png`,
  `rasters_mask_zoom.png`.
- Canonical rasters stay at 3″; the web derivative is 6″ (below).

**2026-11-10 - animation fields, web layers, viewer, shader validation**
- Season = **1 Oct → 30 Jun**. S resets 1 Oct, P is defined 1 Mar–31 Jul, and
  stopping at 30 Jun avoids the August jump when P returns to zero with S
  still large. 75 seasons 1950/51–2024/25, 92–93 frames each (~6.9k frames).
- `build_web_fields.py` (new, exits 0, 18.6 s): one gzip per season,
  magic `ICEF1\0\0\0` + `uint32 n, h, w` (20-byte header so the uint16 payload
  is 4-byte aligned), then S frames then P frames, 81x171. Total 89.3 MB.
- `build_web_layers.py` (new, exits 0, 11 s): 6″ (~185 m) decimation to
  **coverage + depth pairs** — `gl_cov/gl_dep` 7,513x4,729 and
  `inland_cov/inland_dep` 4,557x3,868, 2.3 MB total. Two channels, not one
  byte, because a single mipmapped byte averages land 0 with deep water and
  paints a bright halo on every shore; the shader recovers
  `code = dep*255/cov`. GL blocks use a 2-of-4 majority, inland any-of-4 (a
  10 ha lake is 1–2 cells). Out-of-scope water (Erie, Ontario, St Clair,
  rivers, tile-edge neighbours) is 151,756 cells at code 255 and draws as flat
  dark water, never ice.
- Viewer `web/index.html`, `web/style.css`, `web/app.js`: WebGL2, one
  fragment shader evaluating `h = max(0, k·sqrt(max(0, S − C1·ln d)) − m·P)`,
  R16UI array textures for S/P, manual bilinear on the 0.1° field, mipmapped
  coverage/depth sampling, navy→white ramp with white at ≥20.32 cm, season
  scrubber, speed, play/pause, wheel zoom, drag pan, dblclick reset, subtle
  blue city/lake labels, disclaimer. Next two seasons prefetched; four kept
  in GPU memory.
- Inland lakes fade out below ~2x zoom: a 10 ha lake is ~0.3 px statewide, so
  an honest average reads as stipple rather than information. They appear
  when zoomed in.
- `serve_web.py` (new): local static server, no-store, port 8000.
- `validate_shader.py` (new, exits 0): 2,205 samples of the real
  canvas at zoom 8 (mip 0) against a Python reference that re-derives the
  colour from the same files. **452 flat samples match exactly (0/255 on every
  channel), including 8–11.6 cm ice**; 1,753 samples sitting on depth contours
  are within 6/255 (~2 cm), the expected half-texel effect. Painless proof
  that depth decode, field bilinear, constants, ramp and coverage blend on the
  GPU match the model.
- Visual QA in the browser: no console errors; Feb 1967 statewide view
  (Superior solid, southern Huron and the Michigan deep basin thin), May melt
  gradient, Houghton Lake per-pixel depth at zoom 6, Georgian Bay/North
  Channel islands at zoom 5, context water dark, land dark. The Lake Michigan
  mid-winter blue interior is the real depth/water-temperature term, not a
  rendering artifact.

**2026-11-10 - the black screen was a CSS overlay, not the October default**
- The map was covered by an opaque full-screen panel the whole time.
  `#error` is styled `position:absolute; inset:0; display:grid; background:
  #05080d`, and an author `display` rule **overrides** the UA stylesheet's
  `[hidden] { display: none }`. So the error panel, empty and `hidden`, was
  painted on top of the canvas on every load. The page looked black while every
  pixel of the framebuffer was correct.
- This is why the earlier checks all passed: `gl.readPixels()` and
  `canvas.toDataURL()` read the canvas only, and never see a DOM overlay above
  it. The page was "healthy" by every measurement that ignored compositing.
- Fixed with one rule, `#error[hidden] { display: none; }`. Verified in the
  live page: with no error, `elementFromPoint(centre)` is the canvas and
  `#error` computes `display: none`; when `fail()` runs, the panel computes
  `display: grid` and is the topmost element. A real page screenshot (composited,
  not `toDataURL`) now shows the map, chrome and labels.
- **Process note:** for this page, verify with a composited screenshot and
  `elementFromPoint`, not with canvas readback alone. A canvas readback cannot
  detect anything drawn over the canvas.
- `reports/figures/app_page.png` is a composited screenshot; `app_shot.png` is a
  canvas-only capture and cannot show overlay defects.

**2026-11-10 - opening view fixed: the map opened on a season with no ice**
- A clean load opened on **1 October 1950/51**. S = 0 everywhere that day, so
  the canvas was 70.5% land `(12,15,20)` and 20.8% water `(10,21,43)` — close
  enough in luminance that the page read as an empty dark rectangle even though
  the server, textures and shader were all healthy.
- Default is now **1 February of the most recent season** (2024/25), with
  `?season=1966` still selecting a season. A fresh load reports 5,999 distinct
  colours and 5.8% white.
- Ramp floor lifted from `(0.039, 0.082, 0.169)` to
  `(0.043, 0.098, 0.196)` so zero-thickness water stays visibly water rather
  than merging with land; the CSS legend start moved to `#0b1932` to match.
- `validate_shader.py` re-run against the new ramp on a fresh 2,205-sample
  capture: **452 flat samples still match exactly (0/255)**, 1,753 contour
  samples within **6/255**. Recorded in `diagnostics/shader_validation.txt`.
- `web/README.md` (new): serve command, controls, data flow, encoding, default
  view and the inland-zoom caveat.
- **Docs encoding repaired.** A PowerShell `Get-Content -Raw` / `Set-Content
  -Encoding utf8` round-trip (Windows PowerShell 5.1 reads BOM-less UTF-8 as
  cp1252) had double-encoded this file and `model-v1.md`, turning every em dash
  into `â€”` and every `±`, `°`, `≥` into mojibake. Fixed with
  `python/repair_double_encoded.py`, which reverses only the mojibake runs and
  leaves correctly encoded text alone, after backing up to `*.bak-mojibake`.
  Do not use PowerShell `Set-Content` to rewrite these reports; use the editor
  or Python with an explicit `encoding="utf-8"`.
