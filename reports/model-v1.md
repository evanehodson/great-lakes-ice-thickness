# Model v1 — three-constant ice thickness

**Date:** 2026-11-09 (v2 melt adopted; v1 growth-only archived)
**Status:** v2 is canonical — arm A, `k = 1.6865`, `C1 = 34.38`, `m = 0.3318`,
`C0 = 0`. v1 (no melt) is retained as the frozen comparison.
**Scripts:** `python/sample_cfdd.py`, `python/shallow_bay_check.py`,
`python/fit_model.py` (v1), `python/bootstrap_model.py`,
`python/skill_diagnostics.py`, `python/build_pdd_fields.py`,
`python/verify_pdd_fields.py`, `python/fit_melt.py`,
`python/confirm_arm_a.py`, `python/fit_model_v2.py`
**Reproduce:** `$env:PYTHONIOENCODING="utf-8"`, then run the fit scripts in that
order. Each exits 0 and writes new files only. Canonical v2 parameters are in
`data/derived/model_parameters.csv`; v1 outputs are archived in
`data/derived/backups/pre_v2_melt/`.

> **Sign convention:** residual = observed − predicted everywhere in this
> report. A positive residual means the model under-predicts; a positive
> residual-vs-depth slope means deep lakes are under-predicted relative to
> shallow ones.

---

## At a glance

**v2 now has established skill over seasonal climatology on levels — narrowly.**
On 1,110 held-out observations v2 scores RMSE 11.23 cm against 12.60 cm for the
per-month training-mean baseline. The paired lake-cluster difference is
**−1.38 cm, 95% [−2.61, −0.66]** — it excludes zero. v1 did not clear this bar
(12.16 cm, −0.44 [−1.94, +0.65]). The gain is concentrated in February–April
(monthly deltas +1.48, +1.65, +2.97 cm); November and December are still worse
than climatology (−4.23 on 6 observations, −1.20 on 110).

**But it does track winter severity.** Within held-out lake-month cells the
model's anomaly correlates with the observed anomaly at **r = 0.41** (cluster
95% [0.34, 0.46]) with a slope of **0.83** ([0.71, 0.99]), explaining about 16%
of within-cell variance. The level errors that dominate RMSE are cross-lake
offsets, not a failure to respond to cold.

**The depth-bias check is unresolved.** v2's full held-out residual slope is
**+3.04 cm per ln(m)** (lake-cluster 95% [−1.55, +6.41]); v1 was +2.73
[−2.01, +6.63]. Both point positive, neither interval excludes zero. The
+0.23 cm per ln(m) result is a **post-hoc** subset that removes three
flowing-water stations; it is reported separately and changes nothing.

**The depth term is weakly identified.** For v1, `C1 = 0` scored 11.89 cm
held-out versus 12.16 (paired +0.27 [+0.03, +0.51]). With melt, arm B (`C1 = 0`)
scored 10.95 cm held-out versus A's 11.23, but the training-CV difference was
+0.12 [−0.09, +0.31] and the pre-agreed rule keeps A. Depth is retained on
product grounds, not measured skill.

**What v1 is:** a mid-winter growth curve, ±12 cm, useful for guidance, not a
safety product. It cannot melt. The melt term's P definition is fixed
(hourly-integrated, chosen on 1966–1970 GHCN temperature alone) and production P
is now built and verified for 1966–2025. The melt fit is done: arm A is
retained by training CV, the held-out set is reported, and the arm choice is at
the gate.

---

## 1. Model

```
h = k * sqrt(max(0, S - C0 - C1 * ln(d)))
```

| Symbol | Meaning | Units |
|---|---|---|
| `h` | ice thickness | cm |
| `S` | cumulative freezing degree-days since 1 October at the observation point | °C-day |
| `d` | water depth at the station (deep hole / point sample) | m |
| `k`, `C0`, `C1` | three fitted constants | — |

**Physical constraint: `C0 >= 0`.** With no freezing degree-days there is no
ice. The unconstrained fit puts `C0 = -111`, which predicts 15.6 cm of ice on a
1.2 m lake at `S = 0` — ice on 1 October. The constrained fit is also better on
held-out data. `C0` lands on the bound, so in practice this is a two-parameter
fit with a physical onset floor.

## 2. Data

| | |
|---|---:|
| Observations | 4,766 (of 4,785; 19 in 2026 have no ERA5 yet) |
| Lakes | 74 |
| Training | 3,656 obs / 59 lakes |
| Held-out | 1,110 obs / 15 lakes |
| Date range | 1966-01-18 to 2025-03-18 |

CFDD is bilinear in space and linear in time between the stored 3-day steps.
The 19 observations from January–April 2026 are listed in
`data/derived/diagnostics/cfdd_missing_2026.csv` (15 train, 4 held-out) and are
excluded per decision D2.

## 3. Parameters and uncertainty

| Parameter | Point estimate | Gauss-Newton SE | Lake-cluster 95% |
|---|---:|---:|---:|
| `k` | 1.6049 | 0.0115 | 1.494 – 1.742 |
| `C0` | 0.0 (bound) | 8.6 | 0.000 – 0.017 |
| `C1` | 26.52 | 4.49 | 5.95 – 51.64 |

**The Gauss-Newton SEs are not cluster-robust.** They treat every observation
as independent; observations repeat within lakes, so they are too narrow. The
cluster intervals come from `bootstrap_model.py`: 2,000 replicates resampling
the 59 training lakes with replacement and refitting all three constants each
time (seed 20261108). `C0` was above zero in 15.5% of replicates — the bound is
active but not imposed by a knife-edge.

`C1` is the weakly identified parameter: its cluster interval spans an order of
magnitude. That matters for interpretation, below.

Implied onset threshold `C0 + C1 ln(d)`:

| Depth | Threshold |
|---:|---:|
| 1.2 m | 4.8 °C-day |
| 3 m | 29.1 |
| 10 m | 61.1 |
| 30 m | 90.2 |

### Depth lever at C1 uncertainty

Predicted thickness at the median held-out CFDD (618 °C-day):

| Depth | Point | C1 − 2 GN SE | C1 + 2 GN SE | C1 cluster 2.5% | C1 cluster 97.5% |
|---:|---:|---:|---:|---:|---:|
| 3 m | 38.93 | 39.26 | 38.60 | 39.67 | 38.01 |
| 10 m | 37.86 | 38.56 | 37.15 | 39.44 | 35.84 |
| 30 m | 36.85 | 37.91 | 35.77 | 39.22 | 33.74 |

Across the full cluster interval the 3 m→30 m lever is **0.45–4.27 cm**; on the
point estimate it is 2.1 cm. The depth term is real only in the sense that it
is positive; its magnitude is poorly constrained.

### At low CFDD the depth term is a heavy switch

The lever above is a mid-winter number, where `sqrt(S − C1 ln d)` is flat. Early
in the season it is steep: a 30 m lake needs `C1 ln 30 ≈ 90 °C-day` before any
ice forms, so at S = 50 the model predicts zero ice for 10 m and 30 m lakes and
7.3 cm for a 3 m lake. This is the model's strongest and least-supported claim.

| S (°C-day) | h(3 m) | h(10 m) | h(30 m) | 3→30 m lever | Lever at C1 2.5% / 97.5% |
|---:|---:|---:|---:|---:|---:|
| 50 | 7.33 | 0.00 | 0.00 | −7.33 | −1.82 / 0.00 |
| 100 | 13.51 | 10.01 | 5.02 | −8.49 | −1.18 / −10.56 |
| 200 | 20.98 | 18.92 | 16.82 | −4.16 | −0.80 / −11.29 |
| 618 | 38.94 | 37.87 | 36.87 | −2.07 | −0.45 / −4.27 |

The sign is negative because deeper water freezes later. The uncertainty at low
CFDD is as large as the effect itself, and the "zero ice below 90 °C-day on a
30 m lake" behaviour is an extrapolation: held-out depths are sparse above
6 m in mid-winter. This is a reason to favour the no-depth arm (B) unless the
data say otherwise. Source: `skill_diagnostics.py`.

## 4. Held-out performance

### Headline: against a per-month climatology

The fair benchmark is not a single constant but the training-mean thickness for
each calendar month. On 1,110 held-out observations:

| Comparison | RMSE | Difference |
|---|---:|---:|
| Model | 12.16 cm | — |
| Per-month training-mean baseline | 12.60 cm | −0.44 cm (model better) |
| Single training-mean baseline | 16.90 cm | −4.74 cm (model better) |

The paired lake-cluster interval for the difference against the per-month
baseline is **[−1.94, +0.65] cm**. It includes zero: **the model does not
establish skill over seasonal climatology on held-out lakes.** The 28%
improvement over the single-mean baseline is real but is mostly the seasonal
cycle, which the per-month baseline already knows.

### By month

| Month | n | Model RMSE | Model bias | Per-month baseline RMSE | Baseline bias | Model − baseline |
|---|---:|---:|---:|---:|---:|---:|
| November | 6 | 9.34 | +8.50 | 5.11 | +3.34 | +4.23 |
| December | 110 | 8.12 | +1.29 | 7.49 | −2.85 | +0.63 |
| January | 267 | 9.67 | +1.21 | 10.95 | −1.46 | −1.28 |
| February | 319 | 11.26 | +3.53 | 12.44 | −0.25 | −1.18 |
| March | 302 | 12.28 | +1.92 | 13.26 | −0.39 | −0.98 |
| April | 102 | 19.72 | −6.79 | 18.22 | +1.32 | +1.51 |
| May | 4 | 38.19 | −38.13 | 20.32 | −19.20 | +17.87 |

The model beats climatology in **January–March only** and loses in November,
December, April and May. November and December are early-season: the model
under-predicts by 8.5 and 1.3 cm. April–May are melt: the growth-only model
cannot thin, with a −6.8 cm April bias and −38 cm May bias.

Two caveats on the baseline: **May has no training observations**, so its
baseline is the pooled training mean (33.9 cm), not a May climatology; November
rests on only 24 training observations. Both months have negligible held-out
weight (6 and 4 observations). The January–April comparison is the meaningful
one.

This is the clearest statement of what v1 is: a mid-winter growth model. It is
not yet a full-season product.

### Anomaly skill: the model does track winter severity

The RMSE comparison above is a test of levels, and it is dominated by
between-lake bias. A different question: does the model get *departures* from
climatology right? Anomalies are formed by subtracting the training per-month
mean from both observed and predicted, then by subtracting each held-out
lake-month cell mean from both (which removes between-lake and seasonal
differences and leaves the within-cell variation).

| Variant | n | Correlation | Cluster 95% | Slope (obs on pred) | Cluster 95% | Skill vs zero anomaly |
|---|---:|---:|---:|---:|---:|---:|
| Month-demeaned | 1,110 | **+0.377** | +0.226 – +0.569 | **+0.593** | +0.374 – +0.978 | +0.065 |
| Within lake-month | 1,110 | **+0.407** | +0.343 – +0.464 | **+0.834** | +0.713 – +0.986 | +0.159 |
| January–March, month-demeaned | 888 | +0.472 | — | +0.793 | — | — |

Both cluster intervals exclude zero. The within-cell slope of 0.83 is close to
1, so once between-lake and seasonal differences are removed the model's
variation is of about the right amplitude. **The model has real but modest
anomaly skill** (r ≈ 0.4, about 16% of within-cell variance explained); what it
lacks is cross-lake level accuracy, and that is what the RMSE headline measures.
The two statements are compatible and both belong in the summary. Source:
`skill_diagnostics.py`.

### By region (V0)

| Region | n | RMSE | MAE | Bias | R² | Single-mean baseline |
|---|---:|---:|---:|---:|---:|---:|
| Lake Superior | 446 | 12.83 | 9.71 | −0.55 | 0.51 | 18.35 |
| Lake Michigan | 96 | 13.87 | 10.03 | +1.47 | 0.19 | 15.50 |
| Lake Huron | 97 | 14.12 | 11.53 | +9.23 | 0.25 | 18.07 |
| Michigan inland | 280 | 11.96 | 9.05 | +2.69 | 0.46 | 17.63 |
| NTL Wisconsin | 191 | 8.23 | 6.12 | −0.88 | 0.48 | 11.47 |

Michigan inland is the product headline (280 obs, 8 held-out lakes). Michigan
and Huron each rest on one held-out station.

### Lake-cluster intervals for the held-out statistics

2,000 replicates resampling the 15 held-out lakes with replacement, parameters
fixed at the point fit. This interval is conditional on the fitted parameters;
parameter uncertainty is the separate training-lake bootstrap in §3.

| Statistic | Point | Cluster 95% |
|---|---:|---:|
| RMSE | 12.16 cm | 10.80 – 13.12 |
| MAE | 9.11 cm | 7.91 – 10.09 |
| Bias | +1.24 cm | −2.83 – +5.52 |
| Depth-bias slope | +2.73 cm per ln(m) | **−2.01 – +6.63** |
| Snow slope | −0.14 cm per cm snow | **−0.17 – −0.07** |

## 5. Depth-bias gate

**The gate fails.** Residual against `ln(depth)` on the full held-out set:

| Sample | Slope | GN SE | Cluster 95% |
|---|---:|---:|---:|
| All held-out (1,110 obs, 15 lakes) | **+2.73** | 0.38 | −2.01 – +6.63 |

The point estimate says deep lakes are under-predicted relative to shallow
ones, and the pre-set gate was "no depth bias", so this is recorded as a
**failure**, not softened by the wide interval. The cluster interval including
zero means the evidence is not statistically decisive — but the gate was not
written as a significance test, and the point estimate is what was specified.

By region the slope is estimable only where depth varies:

| Region | n | Slope (GN SE) |
|---|---:|---:|
| Lake Superior | 446 | +8.14 (0.55) |
| Michigan inland | 280 | +1.82 (0.95) |
| NTL Wisconsin | 191 | +16.26 (6.63) |
| Lake Michigan | 96 | not estimable (one station) |
| Lake Huron | 97 | not estimable (one station) |

### Post-hoc subset (clearly labelled)

After seeing the per-station residual table, three stations were set aside:
**127 Portage Lake–Keweenaw Waterway, 147 East Nipigon Bay, 314 DeTour–
St. Marys River.** The selection rule was: *validation-only stations whose
reported depth lies in a dredged channel or river rather than a lake basin,
identified by station name and ice type.* Exactly these three, no others.

| Sample | n | Slope | Cluster 95% | RMSE | RMSE cluster 95% |
|---|---:|---:|---:|---:|---:|
| Excluding 127/147/314 | 759 (12 lakes) | +0.23 | −3.68 – +4.44 | 11.74 | 9.95 – 13.09 |

This is **post hoc** and is not used to set parameters or to re-define the
validation set. All three stations remain in every headline number above. They
are reported separately because they are flowing water: the model has no
current term, and their per-station biases (−9.8, −7.9, +7.6 cm) are the
largest in the set.

## 6. Does the depth term earn its place? (C1 = 0)

Fitted with `C1 = 0`, `k` and `C0` free, `C0 >= 0`:

| Fit | k | C0 | Held-out RMSE | MAE | Bias | R² | Depth slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 (C1 = 26.52) | 1.6049 | 0.0 | 12.16 | 9.11 | +1.24 | 0.479 | +2.73 |
| C1 = 0 | 1.5436 | 4.56 | **11.89** | 8.84 | +0.83 | 0.501 | +1.54 |

The no-depth model is **slightly better on held-out data**. The paired
lake-cluster difference `RMSE(V0) − RMSE(C1=0)` is **+0.27 cm, 95%
[+0.03, +0.51]** — the depth term costs about 0.3 cm of accuracy.

So the depth term is not supported by held-out accuracy; it is supported by
the physical expectation that deep lakes freeze later, and by the depth-bias
slope moving from +2.73 to +1.54. Neither model removes the positive slope.

**Decision status:** the approved configuration keeps V0. This section records
that the choice rests on physical plausibility, not on held-out skill.

### Where they differ: November–December and the display scale

| Sample | n | V0 RMSE | C1 = 0 RMSE | Difference | C1 = 0 better by |
|---|---:|---:|---:|---:|---:|
| All held-out | 1,110 | 12.16 | 11.89 | +0.27 | 2% |
| January–March | 888 | 11.18 | 11.09 | +0.09 | 1% |
| **November–December** | **116** | **8.19** | **6.74** | **+1.45** | **18%** |
| Both sides clamped at 20.3 cm | 1,110 | 3.23 | 2.83 | +0.41 | 12% |

The November–December count is small (12 lakes) but the paired cluster interval
for the difference is **+1.63 cm, 95% [+0.31, +3.22]** and excludes zero: the
depth term measurably hurts early-season prediction. Physically this is the
wrong sign for a depth effect — deep lakes should freeze *later*, i.e. thinner
early ice, and the model does that, but the observed early-season values are not
lower on deep lakes.

On the display scale, **82% of held-out observations exceed the 20.3 cm (8 in)
clamp**, so the both-clamped metric is compressed and mostly measures saturation,
not skill. Where both sides are clamped the models agree to ~1 cm MAE. What the
clamp does mean for the product: in mid-winter the map will be white almost
everywhere, so spatial structure will show at the ice edge and in early/late
season, not at peak. That is the specified design, noted once here.

Source: `skill_diagnostics.py`, `data/derived/skill_diagnostics.csv`.

## 7. Depth-convention variants

| Variant | Depth convention | k | C1 ± GN SE | Held-out RMSE | Common-subset RMSE |
|---|---|---:|---:|---:|---:|
| V0 | point depth | 1.605 | 26.5 ± 4.5 | 12.16 | **11.68** |
| V1 | 1.5 km median, 4 shallow stations | 1.618 | 30.8 ± 5.3 | 12.35 | 11.78 |
| V2 | drop the 4 shallow stations | 1.607 | 26.6 ± 4.5 | **11.68** | 11.68 |
| V3 | 1.5 km median, all Great Lakes | 1.620 | 32.1 ± 4.8 | 12.38 | 11.81 |
| V3a | 3 km median, all Great Lakes | 1.609 | 27.0 ± 4.7 | 12.27 | 11.70 |
| V3b | max(point, 1.5 km median), all GL | 1.623 | 32.0 ± 4.9 | 12.38 | 11.81 |
| V0u | V0, no C0 constraint (sensitivity) | 1.582 | 70.2 ± 6.4 | 12.39 | 12.05 |

On the like-for-like common subset (881 held-out observations, the three
shallow validation stations removed) the variants span 11.68–11.81 cm — a
0.13 cm difference, far inside the cluster uncertainty. The depth convention
is not a material lever. Keep V0.

V2 scores best on the full held-out set only because it deletes the three
shallow validation stations from scoring; its common-subset score equals V0.

## 8. Snow — coverage and diagnostic

Snow is excluded from v1. G00803 snow depth is now read from columns `[36:38]`
(diagnostic column only, see below). Residual against observed snow depth, by
source, with lake-cluster intervals:

| Group | n | Lakes | Slope | GN SE | Cluster 95% |
|---|---:|---:|---:|---:|---:|
| G00803 | 2,545 | 61 | **+0.150** | 0.029 | +0.019 – +0.282 |
| NTL | 972 | 11 | **−0.097** | 0.030 | −0.144 – −0.054 |
| ECCC | 20 | 2 | −0.208 | 0.242 | −0.469 – +0.188 |
| **Pooled** | **3,537** | **73** | **+0.086** | 0.022 | **−0.018 – +0.183** |

**The sign is source-dependent, and the pooled interval includes zero.** The
earlier negative pooled slope (−0.14 held-out) came entirely from NTL/ECCC;
adding the Great Lakes record flips the pooled estimate positive. A single
pooled snow coefficient is not justified, and the earlier "insulation sign is
robust" conclusion is withdrawn. No snow term is added to v1.

Coverage and missingness (fitting matrix, 4,766 observations; the 19 2026
observations are excluded):

| Group | Observations | With snow | % | Mean thickness (snow / no snow) | Mean residual (snow / no snow) |
|---|---:|---:|---:|---:|---:|
| G00803 | 3,754 | 2,545 | 67.8% | 39.4 / 35.6 cm | +2.4 / −2.4 cm |
| NTL | 991 | 972 | 98.1% | 40.1 / 37.4 | −0.7 / +2.4 |
| ECCC | 21 | 20 | 95.2% | 51.0 / 47.0 | +4.0 / +6.4 |
| All | 4,766 | 3,537 | 74.2% | 39.7 / 35.6 | +1.6 / −2.3 |

Snow recording is not missing at random: 83% of January and 87% of February
observations carry snow, but only 41% of April and 0% of May. Within G00803,
observations with snow recorded have a +2.4 cm residual versus −2.4 cm without,
so the slope can partly reflect *when* snow is recorded. 145 explicit zero-snow
observations are kept as 0 (all NTL); blanks stay NaN.

### G00803 snow-depth field: verified

The metric file `data/raw/g00803/extract/data2.dat` is fixed-width, 7,547
records, every line 79 characters. The Fortran reader shipped with the archive
(`display_ice2.f`, format 100) places **snow depth `SD` at 0-based columns
[36:38]** (`i2`), between water level `[33:35]` and snow condition `[39:42]`.

| Check | Result |
|---|---|
| Records with non-blank `[36:38]` | 3,758 / 7,547 (49.8%) |
| Cross-check vs English `data1.dat` × 2.54 | median absolute difference 0.30 cm over 6,627 matched records |
| Same check on total thickness | 0.28 cm (confirms alignment) |
| After wiring in, G00803 snow non-null | 2,545 / 3,754 observations |
| Direct-parse cross-check | all 2,545 values identical |

**Adding the column changed nothing else.** `observations.csv` and
`training_matrix.csv` match the backups in row order and in every column except
`snow_cm`; counts remain 4,785 / 74 / 3,671 / 1,114; `fit_results.csv` and
`validation_predictions.csv` are identical, so held-out RMSE is still 12.16 cm.
`sample_cfdd.py`, `build_pdd.py` and the parameter fit never read `snow_cm`;
`fit_model.py` and `bootstrap_model.py` read it only inside diagnostic blocks.
Verification log: `data/derived/diagnostics/snow_column_verification.txt`.

## 9. Melt term — design, pilot, feasibility

Design (amended 2026-11-08):

```
h = max(0, k * sqrt(max(0, S - C1 * ln d)) - m * P)
```

`P` is cumulative positive daily-mean degree-days from 1 March, defined only
inside the window **1 March – 31 July** and zero outside it (before March there
is no melt; after July the next ice season has not started). `m >= 0` in cm per
°C-day. No depth dependence in melt. This keeps three fitted constants.

### Two arms, selected on training data only

- **Arm A** (as written): fit `k`, `C1`, `m`; `C0 = 0`.
- **Arm B** (no depth term): fit `k`, `C0`, `m`; `C1 = 0`.

Selection is **repeated lake-grouped cross-validation on the 59 training
lakes, fixed seed, never held-out**. Arm B replaces Arm A only if its CV error
beats Arm A with a lake-cluster interval excluding zero. Both arms are then
reported on the frozen held-out set **for reporting only**; held-out data never
select the arm. **The arm A vs B CV output must carry November–December error as
a reported column**, because the depth term is already known to hurt that
window (V0 − C1 = 0 is +1.63 cm there) and the seasonal breakdown belongs in the
arm decision record.

### Pass criteria (amended)

1. **April** held-out RMSE beats the per-month baseline, with n reported.
2. **May is reported, not pass/fail** (4 held-out observations).
3. November–March held-out RMSE worsens by ≤0.5 cm.
4. `m` positive with a lake-cluster bootstrap interval excluding zero.
5. `P` spot-checked against GHCN March–May positive degree-days at Duluth,
   Marquette and Ann Arbor.

### 1966 pilot (done 2026-11-08)

`build_pdd.py` unpacks the existing CDS ZIP, computes daily means, and writes
`data/derived/pdd/pdd_1966.npz` on the exact CFDD grid (123 dates, 81 × 171,
uint16). P is zero on all 20 pre-March steps and all 52 post-July steps; the
maximum inside the window is 2,215 °C-day. Log:
`data/derived/diagnostics/pdd_pilot_1966.txt`.

GHCN March–May 1966:

| Station | GHCN P | ERA5 P | Diff |
|---|---:|---:|---:|
| Duluth MN | 339 | 302 | −10.9% |
| Marquette MI | 416 | 257 | −38.1% |
| Ann Arbor MI | 815 | 639 | −21.5% |

### Hourly trial (done 2026-11-08)

One sequential CDS request for `t2m` at all 24 hours, 1 March–31 July 1966:

| Measure | Value |
|---|---:|
| Wall time | **668 s (11.1 min)** |
| Compressed size | **94.9 MB** |
| Steps returned | 3,672 = 153 days × 24 hours |
| Grid | 81 × 171, exact CFDD grid |
| Linear 76-year estimate | **14.1 h, 7.2 GB** |

The request includes queueing, so 668 s is an upper bound on data transfer. A
4-sample request carries one sixth of the data (~16 MB/year, ~1.2 GB total) and
would likely take 2.5–4 h, but that is inferred, not measured.

### Three P definitions, 1966

Built from the hourly field on the CFDD grid, cumulative from 1 March:

- **H** — hourly-integrated: Σ over 24 h of max(0, T)/24.
- **P4** — mean of positive parts over the four synoptic hours.
- **M4** — positive part of the four-hour mean (the current CFDD-consistent
  method).

Cumulative P to 31 May 1966 vs GHCN daily `(TMAX+TMIN)/2`:

| Station | H | P4 | M4 | GHCN | Bias H | Bias P4 | Bias M4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Duluth MN | 340 | 331 | 302 | 339 | **+0.1%** | −2.5% | −10.9% |
| Marquette MI | 281 | 273 | 257 | 416 | −32.4% | −34.3% | −38.1% |
| Ann Arbor MI | 671 | 659 | 640 | 815 | −17.6% | −19.1% | −21.5% |
| **Mean abs bias** | | | | | **16.7** | 18.7 | 23.5 |
| **Spread** | | | | | 32.5 | 31.8 | **27.2** |

Choosing not to let cold nights cancel warm afternoons (H or P4) recovers about
10 points of Duluth bias and 2–4 points at the other two. The remaining 18–34%
deficit at Marquette and Ann Arbor is **not** a definition problem: Marquette is
a lakeshore site whose ERA5-Land cell is smoothed, and no candidate definition
moves it. H wins the pre-written first criterion (smallest mean absolute bias);
M4 has the smallest spread. Since the rule is bias first, **1966 favours H**, but
one year and three stations do not settle it.

Source: `compare_pdd_definitions.py`,
`data/derived/diagnostics/pdd_definition_comparison_1966.txt`.

### Rule applied over 1966–1970: H is selected

Five years, 15 year-station pairs, ranked by the rule fixed before any
comparison was seen. Mean absolute percent bias, with the per-station split:

| Definition | Mean abs bias | Duluth | Marquette | Ann Arbor | Mean per-year spread |
|---|---:|---:|---:|---:|---:|
| **H hourly** | **19.47** | **10.3** | **32.0** | **16.1** | 23.21 |
| P4 mean of positive parts | 21.09 | 12.1 | 33.6 | 17.6 | 23.06 |
| M4 positive part of mean | 24.16 | 16.5 | 36.4 | 19.6 | 21.66 |

H wins on the primary criterion. The margin over P4 is 1.62 points, and unlike
1966 it is **not** a Duluth artefact: H is better at all three stations
(Duluth +1.9, Marquette +1.5, Ann Arbor +1.4 points) and in **every one of the
five years** (per-year mean absolute bias 16.7/17.6/19.7/22.3/21.0 for H versus
18.6/19.2/21.4/23.8/22.5 for P4). M4 has the smallest spread, but the rule is
bias first, so the tie-break is not reached.

**Selected P definition: H, hourly-integrated** (Σ over 24 h of max(0, T)/24,
cumulative from 1 March, zero outside 1 March–31 July).

Two caveats that do not change the choice:

- Over 1966–1970 H is **19.5% low on average** against GHCN. The all-year figure
  over 1966–2025 is smaller (−13.1%); see the E2 section below, which also shows
  that the bias is not stationary.
- **Marquette is 32% low under every definition.** That offset is station/grid
  representativeness, not the daily-mean method, and it is not fixed here.

Source: `apply_pdd_rule.py`, `data/derived/pdd_definition_rule_summary.csv`,
`data/derived/diagnostics/pdd_rule_1966_1970.txt`.

### Pull decision (approved 2026-11-08)

**Hourly pull**, 1 March–31 July, all 24 hours, one request per year, sequential,
raw files kept. Order:

1. **1966–1970** first, so the definition rule can run early;
2. then **1971–2025**;
3. then **1950–1965**.

The pull does not pause for the rule. When 1966–1970 are on disk,
`apply_pdd_rule.py` runs automatically and writes the ranking; the pull
continues. Deriving the production P field for all years waits at the gate after
that table is reported.

Three points that must travel with the rule result:

1. **Marquette's bias does not depend on the definition.** It is 32–38% low
   under H, P4 and M4 alike, so it is a station/grid representativeness offset,
   not a daily-mean artefact. Any definition comparison that leans on Marquette
   is measuring the station, not the method.
2. **S and P may use different daily-mean definitions.** Nothing requires
   freezing degree-days and positive degree-days to be built the same way; S is
   frozen and P is not yet chosen.
3. **The 1966 ranking rested on Duluth alone.** H beat P4 by 2 points of mean
   absolute bias and both of those points came from Duluth; Marquette and Ann
   Arbor moved only 2–4 points between H and P4. The 1966–1970 table is the
   actual decision.

### Pull mechanics

- Resumable: a year whose raw file exists and is at least 50 MB is skipped, so a
  restart continues where it stopped. Status is written after every year.
- Retries: four attempts per year with backoff. A year that still fails is
  recorded and the pull continues; failures are listed in
  `data/derived/diagnostics/pdd_pull_unresolved.txt` for a decision.
- **No year is ever turned into `P = 0`**, silently or otherwise.
- Disk is checked before each year; the pull refuses below 20 GB free
  (needs ~7.2 GB raw).
- Every raw file is kept: `data/raw/era5/pdd_YYYY_hourly.zip`.

### Feasibility

1,354 training observations fall in March–May (58 lakes, 54 seasons), 277 of
them in April, so `m` is estimable. **May has zero training observations** (4
held-out), so melt beyond April is extrapolation and must be reported as such.

### E2 verification: production P, 1966–2025 (gate)

`verify_pdd_fields.py`, outputs `data/derived/diagnostics/pdd_verify_1966_2025.txt`,
`pdd_verify_summary.csv`, `pdd_ghcn_allyears.csv`, `pdd_join_check.csv`.

**File integrity.** All 60 years pass: 123 stored steps each, exact CFDD dates,
81×171 grid, `uint16`, P exactly zero on every stored step before 1 March and
after 31 July, and non-decreasing across the stored steps inside the window.
Annual maxima run **2,095 °C-day (1984) to 2,908 (2012)**, median 2,345.

**Pull health.** No failed years, no retries, nothing unresolved. 2025 completed
in 27 s after the server restart, which indicates a CDS cache hit, not a short
file: 100.1 MB, and its P field passes every check above.

**GHCN spot check, all 60 years** (reported only; the definition was fixed on
1966–1970 before this ran):

| Station | n | Mean bias | Median | Mean abs | Range |
|---|---:|---:|---:|---:|---:|
| Ann Arbor MI | 60 | −7.7% | −7.0% | 7.7% | −21.2 to −0.2% |
| Duluth MN | 60 | −9.9% | −8.9% | 10.1% | −27.0 to +4.5% |
| Marquette MI | 59 | −21.7% | −21.9% | 21.8% | −40.9 to +2.8% |
| **All** | 179 | **−13.1%** | | **13.2%** | −40.9 to +4.5% |

Marquette's 1979 record is absent from GHCN, hence 59 rows. Per-year mean bias
runs from −27.3% (1972) to −4.1% (2018).

The bias is **not stationary**; it shrinks over the record at two of the three
stations:

| Decade | Ann Arbor | Duluth | Marquette |
|---|---:|---:|---:|
| 1960s | −16.4% | −7.3% | −33.7% |
| 1980s | −6.4% | −8.2% | −21.1% |
| 2000s | −7.9% | −5.6% | −19.8% |
| 2020s | −2.1% | −11.1% | −9.8% |

**Join to the training matrix.** 4,766 observations, 74 lakes: 1,762 inside
March–July, 3,004 outside. Every outside-window observation gets **P = 0 exactly**
and every in-window observation has a finite, non-negative P. The joined S is
unchanged (max absolute difference from `training_matrix.csv` 2.3e-13, i.e.
float noise).

| Split | Month | n | P > 0 | Mean P | Median | Max |
|---|---|---:|---:|---:|---:|---:|
| train | Mar | 1,077 | 812 | 7.4 | 3.2 | 80.1 |
| train | Apr | 277 | 277 | 20.4 | 17.0 | 86.3 |
| train | May | 0 | 0 | – | – | – |
| val | Mar | 302 | 222 | 6.3 | 2.5 | 77.0 |
| val | Apr | 102 | 102 | 23.2 | 17.0 | 77.4 |
| val | May | 4 | 4 | 56.9 | 54.1 | 66.5 |

P = 0 outside the window means the melt term is identically zero in January,
February, November and December; those months remain pure growth. That is why
the November–December column matters in the arm comparison.

**One defect found and fixed at this gate.** The join originally interpolated
between stored 3-day steps, and for a February observation that reached across
1 March, giving P up to 2.0 °C-day outside the window. The sampler now returns
exactly 0 for any date before 1 March or after 31 July, and the table above is
from the fixed run.

**S and P use different daily-mean constructions.** S is the positive part of the
mean of the four synoptic hours; P is the mean of the positive part over all 24
hours. Nothing requires them to match, and they are not meant to.

**The melt coefficient.** `m` is calibrated to the same biased P that the map
displays, so the fit and the display share the offset rather than correcting it.
The real error source is the **station-to-station spread**, not the definition:
Marquette is ~22% low over all years and ~34% low in the 1960s while Ann Arbor
is ~8% low, a representativeness difference no daily-mean choice removes. Because
the bias also drifts through the record, a single `m` fitted on the pooled
training years absorbs the average of a moving bias, and April residuals from
early years should be expected to sit differently from late ones.

**Gate.** Verification only: no fit, no parameter change, no change to
`training_matrix.csv`. Awaiting approval to fit arms A and B.

### Melt fit: arms A and B (gate)

`fit_melt.py`, report `data/derived/diagnostics/melt_fit_report.txt`. Pool is the
frozen `training_matrix.csv` unchanged: 3,656 training rows / 59 lakes and 1,110
held-out rows; P joins with 1,417 rows positive, all inside the window.

Selection is 10 repeats × 5 folds lake-grouped CV on training lakes, seed
`20261106`:

| Arm | CV RMSE | SD | Nov–Dec | Jan–Mar | Apr | May |
|---|---:|---:|---:|---:|---:|---:|
| **A** (depth, `C0 = 0`) | **13.66** | 0.09 | 8.65 | 13.99 | 14.17 | – |
| B (no depth, `C1 = 0`) | 13.78 | 0.06 | 8.28 | 14.16 | 14.05 | – |

The training pool has no May observations, so the May CV cell is empty. The
paired lake-cluster difference (B − A) is **+0.12 cm, 95% [−0.09, +0.31]**. The
interval includes zero, so **arm B does not replace arm A**: the depth term stays
and `C0` stays pinned at 0.

Held-out is reported for both, never used to choose:

| Arm | k | depth par | m | Train RMSE | Held-out RMSE | MAE | Bias | R² | Nov–Dec | Jan–Mar | Apr (n) | May (n) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 1.6865 | C1 = 34.38 | 0.3318 | 13.42 | 11.23 | 8.69 | +1.37 | 0.56 | 8.73 | 10.90 | 15.24 (102) | 22.19 (4) |
| B | 1.6101 | C0 = 17.35 | 0.2998 | 13.56 | 10.95 | 8.37 | +1.02 | 0.58 | 7.13 | 10.74 | 15.06 (102) | 21.20 (4) |

**The held-out set prefers B by 0.28 cm while training CV prefers A by 0.12.**
That disagreement is recorded rather than resolved: the rule fixed in advance
selects on training CV, and held-out is reporting only. B is also clearly better
in November–December (7.13 vs 8.73), which is the window where the depth term was
already known to hurt.

**Pass criteria, chosen arm A:**

1. April held-out RMSE **15.24 cm** versus the per-month baseline **18.22 cm**
   (n = 102) — pass. The melt-free V0 was 19.72 cm on the same rows.
2. May held-out: n = 4, RMSE 22.19 cm — reported, not pass/fail. It is
   extrapolation: zero training observations in May.
3. November–March held-out RMSE **10.67 cm** versus V0 **10.87 cm**, a change of
   **−0.21 cm** — pass (limit was ≤ +0.5 cm).
4. `m = 0.3318 cm per °C-day`, lake-cluster 95% interval **[0.2082, 0.4744]**,
   excludes zero — pass.
5. P spot check: done at E2.

**Residuals by era, arm A** (cm):

| Set | Season | Era | n | RMSE | Bias |
|---|---|---|---:|---:|---:|
| train | winter | 1966–1979 | 2,615 | 14.51 | +0.68 |
| train | winter | 1982+ | 764 | 8.39 | −0.17 |
| train | spring | 1966–1979 | 220 | 14.78 | −3.96 |
| train | spring | 1982+ | 57 | 10.08 | +3.93 |
| held-out | winter | 1966–1979 | 825 | 11.26 | +2.17 |
| held-out | winter | 1982+ | 179 | 7.37 | −0.20 |
| held-out | spring | 1966–1979 | 94 | 16.07 | −2.93 |
| held-out | spring | 1982+ | 12 | 10.79 | +3.45 |

1980–1981 have no observations, so the era split is clean but the early era
carries most of the data (and most of it comes from a few repeatedly sampled
lakes). The early era is worse in **variance**, not bias.

The diagnostic refit with a separate `k` per era does not adopt itself: it
returns `k(1966–1979) = 1.6864` and `k(1982+) = 1.6866`, i.e. no era difference
in the growth scale, and identical training and held-out RMSE. The era gap is
therefore not a scale effect. Era and source are also collinear: the early era is
dominated by the long-running station series, the late era by the broader pool,
so a time split also measures a change in source composition.

**S against GHCN, October–March** (reported only; complete seasons, ≥150 of 182
GHCN days):

| Decade | Ann Arbor | Duluth | Marquette |
|---|---:|---:|---:|
| 1960s | +44.7% | −9.3% | +34.4% |
| 1970s | +17.0% | −11.4% | +27.0% |
| 1980s | +12.2% | −8.7% | +32.5% |
| 1990s | +11.4% | −6.9% | +37.5% |
| 2000s | +22.6% | −6.5% | +34.6% |
| 2010s | +11.3% | −6.1% | +12.4% |
| 2020s | −1.8% | −12.4% | −3.3% |

Complete seasons: mean **+11.1%**, mean absolute 17.5%, range −24.2 to +65.0%.
Two Marquette seasons (1978: 92 days, 2013: 107 days) are excluded because a
percentage on a partial season is meaningless; they are kept in
`melt_s_vs_ghcn_decade.csv`. Some of the positive bias is construction: S takes
the positive part of the mean of four hours, GHCN the daily mean, and sub-daily
variation makes the former larger near 0 °C. Unlike P, S is biased **high** at
Marquette and Ann Arbor and **low** at Duluth; the station spread is again the
real signal.

**Zero-thickness sensitivity.** There are **0 explicit zero-thickness records
from March to July** anywhere in the pool (minimum observed thickness 1 cm), so
the labelled sensitivity refit that includes them is identical to the main fit by
construction. The pool was not changed.

**Gate.** The arm decision was reported before anything was applied.

### v2 adopted (2026-11-09)

Arm A is adopted under the **pre-agreed asymmetric rule**: B replaces A only if
its lake-grouped CV error is lower with a cluster interval excluding zero. It is
not, so A stands. Depth is retained **on product grounds**: the map is per-lake
and depth varies inside a lake, and the held-out depth-bias test remains the
standing check on that term.

Confirmation numbers, `confirm_arm_a.py`:

| Check | Value |
|---|---|
| April held-out, arm A vs per-month baseline | 15.24 vs 18.22 cm, n = 102, 9 lakes |
| Paired difference (baseline − A) | **+2.97 cm, 95% [+1.55, +3.60]**, excludes zero |
| `m` | **0.3318**, 95% [0.2082, 0.4744] |
| CV overall, A vs B | 13.66 vs 13.78 cm; B − A **+0.12 [−0.09, +0.31]**, n = 3,656 |
| CV November–December, A vs B | 8.65 vs 8.28 cm; B − A **−0.37 [−1.55, +0.67]**, n = 288 |

Canonical v2 outputs are new files: `fit_results_v2.csv`,
`validation_predictions_v2.csv`, and `model_parameters.csv` (the single pointer
downstream reads). v1 is archived in `data/derived/backups/pre_v2_melt/`
(`fit_results.csv`, `validation_predictions.csv`, `fit_report.txt`,
`fit_model.py`). `fit_results.csv` and `validation_predictions.csv` stay in place
as the frozen v1/V0 comparison and are not overwritten.

**Count verification** (`fit_model_v2.py` → `v2_count_verification.txt`), all
checks pass:

| Check | Result |
|---|---|
| Observations | 4,785 rows, 74 lakes |
| Training matrix | 4,766 rows; 3,656 train / 1,110 held-out |
| Training rows vs v1 | 3,656 = 3,656 |
| Matrix keys vs pre-snow backup | identical, 4,766 |
| Matrix columns except `snow_cm` vs backup | 24 of 24 identical |
| Held-out keys vs v1 | 1,110 = 1,110 |

**Classification accuracy on held-out data**, lake-cluster 95% intervals
(`v2_classification.csv`):

| Threshold | Accuracy | 95% | Observed above | Predicted above | False above | Missed above |
|---|---:|---|---:|---:|---:|---:|
| 8 in (20.32 cm) | **0.893** | [0.851, 0.932] | 914 | 961 | 83 | 36 |
| 4 in (10.16 cm) | **0.947** | [0.920, 0.972] | 1,032 | 1,045 | 36 | 23 |

The model over-predicts the 8-inch class slightly (961 predicted vs 914
observed), which is the expected direction for a model whose cold-season bias is
+1.37 cm. Both thresholds are dominated by class imbalance: 82% of held-out rows
are above 8 inches, so a constant "above" rule would score 0.82 at 8 in and 0.93
at 4 in. The 8-inch accuracy is therefore the informative one.

### Raster gate checks (2026-11-09)

**v2 against per-month climatology.** `v2_vs_climatology.py`:

| | Value |
|---|---|
| Held-out n / lakes | 1,110 / 15 |
| v2 RMSE | **11.23 cm** |
| Per-month baseline RMSE | **12.60 cm** |
| Paired difference (v2 − baseline) | **−1.38 cm, 95% [−2.61, −0.66]** |

The interval excludes zero, so unlike v1 the melt model clears seasonal
climatology on levels. By month (delta = baseline − v2, positive favours v2):
Jan +1.01, Feb +1.48, Mar +1.65, Apr +2.97, May −1.88 (n = 4), Nov −4.23
(n = 6), Dec −1.20 (n = 110).

**Depth encoding: proposal, not applied.** `depth_encoding_proposal.py`. Current
scheme is uint8 with 1 m steps and a 255 m cap. Proposed:

```
v = clip(round(ln(d) * 40), 0, 255)     d in metres, floor 1 m
d_hat = exp(v / 40)
```

One code step is 2.53% in depth; the cap becomes **587 m**.

*The 255 m cap today.* Lake pixels are 3 arc-second cells; the lake mask is
elevation below the surface.

| Lake | Lake pixels | Max depth | Pixels > 255 m | % of lake | Area clipped |
|---|---:|---:|---:|---:|---:|
| Superior | 14,305,485 | 405.1 m | **1,098,097** | 7.68% | 6,354 km² |
| Michigan | 9,436,209 | 275.9 m | **77,859** | 0.83% | 483 km² |
| Huron | 9,848,352 | 224.3 m | 0 | 0% | 0 |

Under the log cap, 0 pixels are clipped in all three lakes.

*Quantization, in the growth argument.* One half step is `C1 × 0.0125 =
0.430 °C-day` at every depth, versus the current scheme's
`C1 × ln(d/(d−0.5))`, which is 13.9 °C-day at 1.2 m and 9.9 at 2 m. Thickness
error follows through `k / (2√A)`:

| True depth | 1 m ΔS | log ΔS | 1 m Δh, S=100 | log Δh, S=100 | 1 m Δh, S=50 | log Δh, S=50 |
|---|---:|---:|---:|---:|---:|---:|
| 1.2 m | 13.9 | 0.43 | 1.21 cm | 0.04 cm | 1.78 cm | 0.06 cm |
| 2 m | 9.9 | 0.43 | 0.96 | 0.04 | 1.63 | 0.07 |
| 3 m | 6.3 | 0.43 | 0.67 | 0.05 | 1.51 | 0.10 |
| 5 m | 3.6 | 0.43 | 0.46 | 0.05 | – | – |
| 10 m | 1.8 | 0.43 | 0.33 | 0.08 | – | – |

("–" is at or below onset, where ice is zero.) Near the onset the derivative is
unbounded but ice is clipped at zero; the largest a half step can add there is
`k√0.43 = 1.11 cm`, about half of that after the clip. The log scheme is
therefore roughly an order of magnitude better at small lakes and its error does
not grow as the lake shallows.

*The 1 m floor.* It collapses 0.31% of Superior's lake pixels, 0.46% of
Michigan's and 1.07% of Huron's; inland, 10 of 3,548 Michigan lakes (0.28%) have
a tabulated maximum depth below 1 m. Those are shallower than the model's
validated depth range (minimum held-out 0.9 m), so the floor is not a new
limitation.

**Status: proposal only.** No raster, encoder or product file has been changed.

### Raster output proposal (2026-11-09)

**Fitted-station depth by source.** `station_depth_by_source.py`, column `V0` of
`training_matrix.csv` (the point depth the v2 fit uses), observation-weighted:

| Source | n | Lakes | Min | p5 | p25 | Median | p75 | p95 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g00803 | 3,754 | 62 | 0.92 | 2.23 | 3.25 | 7.70 | 11.41 | 21.34 | 109.31 | 9.46 |
| ntl | 991 | 11 | 2.50 | 2.50 | 7.90 | 20.00 | 21.30 | 35.70 | 35.70 | 16.68 |
| eccc | 21 | 2 | 13.00 | 13.00 | 13.00 | 13.00 | 20.40 | 20.40 | 20.40 | 15.47 |
| **Pooled** | **4,766** | **74** | **0.92** | **2.23** | **3.31** | **8.30** | **12.34** | **29.36** | **109.31** | **10.99** |

Metres. g00803 carries the pool: its 109.3 m maximum is one deep Lake Superior
station, not a lake-wide value. ntl is the deepest source by median because its
lakes are large. eccc contributes 21 observations on 2 lakes and does not move
the pooled distribution. Within g00803 the regional medians are 9.08 m on
Superior, 3.05 m on Michigan and 3.90 m on Huron.

For the raster the per-pixel lake maximum is the bound that matters, not station
`V0`; the log cap (587 m) is well above the deepest NOAA pixel (405 m).

**Grid and mask.** Output is one lattice for every raster layer:

- EPSG:4269 (NAD83), 3 arc-second (0.0008333333°), north-up, pixel-is-area. This
  is the native NOAA bathymetry lattice, so the Great Lakes need no resampling.
- Depth: uint8, `v = clip(round(ln d * 40), 0, 255)`, floor 1 m, as approved.
- Mask: a separate layer, `0` = no lake, low bits = lake code, high bit = land
  collar. The flag is 128 for uint8 and 32768 for uint16, so the lake code is
  always recoverable and code `0` never means 1 m of water.
- Collar: two nearest-neighbour passes put the nearest lake code and its depth
  on land within two pixels of the shore. The flag marks those pixels: they are
  for texture filtering and edge sampling, not drawn as open water.

**Depth sources, as in the STATUS plan.**

- Great Lakes: NOAA NCEI 3 arc-second bathymetry, depth = −elevation, per pixel.
- Michigan inland: DNR deep point and contour integration in
  `mi_lake_depth.csv`, joined on `lake_idx` (the H2 join). The 1,989 lakes
  without a survey use the area-bin imputed maximum and a per-lake bowl. No
  join is by name, so the H2 failure mode cannot return.

*Great Lakes.* One combined mosaic of the three native grids, **9,457 × 15,025**
(142.1 M cells), **33,489,600 water pixels**; lake codes 1/2/3. The mosaic is
delivered as 2,048-pixel tiles, only where water exists.

*Michigan inland.* One grid over the DNR bounding box, **7,736 × 9,114**
(70.5 M cells), **493,108 water pixels** (0.70% of the grid); lake codes
`lake_idx + 1`, 1–3,548, matching the DNR geojson order used by
`build_depth.py`. The collar adds 339,814 flagged land pixels.

**Layer sizes** (`raster_layer_sizes.csv`; PNG, maximise compression):

| Layer | Grid | PNG | Notes |
|---|---|---:|---|
| Great Lakes depth | 9,457 × 15,025 | 5.18 MB | 4.74 MB as 26 non-empty 2,048 tiles |
| Great Lakes mask | 9,457 × 15,025 | 0.47 MB | 0.42 MB tiled |
| Inland mask | 7,736 × 9,114 | 0.61 MB | 0.52 MB as 15 non-empty tiles |
| Inland depth | 7,736 × 9,114 | 0.29 MB | flat per-lake **lower bound**; 15 tiles |

**Total download, depth + mask only: 5.98 MB** — Great Lakes 5.17 MB, inland
0.81 MB, across 82 PNG tiles plus a small georeferencing index. The inland depth
figure is a floor: contour and bowl detail add entropy, so budget **0.8–1.1 MB**
for inland and **6.0–6.3 MB** overall.

Serving the Great Lakes as three per-lake files instead costs 5.35 MB but leaves
the one-pixel Straits seam unresolved. The per-frame S and P fields are separate
and small (81 × 171 uint16, ~28 KB each); the depth and mask layers do not
repeat per frame.

**How Michigan and Huron join at the Straits.** All three NOAA tiles lie on the
same 3 arc-second lattice. Pixel offsets are

| From → To | Column | Row |
|---|---:|---:|
| Superior → Michigan | +5,040 | +4,092 |
| Superior → Huron | +9,240 | +3,600 |
| Michigan → Huron | +4,200 | −492 |

so the mosaic needs no resampling. Water appears in more than one tile on
**100,446** pixels:

| Pair | Pixels | Where |
|---|---:|---|
| Superior–Michigan | 34,908 | top rows of northern Lake Michigan |
| Superior–Huron | 65,202 | St Marys River / North Channel |
| Michigan–Huron | 336 | the one-pixel seam at lon −84.5004 |

The build rule must be **nearest lake body**, not tile order, so the
northern-Michigan water does not become Superior. Michigan's east edge
(−84.4996) and Huron's west edge (−84.5004) leave a 0.0008° gap, one pixel cell;
the collar pass fills it from the nearest valid water. S and P are shared
atmospheric fields, so no artificial seam is introduced; the only possible step
is the two source grids' depth difference at the boundary, taken from the valid
tile on each side. The lake code changes for labelling.

**Gate note: the bathymetry is not a lake mask.** Elevation below the surface
also includes the St Marys River / North Channel (the 65,202 overlapping pixels
above) and any other connecting channel. Scope is the three lakes. Two options:
(a) clip with a Great Lakes shoreline polygon, one small download, shoreline at
the NOAA 0 m contour; (b) exclude the connecting channels with a corridor mask.
Recommendation is (a). Not applied — this needs approval before raster
generation.

**Status at the time: proposal only.** The build that followed changed one
thing in this plan — the Great Lakes are clipped by a lake-body polygon rather
than by the NOAA 0 m contour — and is recorded next.

### Product build (2026-11-10)

Everything above was a plan. This is what was actually written.

**Shoreline gate.** `python/build_shoreline_mask.py` (exits 0, writes no
product raster) on the same 3″ EPSG:4269 lattice, 9,457 × 15,025.

- Lake bodies: **HydroLAKES v1.0** (CC-BY 4.0), Superior/Michigan/Huron kept
  separate. Huron includes Georgian Bay and the North Channel; the Straits of
  Mackinac are in. The ESI `mask_detail` layer is not used.
- Connecting river: **NOAA ESI `GL_St_Marys_River_2021` HYDROP** (CC0-1.0),
  538 water polygons unioned and dilated two cells, clears **60,952 Huron px
  (0.624%)**. The ESI area stops at the documented river mouth (46.05556 N) and
  does not cut the North Channel.
- Mask = lake-body polygon **∩** bathy water. The **164,117 px (973.1 km²)**
  that the polygon calls lake but the bathy has no depth for are **left out**
  (a one-pixel shoreline ring plus the Straits seam); no seam fill was added.
- **DNR inland polygons win** inside their boundary: 1,443 Great Lakes px
  cleared.
- Approved counts: **GL 33,015,374 px / 196,739.5 km²** (Superior 14,050,667;
  Michigan 9,307,694; Huron 9,658,456), **inland 493,108** water px. Report
  `data/derived/diagnostics/shoreline_mask_gate.txt`.

**Canonical product rasters.** `python/build_rasters.py` (exits 0, 49.6 s)
writes depth and mask PNG tiles, 2,048², 24 for the Great Lakes and 14 for the
inland layer. Encoding is the approved `v = clip(round(ln(max(d,1))*40), 0,
255)` with a separate mask (uint8 GL codes 1/2/3; uint16 inland `lake_idx + 1`
with high flag 32768 = collar). Two collar passes give each shoreline land
pixel the nearest lake code so the 3″ product antialiases; the collar is
flagged and never counted as water. Inland depth: 1,552 lakes from contour
interpolation, 1,996 from a per-lake bowl. `python/verify_rasters.py` (exits 0)
reads the PNGs back and passes every check: water counts equal the gate, 3,548
lake codes present, decoded depth ≥ 1 m on every water cell, St Marys mouth
clear.

**Animation fields.** `python/build_web_fields.py` (exits 0) writes one gzip
per season, 1 Oct → 30 Jun, 75 seasons 1950/51–2024/25, 92–93 frames each,
89.3 MB. S resets 1 Oct; stopping at 30 Jun avoids the August step where P
returns to zero with S still large. Header `ICEF1\0\0\0` + `uint32 n, h, w`
(20 bytes, so the uint16 payload is 4-byte aligned), then S frames then P
frames on the 81 × 171 grid.

**Web layers.** `python/build_web_layers.py` (exits 0) decimates to 6″
(~185 m) and ships **coverage + depth pairs**: `gl_cov/gl_dep` 7,513 × 4,729,
`inland_cov/inland_dep` 4,557 × 3,868, 2.3 MB total. Two channels rather than
one byte, because a single mipmapped byte averages land `0` with deep-water
`200` into a fake mid-depth and paints a bright halo on every shore. The shader
recovers `code = dep*255/cov`; mipmaps then average coverage honestly, which
antialiases the shore and fades a sub-pixel lake instead of stippling it. GL
blocks use a 2-of-4 majority, inland any-of-4. Out-of-scope water (Erie,
Ontario, St Clair, rivers, tile-edge neighbours) is 151,756 cells at code 255
and draws as flat dark water, never ice.

**Viewer and local server.** `web/index.html`, `web/style.css`, `web/app.js`
(WebGL2) evaluate the model per fragment from R16UI array textures, with the
0.1° field sampled bilinearly. `python/serve_web.py` serves the folder; the
browser cannot `fetch` from `file://`. Controls: season scrubber, speed,
play/pause, wheel zoom, drag pan, double-click reset, space and arrow keys.
The page opens on **1 February of the most recent season**, not on 1 October:
a season starts with no ice anywhere, so the first paint would be a dark,
apparently empty map. The ramp floor is also lifted slightly off the land
colour (0.043, 0.098, 0.196) so open water still reads as water in October and
in any lead, instead of disappearing into the basemap.

**Why the inland layer fades out.** A 10 ha lake is about 0.3 px at the
statewide view. Drawing its honest average is stipple, which reads as noise, so
inland coverage ramps in between about 1.8× and 3.2× zoom. This is a display
decision, recorded rather than hidden.

**Shader validation.** `python/validate_shader.py` compares 2,205 samples of
the real canvas at zoom 8 (mip 0) with a Python reference that re-derives the
colour from the same files. **452 flat samples match exactly, 0/255 on every
channel, including 8–11.6 cm of ice**; 1,753 samples sitting on depth contours
are within 6/255 (~2 cm), the expected half-texel effect. That covers depth
decode, field bilinear, constants, ramp and coverage blend end to end.




## 10. Limitations

1. **Skill over climatology is established but narrow and seasonally uneven.**
   v2 scores 11.23 cm against 12.60 cm for the per-month baseline (−1.38
   [−2.61, −0.66]); the gain comes from February–April, while November–December
   remain worse. The anomaly signal within lake-month cells (r = 0.41, slope
   0.83) is measured on v1 and is not a separate skill claim.
2. **Melt term adopted (v2), but the melt half is the weakest part.** Arm A
   cuts April held-out RMSE from 19.7 to 15.2 cm (n = 102) yet May remains
   22.2 cm on 4 observations and is extrapolation; there are no May training
   observations. `m` is calibrated to the biased ERA5 P, and the
   station-to-station P bias drifts with time.
3. **Depth term is weakly identified.** Cluster `C1` interval 5.9–51.6;
   `C1 = 0` is marginally better on held-out data and clearly better in
   November–December (+1.63 cm, 95% [+0.31, +3.22]).
4. **Station identity dominates.** Mackinaw City residuals reach +53 cm;
   Portage Waterway −47 cm. Between-lake differences not captured by depth are
   the largest single error source.
5. **Thin validation depth coverage.** Held-out depths span 0.9–25.3 m (median
   3.7); training spans 1.2–109.3 m. The depth-bias test is partly
   extrapolation.
6. **Two Great Lakes validation sets are single-station** (Michigan, Huron).
7. **Error is ±12 cm**, about 60% of the 0–20 cm display range.

## 11. Decisions recorded

1. `C0 >= 0` adopted. Unconstrained `C0 = -111` predicts ice at `S = 0` and is
   worse on held-out data.
2. Depth variant V0 kept; all variants within 0.13 cm on the common subset.
3. The three flowing-water stations are **reported separately, not deleted**,
   and only as a post-hoc line. The headline depth-bias result is the failure
   on the full held-out set.
4. Melt term approved in principle: three constants, two arms (A: k/C1/m;
   B: k/C0/m), selected by repeated lake-grouped CV on training lakes only.
   Implementation waits on the P pull.
5. Display rule: `h = 0 where S = 0`, which removes the sub-1 m artefact
   (East Nipigon, 0.92 m) where `ln(d) < 0`.

## 12. Outputs

| File | Contents |
|---|---|
| `data/derived/training_matrix.csv` | 4,766 observations with CFDD, season, V0–V3b depths |
| `data/derived/fit_results.csv` | parameters and metrics for V0–V3b, V0u |
| `data/derived/validation_predictions.csv` | held-out V0 predictions and residuals |
| `data/derived/bootstrap_intervals.csv` | all cluster intervals above |
| `data/derived/skill_diagnostics.csv` | anomaly skill, low-CFDD lever, V0 vs C1 = 0 |
| `data/derived/diagnostics/fit_report.txt` | full fit and diagnostic report |
| `data/derived/diagnostics/bootstrap_report.txt` | bootstrap, C1 = 0, per-month baseline |
| `data/derived/diagnostics/skill_diagnostics.txt` | anomaly, lever and clamp diagnostics |
| `data/derived/pdd_trial/pdd_1966_H.npz` | 1966 P, hourly-integrated |
| `data/derived/pdd_trial/pdd_1966_P4.npz` | 1966 P, mean of positive 4-sample |
| `data/derived/pdd_trial/pdd_1966_M4.npz` | 1966 P, positive part of 4-sample mean |
| `data/derived/pdd_definition_comparison_1966.csv` | three-definition GHCN comparison |
| `data/derived/diagnostics/cfdd_missing_2026.csv` | 19 flagged 2026 observations |
| `data/derived/model_parameters.csv` | canonical v2 constants and held-out metrics |
| `data/derived/rasters/` | 3″ depth + mask PNG tiles, GL and inland, plus `manifest.json` |
| `data/derived/diagnostics/shoreline_mask_gate.txt` | approved shoreline gate counts |
| `data/derived/diagnostics/raster_verification.txt` | read-back checks on the 3″ product |
| `data/derived/diagnostics/shader_validation.txt` | GPU vs Python reference samples |
| `web/fields/season_YYYY.bin.gz` | per-season S and P frames, 1950/51–2024/25 |
| `web/layers/*_cov.png`, `*_dep.png` | 6″ mipmapped coverage and depth textures |
| `web/index.html`, `web/style.css`, `web/app.js` | the viewer |
| `python/serve_web.py` | local static server for the viewer |
