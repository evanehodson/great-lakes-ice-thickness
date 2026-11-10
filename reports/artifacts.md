# Cross-source artifact audit

**Date:** 2026-11-07
**Revision:** M2 (coarse coordinates) was withdrawn on 2026-11-07 — see §5.
**Scope:** the 5,197 lake-date observations in `data/derived/observations.csv`
and the depth/coordinate join behind them.
**Question asked:** are there glaring differences in standards between G00803,
NTL and ECCC that would poison the fit?

**Short answer:** the thickness values themselves are consistent. The real
artifacts are in the **depth covariate** and in **three lake-identity matches**,
not in the ice measurements. Roughly **9.5% of observations (492)** carry a
depth or identity problem serious enough to change the fit. The thickness
values from the three sources agree on units, rounding and black+white
accounting.

---

## 1. Findings, ranked by impact

| # | Finding | Stations | Obs | Share | Action |
|---|---|---|---|---|---|
| H1 | Harbour depths are a shoreline-edge artifact | 11 | 295 | 5.7% | Drop from fit (recommended) |
| H2 | Duplicate lake names pick the wrong polygon's depth | 5 | 140 | 2.7% | Fix join to use polygon id |
| M1 | Three stations matched to the wrong lake | 3 | 74 | 1.4% | Drop, or verify against gazetteer |
| M3 | Spatial support differs (point vs lake mean) | — | 22 ECCC | 0.4% | Interpretation only; report by source |
| L1 | White-ice share differs by source | — | — | — | Snow diagnostic only |
| L2 | Seasonal sampling is unbalanced | — | — | — | Report validation by month |
| L3 | G00803 and NTL eras do not overlap | — | — | — | Check residual by source/era |
| L4 | `total != black + white` on 7 rows | 7 rows | 7 | 0.1% | Note only; total is authoritative |

Definite depth/identity problems (H1, H2, M1, de-duplicated): **18 lakes,
492 observations (9.5%) — 405 train, 87 validation.** See §13 for what each
drop costs.

---

## 2. H1 — Harbour depths are a shoreline-edge artifact

**What the pipeline does.** For Great Lakes stations, depth is the NOAA 3
arc-second raster sampled at the station coordinate, falling back to the
nearest water pixel if the coordinate lands on shore
(`sample_gl_depth` in `build_observations.py`).

**Why it fails.** A harbour coordinate almost always lands on the shoreline.
The nearest water pixel is therefore the shallowest edge pixel, not the water
where the ice was measured.

| Station | Obs | Depth used | 3–4 km median | Ratio |
|---|---:|---:|---:|---:|
| Menominee MI | 108 | 0.11 m | 6.0 m | 0.02 |
| Isle Royale–Washington Harbor | 70 | 0.24 m | 40.0 m | 0.01 |
| Sturgeon Bay | 28 | 0.11 m | 0.4 m | 0.27 |
| Agate Bay–Two Harbors MN | 22 | 2.27 m | 60.3 m | 0.04 |
| Bayfield Harbor | 18 | 0.05 m | 40.3 m | 0.00 |
| Grand Marais Harbor MN | 15 | 0.25 m | 110.3 m | 0.00 |
| Grand Marais Harbor MI | 13 | 1.16 m | 10.6 m | 0.11 |
| Two Harbors #1 Dock | 7 | 2.27 m | 60.3 m | 0.04 |
| Copper Harbor | 6 | 2.22 m | 55.8 m | 0.04 |
| Two Harbors #1 Dock–New Site | 5 | 2.27 m | 60.3 m | 0.04 |
| Whitefish Point Harbor | 3 | 0.13 m | 33.7 m | 0.00 |

**267 of 295 harbour observations (91%) use a depth below 20% of the local
median.** The true dock depth is somewhere between the two numbers and is not
recoverable from the raster. Note that the 3–4 km median is not a substitute
either — for Grand Marais MN it is 110 m, which is the open lake, not the
harbour.

**Why it matters.** Harbours are almost the only sub-2 m depths in the pool.
If `C1` is identified from them, it will be inflated, and when that `C1` is
applied to the map's shoreline pixels it will paint a thick-ice ring along
every shore. The depth lever is weak at peak season (≈0.5 cm for a 1.5→6 m
change) but strong at the shallow end: with `C1 = 40`, 0.1 m vs 10 m is a
factor of ~1.6–2.0 in thickness early in the season, when `S` is 150–200 °C·day.

**Recommendation.** Drop the 11 harbour stations from the fit (295 obs). Their
depth is unknowable from the raster, and sheltered dock ice is a different
physical regime from open-lake ice. The map can still render harbours from
depth + CFDD; it just will not be tuned to them. If you would rather keep them,
the only defensible alternative is a fixed nominal harbour depth with a flag,
which is patching.

---

## 3. H2 — Duplicate lake names pick the wrong polygon's depth

**What the pipeline does.**
`build_observations.py` builds the inland depth lookup with

```python
dmap = depths.drop_duplicates("name").set_index("name")[["max_depth_m", "mean_depth_m"]]
```

The DNR table has **371 duplicated names across 1,476 rows** (51 Mud Lakes, 49
Long Lakes, 30 Round Lakes, 30 Bass Lakes, 23 Clear Lakes, …). `drop_duplicates`
keeps the first row, which is an arbitrary polygon, often a small lake in a
different county.

**Damage.** 7 of the 21 Michigan inland stations matched a duplicated name;
5 have a depth error above 0.5 m:

| Station | Depth used | Nearest same-name polygon | Error | Obs |
|---|---:|---:|---:|---:|
| Crystal Lake (sid 217) | 16.76 m | 53.34 m (39.9 km²) | −36.6 m | 28 |
| Indian Lake (sid 211) | 7.62 m | 19.96 m (35.0 km²) | −12.3 m | 34 |
| Bear Lake (sid 126) | 10.36 m | 1.22 m (0.40 km²) | +9.1 m | 33 |
| Twin Lakes (sid 313) | 7.62 m | 12.19 m (1.78 km²) | −4.6 m | 28 |
| Big Island Lake (sid 210) | 8.23 m | 9.14 m (0.70 km²) | −0.9 m | 17 |

The point is the **polygon identity**, not that every replacement depth is a
survey: Indian Lake's 19.96 m is itself an imputed area-bin median, but it is
the depth of the 35 km² lake the station actually sits on (distance 0.00 km),
whereas 7.62 m belonged to a different, small Indian Lake.

**140 observations (2.7%)** are affected. The map's
`mi_lake_depth.csv` is **not** affected — it is stored per polygon. Only the
station join is wrong.

**Recommendation.** Store the matched polygon's `lake_idx` in
`training_stations.csv` at match time and join depth on `lake_idx`, never on
name. This is a real bug, not a dataset-standard difference.

---

## 4. M1 — Three stations matched to the wrong lake

The station→lake match is spatial (nearest DNR polygon within 5 km), with the
name secondary. Three stations land on a lake whose name does not match, and a
same-named lake exists nearby:

| Station | Matched | Distance | Same-name lake | Distance |
|---|---|---|---:|---|---:|
| Pine Lake (sid 122) | Second Lake | 0.29 km | Pine Lake | 1.12 km |
| Gibson Lake (sid 204) | Lake Ellen | 1.07 km | Gibson Lake | 30.0 km |
| Big Island Lake (sid 210) | Triangle Lake | 1.22 km | Big Island Lake | 7.13 km |

**74 observations (1.4%).** These are not naming variants — the coordinate
and the name disagree, and nothing in the source resolves which is right.

The other five name mismatches are legitimate local-name variants and should
be kept: `Warners Lake → Andrus Lake` (no Warner Lake within 25 km),
`Beaver Island–Lake Barney → Barneys Lake`,
`S. Manitou Is.–Lake Florence → Florence Lake`,
`Muskegon Lake–Snug Harbor → Muskegon Lake`,
`Bois Blanc Island–Twin Lake → Twin Lakes`.

**Recommendation.** Drop the three ambiguous stations (74 obs), consistent
with the project rule to drop rather than patch.

---

## 5. M2 — Coordinate precision: withdrawn

**This finding was an artifact of the degrees-to-decimal conversion, not a
property of the data.** `ice.loc` records every station to the nearest minute
(`N47-57/W084-52`) and no station has seconds, so a minute that is a multiple
of 6 converts to one decimal place (06′ = 0.1°), a multiple of 3 to two places,
and every other value to a repeating decimal (46′ = 0.7667°). Counting decimal
digits after conversion therefore measured how the minutes divide into 60, not
how precisely the station was located — Duluth Harbor, at 92°06′, converts to
92.1000 and was classified "1-decimal".

**Actual precision is uniform at ±0.5 arc-minute, about ±0.9 km north–south and
±0.65 km east–west at 46 °N.** That is under a sixth of an ERA5-Land cell in
either direction. Even if the minutes had been truncated rather than rounded
(±1 arc-minute), the error would still be under a third of a cell. There is no
coordinate-precision problem in this dataset.

The stations whose sampled depth is far below their neighbourhood median
(East Nipigon, Little Traverse, Mackinaw City, DeTour) are a **depth-convention**
question, not a coordinate one: all four coordinates fall on water and are
sampled at the exact pixel. They are tested separately as variants in the
shallow-bay check.

---

## 6. M3 — Spatial support differs between sources

| Source | Support | Obs |
|---|---|---:|
| G00803 | single drill hole at a station | 4,166 |
| NTL | single station per lake | 1,009 |
| ECCC | 12–16 probes per date, averaged | 22 |

ECCC is a **lake-mean**; G00803 and NTL are **point** measurements. Mixing them
means the target is not the same quantity.

Crystal Lake illustrates it: NTL and ECCC sample different dates, and year
means differ non-systematically:

| Year | NTL mean | ECCC mean | Diff | NTL max | ECCC max | Max diff |
|---|---:|---:|---:|---:|---:|---:|
| 2010 | 27.0 | 37.5 | +10.5 | 39.0 | 41.0 | +2.0 |
| 2011 | 42.0 | 51.7 | +9.7 | 51.0 | 54.4 | +3.4 |
| 2012 | 35.5 | 22.5 | −13.0 | 41.0 | 28.5 | −12.5 |
| 2013 | 41.7 | 37.2 | −4.5 | 61.0 | 45.5 | −15.5 |
| 2014 | 56.7 | 62.3 | +5.6 | 73.0 | 70.0 | −3.0 |
| 2017 | 35.7 | 38.3 | +2.6 | 40.0 | 42.0 | +2.0 |

No date overlap, so this is **not** a bias and **not** double-counting — it is
within-lake spatial variability plus different sampling days. It is the same
signal as the within-date ECCC spread (`sd_cm`, median 4.1 cm, max 17.7 cm).

**Recommendation.** No data change. Keep ECCC as a lake-mean target, report
validation separately for point and lake-mean sources, and use `sd_cm` as the
measurement-error floor.

---

## 7. L1 — White-ice share differs by source

| Source | Mean white share | Median | IQR |
|---|---:|---:|---|
| ECCC | 0.70 | 0.74 | 0.54–0.92 |
| NTL | 0.32 | 0.27 | 0.10–0.49 |
| G00803 | 0.22 | 0.16 | 0.00–0.38 |

Mostly season, not protocol: ECCC has 15 of 22 dates in March (late, snow-ice
dominant), G00803 has 400 December dates (fresh black ice). The white share is
**not used in v1** and only matters for the snow residual diagnostic.

One caveat: ECCC black/white coverage is partial on 3 of 22 dates (Crystal
2012-03-17 has 2 of 29 probes; two Little Turkey dates have none). Because the
collapse averages total over all probes but components over the non-null
subset, the ECCC white share is computed on a partial sample. That is fine for
a descriptive table, but do not use it as a model input.

**Recommendation.** Regress residuals on **observed snow depth**, not on white
fraction, and report that diagnostic by source.

---

## 8. L2 — Seasonal sampling is unbalanced

| Source | Dec | Jan | Feb | Mar | Apr | May |
|---|---:|---:|---:|---:|---:|---:|
| G00803 | 400 | 949 | 1,205 | 1,256 | 328 | 4 |
| NTL | 1 | 335 | 353 | 245 | 69 | 0 |
| ECCC | 0 | 0 | 6 | 15 | 1 | 0 |

Early season is effectively G00803-only; late season is G00803 + NTL. This is
fine for fitting but means validation must be **stratified by month**, or the
March-heavy ECCC contribution will look better than the model's December skill.

The 4 May G00803 observations are all Lake Superior (1972-05-05, 1975-05-02,
1968-05-01, 1968-05-08) and are plausible late ice. Keep.

---

## 9. L3 — The two eras do not overlap

G00803 covers **1966–1979**; NTL starts in **1982**; ECCC is **2010–2018**.
There is no lake-date measured by both G00803 and NTL/ECCC, so a change in
measurement protocol between eras cannot be separated from a change in climate.

**Recommendation.** After fitting, check residuals by source and era. If the
G00803 residuals have a different mean from NTL/ECCC, report it; only add a
source offset if it is large and stable, and flag it as a protocol correction.

---

## 10. L4 — `total != black + white` on 7 rows

Only 7 rows differ by more than 2 cm (max 10 cm):

| Source | Lake | Date | Total | Black | White | Diff |
|---|---|---|---:|---:|---:|---:|
| G00803 | Chequamegon Bay–Ashland WI | 1978-04-02 | 40 | 17 | 13 | +10 |
| NTL | Big Muskellunge Lake | 2019-03-25 | 70 | 55 | 25 | −10 |
| NTL | Allequash Lake | 2022-02-28 | 48 | 29 | 13 | +6 |
| ECCC | Crystal Lake | 2012-03-17 | 22.5 | 15.4 | 12.1 | −5.0 |
| G00803 | Mackinaw City | 1978-03-18 | 55 | 41 | 9 | +5 |
| G00803 | South Bay–Munising | 1978-03-17 | 39 | 27 | 8 | +4 |
| G00803 | St. Martin Bay | 1978-03-11 | 47 | 40 | 4 | +3 |

These are isolated entry or coverage issues, not a protocol change:

- G00803 is exact on **4,162 of 4,166 rows**; all four exceptions are in 1978.
- The ECCC Crystal 2012-03-17 row is a **coverage artifact**, not a bad probe:
  all 29 probes report total, but only 2 report black/white. The collapse averages
  `total_cm` over 29 and `white_cm`/`black_cm` over 2, so the components are not
  comparable. Total thickness is unaffected.
- The ECCC Little Turkey 2026-04-01 −1.5 cm is one bad probe out of 12 (probe 7
  has white+black = 0.80 m against a total of 0.62 m).
- The NTL rows have a plausible total but inconsistent components
  (Big Muskellunge: 55 + 25 = 80 vs total 70).

The model uses `total_cm`, so there is no impact. Note only.

---

## 11. What is *not* a problem

These were checked and cleared:

- **Two Harbors trio is not triple-counting.** `AGATE BAY-TWO HARBORS MN`,
  `TWO HARBORS-#1 DOCK` and `-NEW SITE` share a coordinate but have different
  values on every shared date, and the dock is consistently thicker than the
  open bay (e.g. 1969-02-19: 33 / 41 / 41 cm). They are distinct probes.
- **Lake Fanny Hooe and Copper Harbor share a coordinate but are different
  waterbodies.** Fanny Hooe (inland, 9.1 m) is consistently thicker than the
  harbour (e.g. 1968-03-02: 63 vs 48 cm). The shared coordinate is a town
  geocode; it does not double-count.
- **Gros Cap Light and Point Iroquois** are 2.25 km apart with 86 shared dates
  (means 39.1 vs 36.0 cm). Distinct stations on opposite sides of the St. Marys
  River; legitimate spatial sampling.
- **The Crystal Lake merge is correct.** ECCC Crystal is at 46.0033 / −89.6157;
  NTL `CR` is at 46.003 / −89.612 — the same Wisconsin lake. The Michigan
  G00803 Crystal Lake (44.70 / −86.23) is a different lake and remains separate.
- **Digit preference is resolution, not error.** G00803 is 100% integer cm,
  NTL 92.5%, ECCC 0% (metres converted to cm). The ~11% multiples of 10 in
  G00803 and NTL are normal rounding.
- **No zero/trace artifact.** Values ≤2 cm are 0.1% of both G00803 and NTL.
- **No impossible values.** No depth ≤0, no total ≤0, no white > total, no black
  > total, no duplicate lake-dates after the multi-point collapse.
- **Thickness units agree.** G00803 and NTL are cm; ECCC is m ×100. Total
  thickness distributions are consistent (G00803 mean 38.4, NTL 40.1, ECCC
  51.5 — ECCC is late-season, so the higher mean is expected).

---

## 12. Recommended sequence before the fit

1. **Fix H2** in `build_observations.py`: join inland depth on `lake_idx`,
   not on name. Regenerate `training_stations.csv` with the polygon id.
2. **Drop M1** (Pine, Gibson, Big Island) — 74 obs.
3. **Decide H1**: drop the 11 harbour stations (recommended), or assign a
   flagged nominal depth.
4. Re-run `build_observations.py`, then confirm the split still has no lake on
   both sides and the validation share is still ~20–25%.
5. Fit with and without the affected records as a sensitivity check, and
   compare `C1` and the shoreline ring.

No change is proposed to `mi_lake_depth.csv` or the per-pixel depth rasters:
those are stored per polygon and are correct.

---

## 13. Impact of the proposed drops

| Scenario | Obs | Lakes | Train | Val | Val share | Notes |
|---|---:|---:|---:|---:|---:|---|
| Current | 5,197 | 89 | 3,953 | 1,244 | 23.9% | |
| Drop harbours | 4,902 | 78 | 3,728 | 1,174 | 23.9% | |
| **Drop harbours + wrong-lake (recommended)** | **4,828** | **75** | **3,671** | **1,157** | **24.0%** | Every multi-lake region keeps a validation set |
| Drop harbours + wrong-lake + 2 bad-depth | 4,689 | 73 | 3,568 | 1,121 | 23.9% | Superior loses East Nipigon (36 val) |

The duplicate-name fix (H2) corrects depth but **does not drop observations**.

The recommended pool is **4,828 observations (92.9% retained) across 75 lakes,
with 1,157 held-out observations (24.0%)** and every multi-lake region
represented in validation. For a 3-parameter model that is ample: roughly 1,600
observations per parameter, and the binding constraint is the number of
independent lakes (59 training, 16 validation), not the raw count.

---

## 14. Reproduce

`python/audit_artifacts.py` reproduces every table above:

```
$env:PYTHONIOENCODING="utf-8"
python python/audit_artifacts.py
```

It reads `observations.csv`, `training_stations.csv`, `mi_lake_depth.csv`,
`lakes_10ha.geojson` and the three NOAA GeoTIFFs, and prints the H1, H2, M1 and
L4 checks. Key totals: 5,197 lake-dates; 295 harbour obs; 140 duplicate-name
obs; 74 wrong-lake obs; 267/295 harbour depth ratios <0.2.
