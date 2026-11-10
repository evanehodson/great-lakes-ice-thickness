# Data audit — upper Great Lakes ice thickness model

**Date:** 2026-11-05
**Scope:** Lake Superior, Michigan, Huron, plus Michigan inland lakes ≥10 ha in the Great Lakes watershed.
**Purpose:** Establish exactly what each source contains, what it can and cannot validate, and where the gaps are — before any fitting or map work.

**Status:** all four observation sources and the temperature source are on disk and parsed. No source remains blocked.

---

## 1. Summary

| # | Source | Role | Volume on disk | Spatial coverage | Temporal coverage |
|---|---|---|---|---|---|
| 1 | GLERL G00803 ice thickness | Validation + calibration, Great Lakes | 1.2 MB | 121 georeferenced stations, 41.5–49.5°N | 1966–1979 |
| 2 | ECCC Lake Ice Database | Cold-range calibration, 2 regional sites | 710 KB | 21 sites, 46.0–82.5°N | 1963–2026 |
| 3 | NTL-LTER snow & ice depth | Temperate small-lake calibration | 50 KB | 11 Wisconsin lakes | 1982–2026 |
| 4 | Michigan DNR hydro + bathymetry | Lake polygons, areas, depth | (live service) | Michigan | current |
| 5 | ERA5-Land daily statistics | Air temperature → FDD | (to fetch) | 9 km grid, full basin | 1950–present |

Total independent ice-thickness observations available for fitting and validation: **~10,000 point measurements**, of which **~4,700 are on the three target lakes**.

---

## 2. GLERL Great Lakes Ice Thickness Data Base (G00803)

- **Citation:** Sleator, F. E. (1995). *GLERL Great Lakes Ice Thickness Data Base, 1966–1979* (G00803, Version 1). NSIDC. doi:10.7265/N5KW5CXG
- **Files:** `data/raw/g00803/extract/data2.dat` (metric), `ice.loc` (station index), `ice.txt` (format documentation), `display_ice2.f` (reader source).
- **Licence:** NOAA / US Government, public domain. Citation requested.

### Contents

| Property | Value |
|---|---|
| Records | 7,547 |
| Station IDs | 130 |
| Stations with coordinates in `ice.loc` | 121 |
| Years present | 1966–1979 |
| Months present | Nov–May (peak Feb–Mar) |
| Records with total thickness | 5,837 (77%) |
| Explicit zero-thickness records | 19 |
| Records flagged estimated (`E`) | 231 |
| Total thickness, median / p95 | 37 cm / 69 cm |

Blank thickness fields mean **not recorded**, not zero — confirmed by the presence of explicit `0` values. This matters for freeze-up validation.

**Documentation discrepancy:** NSIDC's landing page states coverage 1 Jan 1965 – 1 May 1977 and "up to 90 stations per year". The metric file contains records through March 1979 (1965/66 season is labelled 1966, which explains the start). The 1977–79 records look continuous and plausible (30 stations/year), so they are retained, but this extension is undocumented and should be treated as slightly less authoritative.

### In-scope subset (station bands 100–399)

Station numbering follows basin: 100s = Superior and its basin, 200s = Michigan, 300s = Huron/Georgian Bay/St. Marys, 400s/500s = Erie, St. Clair and Ontario (out of scope).

| Band | Stations | Rows | With thickness | Median | p05 | p95 |
|---|---|---|---|---|---|---|
| Superior basin | 71 | 3,371 | 2,675 | 40 cm | 6 | 74 |
| Michigan basin | 22 | 1,094 | 893 | 38 cm | 7 | 66 |
| Huron basin | 15 | 1,665 | 1,177 | 41 cm | 8 | 71 |
| **In scope** | **108** | **6,130** | **4,745** | **40 cm** | — | — |
| Erie/Ontario (excluded) | 22 | 1,417 | 1,092 | 25 cm | 3 | 56 |

### Composition (snow ice vs. black ice)

Of 4,726 in-scope observations with ice present, 66% carry a non-zero snow-ice component. Snow ice averages **24% of total thickness**, rising through the season: 9% in January → 19% in March → 26% in April.

### Data quality

- **One clear outlier:** Batchawana Bay, station 104, 1968-03-02, recorded at 201 cm. Adjacent weeks at the same station are 97 and 99 cm; the next-highest value anywhere in 14 years is 99 cm. Almost certainly a keying error. **Excluded.**
- **Nine station IDs have no coordinate entry:** 110, 208, 221, 311, 506, 507, 508, 509, 510. These records can still be used for Great Lakes aggregates but not for spatial fitting.
- **Critical limitation:** every station is a harbour, bay, shoreline point, or basin inland lake. There is **no offshore deep-water sampling**. Validation therefore covers the coastal and shallow regime only; mid-lake behaviour on Superior's 400 m basins cannot be checked directly against this dataset.

---

## 3. ECCC Lake Ice Database

- **Source:** `data-donnees.az.ec.gc.ca`, `water/scientificknowledge/lake-ice-database`
- **Licence:** Open Government Licence – Canada.
- **Files:** `ECCC_LI-TC_v6.csv` (2,870 rows), `External_LI-TC_v2.csv` (4,784 rows), plus metadata and readme. Latin-1 encoded; header row located by the cell `Site Name`.

| Property | Value |
|---|---|
| Combined rows | 7,654 |
| Distinct sites | 34 (21 with coordinates) |
| Latitude range | 46.0 – 82.5°N |
| Dates | 1963 – 2026 |
| Variables | snow depth, white ice, black ice, total ice, hydrostatic water level, flags |

**Regional subset (41–49.5°N, 93–75°W):** 511 rows across **two sites only** — Crystal Lake, WI and Little Turkey Lake, ON. The dataset is overwhelmingly Arctic and subarctic.

| Regional statistic | Value |
|---|---|
| Total ice, mean | 0.49 m |
| White ice, mean | 0.33 m (67% of total) |
| Black ice, mean | 0.18 m |
| Snow depth, mean | 0.17 m |

**Verdict:** valuable for cold-range behaviour and for the snow-ice composition test, but it is **not** a temperate small-lake source. It cannot carry the small-lake calibration on its own.

**Known issue:** `External_LI-TC_v2.csv` contains dates as late as 2075, clearly typos. Any record after the current date is dropped.

---

## 4. NTL-LTER snow and ice depth

- **Package:** `knb-lter-ntl.34.35`, "North Temperate Lakes LTER: Snow and Ice Depth"
- **File:** `data/raw/ntl/ntl34_v10.csv` (49,749 bytes)
- **Licence:** CC-BY per EDI/KNB convention; cite the package DOI.

| Property | Value |
|---|---|
| Rows | 1,015 |
| Lakes | 11 |
| Records with ice thickness | 1,010 (1,009 > 0) |
| Dates | 1982-01-18 – 2026-02-13 |
| Median / p95 / max thickness | 40 / 61 / 78 cm |
| White ice, mean | 12.9 cm (32% of total) |
| Blue ice, mean | 26.4 cm |

Lakes: Allequash (AL), Big Muskellunge (BM), Crystal Bog (CB), Crystal (CR), Fish (FI), Mendota (ME), Monona (MO), Sparkling (SP), Trout Bog (TB), Trout (TR), Wingra (WI).

**White-ice fraction is strongly depth-dependent:**

| Lake | Approx. max depth | Median white fraction | Mean total |
|---|---|---|---|
| Crystal Bog | 3 m | 61% | 36 cm |
| Trout Bog | 2 m | 54% | 36 cm |
| Allequash | 8 m | 27% | 45 cm |
| Crystal | 20 m | 22% | 39 cm |
| Sparkling | 20 m | 22% | 41 cm |
| Wingra | 3 m | 22% | 32 cm |
| Fish | 18 m | 18% | 34 cm |
| Big Muskellunge | 21 m | 16% | 45 cm |
| Trout | 36 m | 13% | 44 cm |
| Mendota | 25 m | 11% | 34 cm |
| Monona | 22 m | 9% | 37 cm |

Despite white-ice fractions from 9% to 61%, mean total thickness spans only **32–45 cm**. Snow redistributes black ice into white ice without moving the total much. This is the core evidence for keeping snow out of the v1 model and testing it as a residual instead.

**Limitation:** sampling is sparse — about **2 observations per lake per year** over 44 years. Good for seasonal maxima and for seasonal shape, weak for day-level growth curves.

**Missing:** lake depths are not in the file. Published maximum depths must be sourced separately before these lakes can enter the fit.

---

## 5. Michigan DNR hydrography and bathymetry

- **Service:** `services3.arcgis.com/Jdnp1TjADvSDxMAX/arcgis/rest/services/DNRHydrologyOPENDATA/FeatureServer`
- **Licence:** Michigan DNR open data; attribution required.
- **Projection:** wkid 102123 / 3078 (Michigan GeoRef, oblique Mercator).

| Layer | Name | Features |
|---|---|---|
| 3 | DNR Reference Hydro Poly | 60,079 total; 57,712 `Type=2` (lake/pond) |
| 3 filtered | `Type=2 AND SqKM >= 0.1` | **3,548** |
| 1 | IFR MI Inland Lake Contours | 25,492 lines, **2,166 distinct lakes** |
| 0 | IFR MI Lake Deep Points | **2,651** points |

### Coverage (corrected after spatial matching)

The raw counts above are **not** lake coverage. The deep-point layer contains 2,651 rows but only **2,175 distinct lakes**, and many are below our 10 ha threshold. Matching each unique point to a ≥10 ha polygon (nearest, ≤250 m) and each contour group by vertex-majority gives the real picture:

| Asset | Distinct survey lakes | Matched to a ≥10 ha polygon |
|---|---|---|
| Surveyed deep point | 2,175 | **1,551** |
| Digitised contours | 2,166 | **1,552** |
| Either | — | **1,562** |

| Coverage of the 3,548 displayed lakes | Lakes | Share |
|---|---|---|
| Real depth (point or contour) | 1,562 | **44.0%** |
| Regression required | 1,986 | **56.0%** |

Coverage rises with lake size: 70% of the largest 100 lakes have real depth, 64% of the largest 500, 61% of the largest 1,000. The unmatched deep points sit a median 1,016 m from any ≥10 ha polygon, i.e. they are genuinely small lakes, not failed matches. Contours add only 11 lakes beyond the deep points.

**So just under half of displayed Michigan lakes have surveyed bathymetry, not three quarters.** The area→depth regression carries the other 56%.

### Units — important

`MaxDepth` on the deep-point layer is in **feet, not metres**. Confirmed against published soundings: Torch 250 (285 ft published), Crystal/Benzie 175 (165), Higgins 130 (135), Charlevoix 122 (122), Houghton 21 (21). Convert with ×0.3048 before use. Eight records report zero depth and should be treated as missing.

### Fields

- Layer 3: `Name`, `Name2`, `Type`, `Type2`, `Acres`, `SqMiles`, `SqKM`, `Elevation`, `NameSRC`, `GlobalID`, `OBJECTID`, `Shape__Area`, `Shape__Length`. **No join key to bathymetry.**
- Layer 1: `NewKey`, `MinDepth`, `MaxDepth`, `Name`.
- Layer 0: `Id`, `NewKey`, `MaxDepth`, `LakeName`, `TRS`, `County`, `Latitude`, `Longitude`, `GlobalID`, `OBJECTID`.

**Join strategy:** bathymetry must be linked to lake polygons **spatially** — deep points by their lat/lon, contours by their geometry. There is no shared identifier. This makes the ingest step a point-in-polygon join rather than a table join, and it needs a tolerance for points that fall just outside a polygon edge.

---

## 6. ERA5-Land temperature

- **Dataset:** `derived-era5-land-daily-statistics` via Copernicus CDS.
- **Resolution:** 9 km, global land.
- **Coverage:** 1950-01-01 – present.
- **Variable needed:** `2m_temperature`, daily mean (`daily_mean`), giving cumulative freezing degree-days from 1 October.
- **Status:** API handshake verified (25 kB test tile, ~50 s round trip). Token and licence acceptance confirmed.
- **Decision:** this is the primary temperature source. GEE's `ECMWF/ERA5_LAND/DAILY_AGGR` is the same data but requires a Google Cloud project and awkward bulk export; its `lake_ice_depth` band is retained only as an optional free 75-year sanity check.

**Not used for snow.** ERA5-Land `snow_depth` over lake grid cells is land-scheme snow and does not represent snow on ice. It will not enter the model as a spatial input.

---

## 7. What can and cannot be validated

| Question | Can it be answered? | With what |
|---|---|---|
| Seasonal peak thickness, Great Lakes | Yes, 1966–79 | G00803, 4,745 obs |
| Seasonal peak thickness, small inland lakes | Yes, 1982–2026 | NTL, 1,009 obs |
| Freeze-up date | Partially — Nov–Dec sampling is thin (44 and 473 obs) | G00803 |
| Intra-season growth curve, Great Lakes | Weakly — weekly, coastal only | G00803 |
| Intra-season growth curve, small lakes | Weakly — ~2 obs/lake/year | NTL |
| **Offshore / deep-water thickness on the Great Lakes** | **No** | — |
| **1950–1965** | **No** | — |
| Snow effect on total thickness | Yes, as a residual diagnostic | G00803 snow depth, NTL `avsnow`, ECCC snow depth |
| Cold-range behaviour | Partially | ECCC, 2 regional sites + Arctic sites |

**The two structural gaps are offshore validation and the 1950–65 extrapolation.** Neither can be closed with available data; both must be stated on the finished map.

---

## 8. Gaps to close before fitting

1. **Lake depths for NTL and ECCC regional sites.** Not in the data files. Source published maximum depths (Wisconsin DNR, Ontario MNRF / ECCC metadata).
2. **Deep-point → polygon spatial join** for Michigan, with edge tolerance.
3. **Contour → volume integration** to derive mean depth for the 2,166 contoured lakes.
4. **Area→depth regression** fitted on the 1,551 lakes with a surveyed deep point, for the 1,986 lakes with no bathymetry at all.
5. **ERA5-Land basin subset retrieval** — confirm request size and timing for the full 1950–present, full-basin pull before committing to the 64×64 grid.
6. **NOAA 3 arc-second bathymetry** for the three Great Lakes (Superior, Michigan, Huron) — not yet downloaded.

---

## 9. Licensing and attribution

| Source | Licence | Attribution |
|---|---|---|
| G00803 | NOAA, public domain | Sleator (1995), NSIDC, doi:10.7265/N5KW5CXG |
| ECCC Lake Ice DB | Open Government Licence – Canada | Environment and Climate Change Canada |
| NTL-LTER | CC-BY (EDI) | NTL-LTER, knb-lter-ntl.34 |
| Michigan DNR | State open data | Michigan Department of Natural Resources |
| ERA5-Land | Copernicus licence | Contains modified Copernicus data, ECMWF |

---

## 10. Open decisions

- **Pass/fail gate:** proposed as median annual-max thickness error ≤ 8 cm, freeze-up date error ≤ 10 days, no depth bias. Not yet approved. Given the offshore validation gap and the 1950–65 extrapolation, the gate should be evaluated on the coastal/inland population only.
- **Snow:** excluded from v1 by decision. Residual-vs-snow-depth diagnostic to run after fitting.
