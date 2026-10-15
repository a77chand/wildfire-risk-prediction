# Model card: BC wildfire ignition risk

## Summary
Gradient-boosted classifiers that estimate, for each ~0.5° grid cell in British Columbia and each week of
April–October, the probability of **at least one wildfire ignition in the following 7 days**. Four targets share
one pipeline and feature set: any ignition, large fire (≥ 10 ha), lightning-caused, and person-caused.

**Intended use:** research, teaching and portfolio demonstration of leakage-safe geospatial time-series ML.
**Not intended for:** operational fire management, public warnings, insurance or any safety-critical decision.

## Data
| | |
|---|---|
| Labels | BC Wildfire Service historical incidents, 2005–2025, `FIRE_TYPE == "Fire"` (34,253 records after cleaning) |
| Weather | NASA POWER daily reanalysis (MERRA-2) at each cell centre: max temperature, dew point, precipitation, max wind |
| Terrain / access | Elevation (from POWER); road-segment count per cell from the BC Digital Road Atlas (log-scaled) |
| Derived | Canadian FWI System codes, rolling/lagged weather, neighbour-cell FWI/DC, recent and historical ignition rates |
| Rows | 616 cells × 31 weekly dates × 21 years = 401,016 cell-weeks |

## Splits
Chronological. Train 2005–2018, validation 2019–2020 (iteration count, calibration experiments), test 2021–2025.
Local ignition-history features are fitted on training years only and are leave-one-year-out for training rows.

## Performance (test years 2021–2025)
See [README](../README.md#results-held-out-test-years-20212025) for the full tables, intervals and figures.

| Target | Base rate | ROC-AUC (clim → model) | PR-AUC (clim → model) |
|---|---|---|---|
| Any ignition | 5.5% | 0.861 → 0.887 | 0.281 → 0.322 |
| Large fire (≥ 10 ha) | 1.0% | 0.788 → 0.855 | 0.047 → 0.062 |
| Lightning-caused | 3.2% | 0.885 → 0.914 | 0.234 → 0.275 |
| Person-caused | 2.4% | 0.871 → 0.906 | 0.177 → 0.204 |

"Climatology" is the historical rate for the same cell and month (training years only), a deliberately strong baseline.

## What the model is and is not good at
- **Good at timing.** It tracks which *weeks* are dangerous across the province (Spearman 0.88 vs 0.68 for climatology
  on weekly ignition-cell counts; person-caused 0.69 vs 0.34).
- **Not better than climatology at ranking cells within a week.** If a dispatcher could watch only the 30 highest-risk
  cells in a given week, the model covers about the same share of ignitions as climatology (24.4% vs 23.9%).
  Its pooled-ranking advantage comes mostly from knowing which weeks are active.
- **Calibrated.** Expected calibration error ≈ 0.006 on the any-ignition target. Isotonic recalibration on the
  validation years made test calibration *worse* (the validation years were unusually quiet), so it is not used.

## Known limitations and risks
- Lightning occurrence, fuel type/NDVI and snowpack are not features; lightning is the dominant natural cause in BC.
- Reanalysis weather is coarse (~50 km) and FWI inputs are daily maxima rather than noon observations.
- Incident records reflect reporting and detection practice, which changed over 20 years; "any ignition" is mostly small fires.
- Five test years with large inter-annual variation; climate non-stationarity is not captured by the reported intervals.
- Road-segment counts depend on digitising detail and are a relative access proxy, not road length.
- The live nowcast uses data ~3 days behind real time and is **not** a forecast; training weather and live weather
  come from the same source, so there is no reanalysis-to-forecast domain shift, but also no forward-looking weather.
- Cells are not equal-area.

## Ethical considerations
Predictions concern places and times, not people; person-caused predictions use only aggregate road access and
historical rates. Do not use the outputs to target or surveil individuals or communities.

## Reproducibility
`make data train report export` regenerates everything from open data. Seeds, years, splits and
hyperparameters are in `configs/default.yaml`; hyperparameters were chosen with `make tune` (`reports/tuning.json`).
