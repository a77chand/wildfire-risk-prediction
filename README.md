# Wildfire Ignition Risk for British Columbia

Weekly, grid-cell wildfire **ignition risk** for British Columbia, built from open weather data, the Canadian Fire Weather Index (FWI) System and 20 years of agency ignition records, and evaluated the way it would actually be used: on **future years the model never saw**, against a **strong climatology baseline**.

![Risk map for the busiest held-out week](reports/figures/risk_map_peak_week.png)

*Predicted probability of at least one ignition in each ~0.5° cell during the 7 days after 15 Jul 2024 (a held-out test year). Dots are the 441 ignitions that followed. [Interactive version](reports/risk_map_peak_week.html).*

## The question

> For each grid cell and each week of the April–October season: **will at least one wildfire ignite in that cell in the next 7 days?**

616 cells (0.5° ≈ 55 × 30–35 km) × 31 weeks × 21 years = **401,016 cell-weeks**, 5.0% of which contain an ignition.

## Results (held-out test years 2021–2025)

Training uses 2005–2018, model selection and early stopping use 2019–2020, and everything below is scored on 2021–2025 (95,480 cell-weeks, including the extreme 2021 and 2023 seasons). The split is strictly chronological.

| Model | ROC-AUC | PR-AUC | Brier skill | Ignitions caught in top 5% of risk | Lift @ top 5% |
|---|---|---|---|---|---|
| Climatology (cell × month historical rate) | 0.861 | 0.281 | 0.142 | 31.1% | 6.2× |
| Logistic regression | 0.874 | 0.313 | 0.156 | 33.6% | 6.7× |
| Random forest | 0.886 | 0.324 | 0.174 | 34.7% | 6.9× |
| **Gradient boosting** (deployed) | **0.887** | **0.325** | 0.172 | **34.7%** | **6.9×** |

![ROC and precision-recall](reports/figures/model_comparison.png)

**How to read this.**

- **The baseline is already strong.** Knowing *where and when* fires usually happen gets ROC-AUC 0.861. The earlier version of this project reported ~0.91 on synthetic data and never compared against anything; 0.887 on real held-out years against this baseline is the honest number.
- **The improvement is real but modest.** Cluster-bootstrap over prediction dates (200 resamples) gives gradient boosting **+0.026 ROC-AUC (95% CI 0.020–0.033)** and **+0.044 PR-AUC (95% CI 0.022–0.064)** over climatology. It beats climatology in every one of the five test years.
- **Probabilities are usable.** Expected calibration error is 0.004: when the model says 30%, about 30% of those cell-weeks see an ignition.
- **A negative result, kept on purpose.** Isotonic recalibration fitted on the 2019–20 validation years *hurt* test performance (ECE 0.004 → 0.018): those validation years were quiet and the test years were not. The raw model was already calibrated, so that is what is deployed.

![Per-year skill](reports/figures/per_year.png)

### What actually drives the predictions

![SHAP importance](reports/figures/shap_importance.png)

![Feature-group ablation](reports/figures/ablation.png)

| Feature set (re-trained, scored on test years) | PR-AUC | ROC-AUC |
|---|---|---|
| All features | 0.325 | 0.887 |
| Without FWI codes | 0.323 | 0.886 |
| Without ignition history | 0.306 | 0.879 |
| Location + season + history only (no weather) | 0.313 | 0.878 |
| Location + season only | 0.291 | 0.870 |
| Weather + FWI + season only (no location) | 0.227 | 0.842 |

- Most skill comes from **local ignition history and seasonality**. Weather and FWI add a small, consistent increment (PR-AUC 0.313 → 0.325).
- **The FWI codes add essentially nothing beyond the raw weather features** (0.323 → 0.325). They are in the pipeline because they encode weather *history* (e.g. the drought code integrates months of rain deficit) and the FWI implementation is validated against published reference values, but on this data they are redundant.
- **Spatial generalisation:** holding out whole 2° × 2° blocks (5-fold, per-cell history features removed), the model reaches ROC-AUC 0.885 ± 0.016 versus 0.673 for a no-local-history climatology, so it is not just memorising cells.

## Why these design choices

The first version of this repo trained a Random Forest on **synthetic data whose label was a formula of its own input features**, so its AUC measured how well a forest recovers a rule I had written, and its "inland risk is higher" map came from uniformly random coordinates. This version fixes the problems that mattered:

| Problem | Fix |
|---|---|
| Synthetic data, circular labels | Real labels: BC Wildfire Service ignition records (34k agency-confirmed wildfires, 2005–2025). Nuisance fires, smoke chases and field activity excluded; typo'd dates removed |
| Satellite hotspots only exist where there *is* fire (no negatives) | Cell-week framing: every cell-week without an ignition is a genuine negative |
| `brightness`/`FRP`/`confidence` only exist once a fire is burning | Removed. Features are pre-ignition conditions only: weather, FWI, terrain, season, prior ignitions |
| Random train/test split leaks neighbouring places and times | Chronological split + spatial-block CV; date-clustered bootstrap CIs |
| No baseline | Climatology baseline, logistic regression, random forest, gradient boosting |
| ROC-AUC alone flatters a 5% positive rate | PR-AUC, Brier skill, calibration (ECE), top-k capture and lift |
| Preprocessing fitted on all data | Local-history features are fitted on training years only and are leave-one-year-out for training rows |
| No tests | 20 tests, including that **features at date *t* cannot see anything after *t*** and that labels cover only days *t+1…t+7* |

## Data

| Role | Source | Notes |
|---|---|---|
| Labels | BC Wildfire Service historical incident locations (BC Data Catalogue, “Fire Incident Locations – Historical”) via open WFS | Open Government Licence – BC. No API key |
| Weather | [NASA POWER](https://power.larc.nasa.gov/) daily point API (MERRA-2 reanalysis): max temperature, dew point, precipitation, max wind; also supplies elevation | Open, no key. ~0.5° × 0.625° native resolution |
| Boundary | BC provincial boundary, BC Data Catalogue | Used to select cells and draw maps |
| Optional labels | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area) satellite hotspots (`label.source: firms`) | Needs a free `FIRMS_MAP_KEY`. Implemented but **not exercised in this repo's results or tests**; includes non-wildfire heat sources |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"

make data      # ~5-10 min first time: boundary, ignitions, weather for 616 cells -> data/
make train     # fits models, evaluates on held-out years (add --quick via the CLI to skip bootstrap/CV)
make report    # figures, SHAP, maps -> reports/
make export    # small artefact bundle for the app -> artifacts/
make app       # Streamlit explorer
make test
```

Equivalent CLI: `python -m wildfire.cli {fetch,build,train,report,export,live,all}`. Everything is configured in [configs/default.yaml](configs/default.yaml) (grid size, years, season, horizon, splits, hyperparameters).

The Streamlit app and the committed `artifacts/` bundle (≈1 MB) run from a fresh clone without re-downloading data. The notebook [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) walks through the results.

### Interactive app

- **Back-test explorer**: slide through any held-out week and compare the model to climatology against what actually ignited.
- **Live nowcast**: scores every cell from the latest NASA POWER weather and newest BCWS records, with a per-cell SHAP explanation. NASA POWER lags real time by about 3 days and rate-limits requests, so a first run takes a few minutes and the output is a **nowcast for the 7 days after the latest observation, not a weather forecast**.

## Project layout

```
wildfire/            package
  grid.py            BC boundary + regular grid clipped to it
  ignitions.py       BCWS (and optional FIRMS) loaders, cleaning
  weather.py         NASA POWER client (retries, backoff, resumable)
  fwi.py             Canadian FWI System (Van Wagner 1987), vectorised across cells
  features.py        daily features -> weekly rows, timing contract, local-history encoder
  models.py          climatology baseline, sklearn models, year-based early stopping
  evaluate.py        metrics, date-clustered bootstrap, spatial-block CV, ablation
  explain.py         SHAP (permutation explainer, works for any sklearn model)
  plots.py maps.py   figures and Folium maps
  live.py            nowcast
  cli.py             command line
app/streamlit_app.py interactive explorer
configs/             experiment configuration
tests/               pytest suite
artifacts/           committed, app-ready model + grid + test predictions (~1 MB)
reports/             metrics.json, figures, interactive HTML maps
notebooks/           executed walkthrough
```

## Limitations

- **Missing drivers.** Lightning is the dominant natural ignition cause in BC and human access (roads, settlements) drives person-caused fires. Neither is in the features. Fuel type/NDVI and snowpack are also absent. These are the most likely sources of the next gain.
- **Coarse weather.** Reanalysis at ~50 km smooths local extremes (convective storms, valley winds). FWI inputs are daily max temperature, dew-point-derived humidity and max wind rather than true noon observations, so the codes run slightly "hot and dry". April FWI start-up ignores snowpack at high elevation.
- **Label effects.** Records reflect where crews are dispatched and reported; detection and reporting practice changes over 20 years. The target is "any ignition" (mostly tiny fires), not "large or damaging fire".
- **Short, non-stationary test window.** Five test years with strong year-to-year variation (see the per-year chart), and a warming climate means the future may not resemble 2005–2018. Reported intervals reflect sampling over dates, not this structural uncertainty.
- **Grid cells are not equal-area** (0.5° of longitude is narrower at 59°N than at 49°N).
- **Not an operational tool.** This is a research / portfolio project. Do not use it for safety decisions; use [BC Wildfire Service](https://www2.gov.bc.ca/gov/content/safety/wildfire-status) information.

## Next steps

1. Add lightning (e.g. ERA5 CAPE/convective indices as a proxy, or the Canadian Lightning Detection Network if accessible) and human-access features.
2. Add fuel/vegetation (MODIS NDVI/EVI, BC fuel types) and snow cover.
3. Predict *large* fires (e.g. ≥ 10 ha) as a separate target, which matters more for impact than ignition counts.
4. Move to a forecast: replace nowcast weather with NWP forecasts and re-validate (domain shift from reanalysis).
5. Sequence / spatial deep models (ConvLSTM on gridded weather) once there is a clear gap over boosted trees to justify them.

## Tech

Python · pandas · NumPy · scikit-learn (HistGradientBoosting, random forest, logistic regression) · SHAP · GeoPandas / Shapely · Folium · Streamlit · pytest · GitHub Actions · ruff
