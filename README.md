# Global Tropical Cyclone Impact Model (0.1° Grid)

This repository provides a high-performance pipeline for generating a high-resolution (**0.1° resolution**, approx. 11km at the equator) global dataset of Tropical Cyclone (TC) impacts. It includes a **Two-Stage XGBoost** model designed to predict sub-national **affected populations** by integrating physical hazards with socioeconomic vulnerability and settlement morphology.

The core of this project is the **Data Factory**, which automates the collection and processing of 16 spatial and dynamic features, anchored strictly to verified [EM-DAT](https://www.emdat.be/) impact records, into a standardized `training_dataset.parquet`.

---

## Citation

If you use this repository or the generated dataset in your research, please cite:
> *Moss, F et al. Global Sub-national Impact-based Forecasting for Tropical Cyclones Using Open Data: Combining Machine Learning and Exposure-based Approaches, EGUsphere (2026). doi: 10.5194/egusphere-2026-1996 (under review).*

---

## Repository Structure

```text
├── data/
│   ├── input/                # Raw downloads (GADM, SRTM, WorldPop, EM-DAT, etc.)
│   └── output/               # Processed feature CSVs and Model results
├── src/
│   ├── collectors/           # Data acquisition (general_collector.py, pps_collector.py,
│   │                         #   archived forecasts from TIGGE: forecast_collector.py)
│   ├── static_features/      # Spatial Processing (grid_cells.py, process_gadm.py, etc.)
│   ├── dynamic_features/     # Event-Based Processing (process_emdat.py, process_wind.py, etc.),
│   │                         #   forecast tracks & their wind fields (process_forecast_tracks.py)
│   ├── models/               # Two-Stage XGBoost, two-part GLM & Baselines (train.py),
│   │                         #   tuning (hyperparameter_search.py, glm_search.py),
│   │                         #   impact from an archived forecast (forecast_model.py)
│   ├── evaluation/           # LOOCV (cv_strategies.py), metrics & bootstrap CIs (bootstrap.py),
│   │                         #   scored results (paper_metrics.py, basin_metrics.py),
│   │                         #   paired model differences (model_comparison.py),
│   │                         #   forecast study & its lead-time scores
│   │                         #   (forecast_study.py, forecast_metrics.py)
│   ├── interpretability/     # SHAP Analysis & Visualization
│   ├── utils/                # Shared helpers (geo_utils, logging_setup, time interpolation)
│   ├── config.py             # Global constants, URLs, and ISO3 List
│   └── dataset_builder.py    # Master script to compile the final Parquet
├── visualization/            # Notebooks producing the figures, with lib.py behind them
├── main.py                   # Unified CLI Entry Point
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Prerequisites & Data Preparation

### 1. System Dependencies
The pipeline relies on several low-level geospatial libraries for raster processing and atmospheric data handling:
* **Python 3.10-3.12**: The core environment.
* **GDAL (Geospatial Data Abstraction Library)**: Required for processing SRTM elevation data and JRC urbanization rasters (specifically `gdaldem`). The pip `gdal` bindings are built against your system libgdal, so install GDAL **first** and keep the versions matched.
* **OpenMP runtime (macOS)**: `xgboost` needs a recent `libomp` — `brew install libomp` (an outdated libomp fails at import with a missing-symbol error).



#### **GDAL (Geospatial Data Abstraction Library)**
* **Installation:**
    * **Ubuntu/Debian:** `sudo apt-get install gdal-bin libgdal-dev`
    * **macOS (Homebrew):** `brew install gdal`
    * **Windows:** Use the [OSGeo4W installer](https://trac.osgeo.org/osgeo4w/) or install via Conda: `conda install -c conda-forge gdal`.
    * **No root (shared servers/HPC):** `conda install -c conda-forge gdal` works everywhere, not just on Windows, and has the advantage that `libgdal` and the binaries are version-matched inside the environment.

> **`gdaldem` must be on `PATH`, not merely installed.** The terrain step shells
> out to the `gdaldem` binary (`subprocess.run(["gdaldem", ...])`), which is a
> `PATH` lookup — unlike everything else here, which only needs importable Python
> packages. Calling the environment's interpreter by absolute path
> (`/path/to/env/bin/python main.py`) without activating the environment imports
> fine and *then* fails on every country. `process_all_srtm()` now checks this up
> front and raises, but activate the environment (`conda activate <env>`) or put
> its `bin/` on `PATH` before running the pipeline. Verify with `which gdaldem`.


### 2. Python Environment
Install the required Python libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Note**: Install GDAL first, then the requirements.txt pip list

On an unreliable connection, pass retries so a momentary CDN timeout is not
reported as a missing package (pip renders a failed index fetch as
`from versions: none`, which looks like a bad pin):
```bash
pip install --retries 15 --timeout 90 -r requirements.txt
```

### 3. External Data Requirements
* **EM-DAT (Ground Truth)**: The pipeline is anchored by verified disaster records. You must place your processed EM-DAT file at:  
  `data/input/EMDAT/emdat.csv`

    **Required CSV Schema:**
    * `sid`: IBTrACS Storm ID (e.g., 2013309N06133).
    * `iso3`: 3-letter country code.
    * `DisNo.`: EM-DAT Disaster Number.
    * `Total Affected`: The target variable (Total number of people affected).
    * `Total Deaths`: Secondary target/tracking variable.
    * `Admin Units`: JSON-formatted string of affected administrative regions using GAUL 2014-2015 dataset(used for spatial expansion).

**Note: we leave to the user the "sid" and "Disno." matching of storms. This involves manual labeling based TC names, locations and dates on top of classic fuzzy-matching techniques or (alternatively) the use of LLM matching approaches.** 

* **GAUL 2014-2015 Dataset (Administrative Boundaries)**: This spatial dataset is required to map the EM-DAT `Admin Units` to physical geometries. The collector first tries the direct FAO catalog mirror (`GAUL_ADM2_URL` in `src/config.py`); if that link is unavailable, export it through **Google Earth Engine (GEE)**. 
    * **How to obtain**: 
      1. Register for a free [Google Earth Engine account](https://earthengine.google.com/).
      2. Locate the GAUL dataset in the GEE Data Catalog (e.g., `FAO/GAUL/2015/level2`)
      3. Export the dataset as a GeoJSON or Shapefile using the GEE Code Editor or the Python API. 
    * **Placement**: Once exported, place the spatial file in your data directory as `data/input/SHP/global_shapefile_GAUL_adm2.gpkg`.

* **NASA PPS (Precipitation)**: To access GPM-IMERG rainfall data, register a free account at [NASA PPS](https://registration.pps.eosdis.nasa.gov/registration/). Once registration is complete, PPS will use your **email address (lower-cased) as both your username and password**.

    Create a `.env` file at the project root (already listed in `.gitignore`) with your credentials:
    ```
    NASA_PPS_USERNAME=your_email@example.com
    NASA_PPS_PASSWORD=your_email@example.com
    ```
    `process_rain_features.py` loads this file automatically at startup via `_load_dotenv()`.

    The collector (`pps_collector.py`) pulls from the **`arthurhouhttps` mirror**, which works with a standard PPS registration — the `jsimpsonhttps` mirror requires separately-approved elevated access that most accounts don't have. The products served there are the **IMERG Final run** (the product names carry no `-L` suffix), i.e. gauge-calibrated, which is what a historical impact record wants.

    `rainfall_max_24h` is the **maximum single-day accumulated rainfall (mm) per grid cell** over the ±2-day window around landfall — not a rain-rate average. It can be built from either of two IMERG products, selected with `--imerg-product` or the `TC_IMPACT_IMERG_PRODUCT` environment variable:

    | | `half_hourly` (default) | `daily` |
    |---|---|---|
    | granules | 48 × `3B-HHR-GIS ... total.accum` per day | 1 × `3B-DAY-GIS` per date |
    | stored units | `0.1 mm` per half hour | `0.1 mm/hr` — a **rate** |
    | conversion | `raw / 10`, summed over the 48 | `raw / 10 * 24` |
    | resolution | 0.1 mm | 2.4 mm/day |
    | download | ~166 MB per storm | ~1 MB per date, shared across storms |
    | stored under | `data/input/gpm_data/{typhoon_name}/` | `data/input/gpm_data_daily/` |

    The two agree to **correlation 0.9994** (median relative difference 0.17%, verified over 576k pixels); above 75 mm/day they are effectively identical (±0.6%), while below ~25 mm the daily route's 2.4 mm quantisation shows (±6%). Use `half_hourly` when light-rain precision matters and `daily` when download volume does — the whole 2000–2022 record is ~2.8 GB as daily granules against ~90 GB as half-hourly ones.

    > **The two products are not interchangeable without their conversion.** The daily granule stores a *rate*, so using it as if it were an accumulation under-counts rainfall by a factor of 24. Each product therefore has its own conversion function in `process_rain_features.py` (`half_hour_raw_to_mm`, `daily_raw_to_mm`) and they are never mixed. Both report short days rather than silently summing a partial one, since the feature is a maximum over days and an incomplete day can only bias it downward.

* **SHDI Index (Vulnerability)**: Download this dataset manually from *https://globaldatalab.org/shdi/download/shdi/* and put it in under `/data/SHDI/GDL-Subnational-HDI-data.csv` (requires logging to GlobalDataLab)

* **TIGGE cyclone tracks (optional — only for the historical forecast study)**: archived ensemble track forecasts are read from NCAR's **GDEX** (which replaced the RDA; older RDA links no longer resolve). No registration is needed and `forecast_collector.py` downloads them itself, so this is not a prerequisite for the main pipeline. The centre is chosen with `--forecast-provider`, named by its WMO originating-centre code: `rksl` (Korea Meteorological Administration, the default), `ecmf`, `kwbc`, `egrr`, `rjtd`, `lfpw`, `ammc`, `babj`. Files land under `data/input/FORECAST/<provider>/<DisNo.>/`.

    > **A centre's coverage decides the study's.** Each centre tracks the basins it forecasts for, so the choice of provider is also a choice of basin: `rksl` reaches the Western North Pacific only. No centre's archive reaches before about 2006, which bounds the study whatever the dataset covers.

---

## Execution Workflow

The `main.py` script manages the end-to-end workflow. By default, it operates as a Data Factory.

### 1. Build the Dataset (Full Pipeline)
Downloads all public data, initializes the 0.1° grid, processes all 16 features, performs sub-national spatial expansion, and compiles the final dataset.
```bash
python main.py
```
The scope is **global by default** (`ISO3_LIST = None` in `src/config.py`; Antarctica excluded); set `ISO3_LIST` to a short country list there to test the plumbing cheaply. Data lives under `./data/` unless overridden with `TC_IMPACT_DATA_DIR` (or `TC_IMPACT_INPUT_DIR` / `TC_IMPACT_OUTPUT_DIR` individually) — useful when the raw downloads should sit on a large scratch disk.

### 2. Partial Execution (By Stage)
Use the `--stage` flag to execute or re-run specific parts of the pipeline:
```bash
# 1. Download raw data (GADM, WorldPop, SRTM tiles, etc.)
python main.py --stage collect

# 2. Generate the 0.1° coordinate reference grid and landmask
python main.py --stage grid

# 3. Process static spatial features (GADM ADM2, Terrain, Vulnerability)
python main.py --stage static

# 4. Process dynamic hazard & impact layers (Wind, Rain, EM-DAT spatial mapping)
#    Add --imerg-product daily to use the daily IMERG granules instead of the
#    default 48-per-day half-hourly ones (same feature, ~50x less download).
python main.py --stage dynamic

# 5. Assemble the final training_dataset.parquet
python main.py --stage build
```

### 3. Hyperparameter Search (optional, but do it before reporting results)
```bash
# Stage 1 -- random search over the tree parameters and the ratios;
# writes data/model_hyperparameters.json
python src/models/hyperparameter_search.py --n-iter 80 --n-events 200

# Stage 2 -- trees frozen at stage 1's winner, every (u1, u2) scored
# exhaustively and the threshold swept for free
python src/models/hyperparameter_search.py --search ratios \
    --from-json data/model_hyperparameters_stage1.json --select-on f1_high

# Tiny budget, just to check the plumbing runs
python src/models/hyperparameter_search.py --dry-run
```

`src/models/hyperparameter_search.py` tunes **both stages together** — the stage-1 classifier, the stage-2 regressor, the two class-balancing ratios (`u1`, `u2`) and the stage-1 decision threshold (`clf_threshold`). Tuning the stages separately would miss their interaction: how aggressively stage 1 flags cells decides which rows stage 2 ever sees.

The two passes split that search rather than separating the stages. The random pass covers the tree parameters, which are many and whose surface is flat, by sampling. The ratios pass then grids `u1 × u2` exhaustively, since those two and the threshold set the precision/recall operating point the high-impact class depends on, and a sampled grid leaves that under-explored. The second pass is cheap because the search factorizes: `u1` enters only stage 1 and `u2` only stage 2, so each fold needs `|u1| + |u2|` fits rather than `|u1| × |u2|`, and every pair is assembled from the stored predictions. The threshold is applied to those same predictions, so sweeping it costs nothing and the whole curve is reported.

Three properties are worth knowing, because they decide whether the reported numbers mean anything:

* **Folds are grouped by physical cyclone (IBTrACS `sid`), never by grid cell or country-event record.** Cells within one cyclone are strongly correlated, and a storm that hits several countries contributes several `DisNo.` records — a row-wise or `DisNo.`-wise split would put the same storm on both sides and report a score the model cannot reproduce on an unseen one. LOOCV in `src/evaluation/cv_strategies.py` leaves one `sid` out per fold for the same reason.
* **A held-out set of events is scored exactly once**, after selection. The cross-validated score is optimistic by construction — it is the quantity that was optimised — so the held-out number is the one to quote.
* **Metrics are computed at ADM1 level on pooled out-of-fold predictions**, where impact is actually reported, at both the "affected at all" (0%) and "highly affected" (15%) thresholds.

The winner is written to `data/model_hyperparameters.json` and **loaded automatically** by `TwoStageXGBoost` — no copying numbers by hand. If the file is absent the model falls back to its built-in defaults, so a fresh clone still runs; explicit constructor arguments always win over the file. Override the location with `TC_IMPACT_HYPERPARAMETERS`.

> If you restricted `ISO3_LIST` to a small development subset while testing, point `--input` at a full dataset for the search — a handful of countries leaves too few events, and every one fails the selection filters.

#### The statistical baseline (two-part GLM)
```bash
# Tune it on the XGBoost search's own cyclones, folds and objective;
# writes data/glm_hyperparameters.json (run the XGBoost search first)
python src/models/glm_search.py

# Also grid the balancing ratios instead of keeping the XGBoost ones
python src/models/glm_search.py --tune-ratios

# Tiny budget, just to check the plumbing runs
python src/models/glm_search.py --dry-run
```

`src/models/two_stage_glm.py` asks whether the learning algorithm matters: it keeps the two-stage structure and the same predictors, training rows, rebalancing, folds and ADM1 aggregation, and replaces the boosted trees with a regularised logistic regression for occurrence and a fractional logistic regression for severity, fitted on cells with reported impact. The hard gate is kept, so the baseline predicts exact zeros like the XGBoost model.

`src/models/glm_search.py` gives it its own tuning under the XGBoost protocol, so that neither model is compared at a configuration chosen for the other. The search pool and held-out cyclones are read from `data/model_hyperparameters.json`, the folds are rebuilt with the same splitter and seed, candidates are scored with the same ADM1 function and selected on the same objective (`f1_high`), and the winner is scored once on the held-out set. Searched: the two regularisation strengths and the stage-1 cutoff. The XGBoost cutoff is not carried over; the GLM picks its own from the same set of cutoffs the XGBoost search scored (`--wide-thresholds` sweeps 0.01 to 0.99, but on `f1_high` alone a very low cutoff can win by a hair by gating nearly every cell, so check the stored curve before preferring one). The ratios stay at the XGBoost model's values unless `--tune-ratios` is given. The JSON also records the XGBoost model's scores on the same pool and held-out set, for a first comparison. Override its location with `TC_IMPACT_GLM_HYPERPARAMETERS`.

The whole comparison, end to end, once the XGBoost search and run exist:

```bash
# 1. Tune the GLM on the XGBoost search's own cyclones, folds and objective
python src/models/glm_search.py

# 2. Fit it through the same LOOCV folds, named for the run it is compared with
TC_RUN_SUFFIX=_v2 python main.py --run-models --models 2-stage-GLM --run-name v2

# 3. Rescore every model, the GLM now among them
TC_RUN_SUFFIX=_v2 python src/evaluation/paper_metrics.py

# 4. The paired interval of the difference between the two
TC_RUN_SUFFIX=_v2 python src/evaluation/model_comparison.py \
    --a 2-stage-XGBoost_global_adm1 --b 2-stage-GLM_global_adm1
```

> **`--run-name` and `TC_RUN_SUFFIX` have to agree.** `paper_metrics` appends
> the suffix to *every* `2-stage-` folder, the GLM's included. A GLM run left
> unnamed is therefore not found under `_v2`, and the comparison quietly falls
> back to whichever XGBoost output carries no suffix — a different run from the
> one being reported. Name the GLM run after the run it is compared against.

Step 4 is the one that answers whether the two differ. Each model's own interval
in the table does not: both are scored on the same storms, so their errors are
correlated and the difference is far better determined than either score. The
paired bootstrap resamples cyclones once per replicate and scores both models on
the same resampled set, so `ci_low`/`ci_high` bracket the difference itself.

### 4. Validation Metrics with Confidence Intervals
```bash
# Point estimates only (as before)
python -m src.evaluation.metrics

# Add event-level bootstrap confidence intervals
python -m src.evaluation.metrics --bootstrap --n-boot 1000
```

`src/evaluation/bootstrap.py` attaches nonparametric 95% CIs to every validation metric by resampling **whole cyclones** (`--event-col` defaults to `sid`; pass `--event-col "DisNo."` only to reproduce the round-1 country-event unit), not ADM1 rows. That distinction is what makes the intervals meaningful: ADM1 rows from one storm share a hazard field and a single EM-DAT report, so they are not independent observations — resampling rows treats correlated data as independent and yields intervals that are far too narrow. On representative data the event-level interval comes out roughly **1.8× wider** than a row-level one, and the wider number is the honest one.

Undefined metrics are treated differently in the point estimate and in the replicates, on purpose: the point estimate scores an undefined ratio as 0.0 (matching `sklearn`'s `zero_division=0`, so headline numbers stay conventional), while a replicate where the metric is undefined is excluded from the percentile and counted in `<metric>_n_undefined_boot`. Check that column before quoting an interval — one built on few valid replicates says little. Intervals are percentile-method, so not bias- or skew-corrected (no BCa).

### 5. Model Training & Interpretability
```bash
# Run 2-Stage XGBoost with leave-one-out cross-validation.
# Uses data/model_hyperparameters.json when present, defaults otherwise.
python main.py --run-models

# Also score the three reference baselines, for comparison
python main.py --run-models --with-baselines

# Other strategies, each to its own folder
python main.py --run-models --strategies global walk_forward geo_constrained

# Fit the administrative units directly instead of aggregating grid predictions
python main.py --run-models --train-at adm1

# Leave a feature out, for an ablation
python main.py --run-models --exclude-features N_events_5_years

# The two-part GLM baseline, through the same folds (or both models in one go)
python main.py --run-models --models 2-stage-GLM
python main.py --run-models --models 2-stage-XGBoost 2-stage-GLM

# Generate SHAP Summary and Dependence Plots for the trained model
python main.py --run-interpretability
```

**The unit predicted and the unit withheld are separate.** Each fold predicts one
EM-DAT country-event record, which is the level impact is reported and evaluated
at, while its training set drops every record of the cyclone under test. A storm
making landfall in several countries is one record per country, all describing
the same hazard field, so withholding only the record under test would leave its
siblings in training. `--predict-unit` and `--withhold-unit` change either; a pair
other than the default writes to its own folder, so two definitions never mix.

A run can be named with `--run-name`, which keeps it apart from an earlier one,
and `--recompute` replaces existing per-fold predictions rather than resuming.

### 6. Historical Forecast Study (optional)

What the model would have said *before* each storm arrived, rather than from the
track as later reconstructed.

```bash
# 1. Download what the archive holds for the evaluated events
python main.py --collect-forecasts --forecast-provider rksl

# 2. Predict impact from every ensemble member of every pre-arrival cycle
python main.py --run-forecast-study --forecast-provider rksl

# Follow each member's track further ahead of its own issuance (default 120 h)
python main.py --run-forecast-study --forecast-lead-hours 72
```

Each archived cycle is run through the model as though it were the present: the
ensemble's forecast wind fields replace the observed one over the same grid
cells, the storm is withheld from training as in any other fold, and the
prediction is dated by how long before the storm's arrival the cycle was issued.

**Only the wind is forecast.** Exposure and terrain do not depend on it, but
rainfall does, and a forecast rainfall field would have to come from the
ensemble's gridded output, which this study does not use — so the observed
rainfall stands in. Predictions are therefore better informed about rain than a
forecaster would have been, and wind is the only quantity whose uncertainty is
represented. Lead time is measured from the **observed** best-track landfall,
or from the closest approach where the storm never crosses the coast.

Collection and prediction are both resumable, and an event reaches its final
filename only once it is whole, so an interrupted run leaves nothing a resume
would mistake for finished. A large event spans tens of millions of rows, so it
is predicted and written one cycle at a time; `src/evaluation/forecast_metrics.py`
reads it back the same way.

---

## Feature Dictionary

The pipeline engineers the following features for every 0.1° grid cell:

| Category | Features | Data Source |
| :--- | :--- | :--- |
| **Physical Hazard** | Max Wind Speed, 24h Max Rainfall, Storm Surge, Flood Risk | IBTrACS, NASA PPS, COAST-RP, GLOFAS |
| **Exposure** | Total Population (Grid-level), interpolated to the event year | WorldPop anchors 2000–2020 (post-2020 extrapolated) |
| **Vulnerability** | Urban/Rural/Water Proportions (interpolated to the event year), SHDI (Subnational HDI) | GHSL SMOD epochs 2000–P2025, Global Data Lab |
| **Terrain** | Elevation, Slope, Ruggedness | SRTM (CGIAR-CSI) |
| **Geography** | Distance to Coast, Coastline Length, Landslide Risk | GADM, World Bank |
| **History** | N Events in Last 5 Years | IBTrACS / EM-DAT |

### 7. Figures and Tables

```bash
# Score every model, with cyclone-clustered intervals -> data/output/metrics<suffix>/
TC_RUN_SUFFIX=_v2 python src/evaluation/paper_metrics.py
TC_RUN_SUFFIX=_v2 python src/evaluation/basin_metrics.py

# Paired cyclone bootstrap of a difference between two models (default:
# 2-stage XGBoost minus the two-part GLM)
TC_RUN_SUFFIX=_v2 python src/evaluation/model_comparison.py

# Then the notebooks, which only read what those wrote
TC_RUN_SUFFIX=_v2 jupyter lab visualization/
```

The forecast study is the one exception to "the notebooks only read": its
hundreds of millions of grid predictions are reduced to one row per (ADM1 unit,
forecast cycle) the first time `visualization/forecast_study.ipynb` runs, and
cached beside the predictions as `adm1_by_cycle.parquet`. Delete that file to
rebuild it. The reduction itself lives in `src/evaluation/forecast_metrics.py`,
not in the notebook.

The notebooks in `visualization/` produce the figures; `visualization/README.md`
says which notebook makes what. A run is selected with `TC_RUN_SUFFIX`, and its
figures are written to `visualization/figs<suffix>/`, which git ignores: the
notebooks rewrite every figure from the model output, so versioning them records
churn rather than history.

---

## Methodology Summary

### 1. Spatial Impact Expansion
To accurately model sub-national variation, `process_emdat.py` parses reported administrative units from EM-DAT and expands them into a full country-event grid. Regions explicitly reported by EM-DAT receive the `Total Affected` value, while all other regions in the country are zero-filled. This creates a contrast baseline that allows the model to learn the specific spatial drivers of impact.

### 2. Two-Stage XGBoost
Because spatial impact data is heavily zero-inflated (many cells have no impact), we utilize a two-stage approach:
1.  **Stage 1 (Classification)**: Predicts the binary probability of any population being affected within a 0.1° grid cell.
2.  **Stage 2 (Regression)**: Predicts the magnitude of the affected population for cells classified as impacted in Stage 1.

Results are reported over three impact categories: no impact at exactly zero,
low impact below 15% of a unit's population, and high impact at **15% or more**.
The boundary belongs to the band above it; the zero cut is strict, since a unit
reported at zero is not affected. Both rules come from one definition in
`src/evaluation/aggregation.py`, which every metric and figure uses. Alongside
the binary metrics, each model is scored on mean and root-mean-square error and
on quadratic weighted kappa over those categories, all with 95% intervals from
resampling whole cyclones.

### 3. Statistical Baseline
A two-part generalized linear model with the same structure: logistic regression for whether a cell is affected, then fractional logistic regression (Papke and Wooldridge, 1996) for the affected fraction, zero wherever the first part does not flag the cell. Heavy-tailed predictors enter as `log1p`, and missing values are imputed and predictors standardised inside each fold. Since only the learning algorithm differs, its comparison with the XGBoost model separates the algorithm's contribution from that of the predictors; `model_comparison.py` gives the paired interval of the difference.

### 4. Historical Forecast Study
Skill is also read against how early it could have been had. Archived TIGGE
ensemble tracks are parsed from CXML, matched to the storm by name, and turned
into one wind field per ensemble member on the model's own 0.1° grid with the
same CLIMADA configuration the historical features use — so a forecast wind and
an observed wind reaching the model mean the same thing. Every member of every
pre-arrival cycle is predicted, and the results are scored at ADM1 level in bins
of lead time, against both impact thresholds and against the windspeed-exposed
baseline evaluated on the same forecasts.

The ensemble is summarised in a fixed order of operations, which differs between
the two quantities deliberately. The **central estimate** takes the median
across members at grid level and aggregates that field to ADM1. The **band**
aggregates each member to ADM1 first and takes the quartiles across members,
because a sum of per-cell quantiles is not a quantile of the sum: summing each
cell's lower quartile describes a field no member forecast, and understates a
unit's total whenever members place their impact in different cells.

A lead bin is left unscored where too little supports it (`MIN_EVENTS`,
`MIN_POSITIVES`). The archive thins with lead time, and in the longest bins
precision reaches 1.0 on a handful of calls while recall falls to 0 because
nothing was called at all — both read as skill on a figure, and neither is.

### 5. Model Interpretability
Interpretability is provided via **SHAP (SHapley Additive exPlanations)**. This allows the model to move beyond "black box" predictions, quantifying how individual hazards (e.g., wind speed) interact with local vulnerability (e.g., SHDI) to drive the final affected population counts.

