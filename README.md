# Global Tropical Cyclone Impact Model (0.1° Grid)

This repository provides a high-performance pipeline for generating a high-resolution (**0.1° resolution**, approx. 11km at the equator) global dataset of Tropical Cyclone (TC) impacts. It includes a **Two-Stage XGBoost** model designed to predict sub-national **affected populations** by integrating physical hazards with socioeconomic vulnerability and settlement morphology.

The core of this project is the **Data Factory**, which automates the collection and processing of 16 spatial and dynamic features, anchored strictly to verified [EM-DAT](https://www.emdat.be/) impact records, into a standardized `training_dataset.parquet`.

---

## 📂 Repository Structure

```text
├── data/
│   ├── input/                # Raw downloads (GADM, SRTM, WorldPop, EM-DAT, etc.)
│   └── output/               # Processed feature CSVs and Model results
├── src/
│   ├── collectors/           # Data acquisition (general_collector.py, pps_collector.py)
│   ├── static_features/      # Spatial Processing (grid_cells.py, process_gadm.py, etc.)
│   ├── dynamic_features/     # Event-Based Processing (process_emdat.py, process_wind.py, etc.)
│   ├── models/               # Two-Stage XGBoost & Baselines (train.py)
│   ├── evaluation/           # LOOCV Pipeline & Metrics
│   ├── interpretability/     # SHAP Analysis & Visualization
│   ├── config.py             # Global constants, URLs, and ISO3 List
│   └── dataset_builder.py    # Master script to compile the final Parquet
├── test/                     # Hyperparameter search (hyperparameter_search.py)
├── main.py                   # Unified CLI Entry Point
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠 Prerequisites & Data Preparation

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


### 2. Python Environment
Install the required Python libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Note**: Install GDAL first, then the requirements.txt pip list

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

    > ⚠️ **The two products are not interchangeable without their conversion.** The daily granule stores a *rate*, so using it as if it were an accumulation under-counts rainfall by a factor of 24. Each product therefore has its own conversion function in `process_rain_features.py` (`half_hour_raw_to_mm`, `daily_raw_to_mm`) and they are never mixed. Both report short days rather than silently summing a partial one, since the feature is a maximum over days and an incomplete day can only bias it downward.

* **SHDI Index (Vulnerability)**: Download this dataset manually from *https://globaldatalab.org/shdi/download/shdi/* and put it in under `/data/SHDI/GDL-Subnational-HDI-data.csv` (requires logging to GlobalDataLab)

---

## 🚀 Execution Workflow

The `main.py` script manages the end-to-end workflow. By default, it operates as a Data Factory.

### 1. Build the Dataset (Full Pipeline)
Downloads all public data, initializes the 0.1° grid, processes all 16 features, performs sub-national spatial expansion, and compiles the final dataset.
```bash
python main.py
```

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

# 5. Assembly the final training_dataset.parquet
python main.py --stage build
```

### 3. Hyperparameter Search (optional, but do it before reporting results)
```bash
# Joint random search over both stages; writes data/model_hyperparameters.json
python test/hyperparameter_search.py --n-iter 80 --n-events 200

# Exhaustive search over a deliberately small grid instead
python test/hyperparameter_search.py --search grid

# Tiny budget, just to check the plumbing runs
python test/hyperparameter_search.py --dry-run
```

`test/hyperparameter_search.py` tunes **both stages at once** — the stage-1 classifier, the stage-2 regressor, the two class-balancing ratios (`u1`, `u2`) and the stage-1 decision threshold (`clf_threshold`). Tuning the stages separately would miss their interaction: how aggressively stage 1 flags cells decides which rows stage 2 ever sees.

Three properties are worth knowing, because they decide whether the reported numbers mean anything:

* **Folds are grouped by event (`DisNo.`), never by grid cell.** Cells within one cyclone are strongly correlated, so a row-wise split would put near-duplicates on both sides and report a score the model cannot reproduce on an unseen storm.
* **A held-out set of events is scored exactly once**, after selection. The cross-validated score is optimistic by construction — it is the quantity that was optimised — so the held-out number is the one to quote.
* **Metrics are computed at ADM1 level on pooled out-of-fold predictions**, where impact is actually reported, at both the "affected at all" (0%) and "highly affected" (15%) thresholds.

The winner is written to `data/model_hyperparameters.json` and **loaded automatically** by `TwoStageXGBoost` — no copying numbers by hand. If the file is absent the model falls back to its built-in defaults, so a fresh clone still runs; explicit constructor arguments always win over the file. Override the location with `TC_IMPACT_HYPERPARAMETERS`.

> The development subset in `ISO3_LIST` (ATG/FJI/HTI) is far too small for a search — every event fails the selection filters. Point `--input` at the full dataset.

### 4. Model Training & Interpretability
```bash
# Run 2-Stage XGBoost with Leave-One-Event-Out Cross-Validation (LOOCV)
# Uses data/model_hyperparameters.json when present, defaults otherwise.
python main.py --run-models

# Generate SHAP Summary and Dependence Plots for the trained model
python main.py --run-interpretability
```

---

## 📊 Feature Dictionary

The pipeline engineers the following features for every 0.1° grid cell:

| Category | Features | Data Source |
| :--- | :--- | :--- |
| **Physical Hazard** | Max Wind Speed, 24h Max Rainfall, Storm Surge, Flood Risk | IBTrACS, NASA PPS, COAST-RP, GLOFAS |
| **Exposure** | Total Population (Grid-level) | WorldPop (2020 UN-Adjusted) |
| **Vulnerability** | Urban/Rural/Water Proportions, SHDI (Subnational HDI) | JRC (GHSL), Global Data Lab |
| **Terrain** | Elevation, Slope, Ruggedness | SRTM (CGIAR-CSI) |
| **Geography** | Distance to Coast, Coastline Length, Landslide Risk | GADM, World Bank |
| **History** | N Events in Last 5 Years | IBTrACS / EM-DAT |

---

## 📝 Methodology Summary

### 1. Spatial Impact Expansion
To accurately model sub-national variation, `process_emdat.py` parses reported administrative units from EM-DAT and expands them into a full country-event grid. Regions explicitly reported by EM-DAT receive the `Total Affected` value, while all other regions in the country are zero-filled. This creates a contrast baseline that allows the model to learn the specific spatial drivers of impact.

### 2. Two-Stage XGBoost
Because spatial impact data is heavily zero-inflated (many cells have no impact), we utilize a two-stage approach:
1.  **Stage 1 (Classification)**: Predicts the binary probability of any population being affected within a 0.1° grid cell.
2.  **Stage 2 (Regression)**: Predicts the magnitude of the affected population for cells classified as impacted in Stage 1.

### 3. Model Interpretability
Interpretability is provided via **SHAP (SHapley Additive exPlanations)**. This allows the model to move beyond "black box" predictions, quantifying how individual hazards (e.g., wind speed) interact with local vulnerability (e.g., SHDI) to drive the final affected population counts.

---

## 🎓 Citation

If you use this repository or the generated dataset in your research, please cite:
> *Moss, F et al. Global Sub-national Impact-based Forecasting for Tropical Cyclones Using Open Data: Combining Machine Learning and Exposure-based Approaches, EGUsphere (2026). doi: 10.5194/egusphere-2026-1996 (under review).*