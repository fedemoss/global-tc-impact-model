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
├── main.py                   # Unified CLI Entry Point
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠 Prerequisites & Data Preparation

### 1. System Dependencies
The pipeline relies on several low-level geospatial libraries for raster processing and atmospheric data handling:
* **Python 3.10+**: The core environment.
* **GDAL (Geospatial Data Abstraction Library)**: Required for processing SRTM elevation data and JRC urbanization rasters (specifically `gdaldem`).



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
    * `Admin Units`: JSON-formatted string of affected administrative regions (used for spatial expansion).

**Note: we leave to the user the "sid" and "Disno." matching of storms. This involves manual labeling based TC names, locations and dates on top of classic fuzzy-matching techniques or (alternatively) the use of LLM matching approaches.** 

* **NASA PPS (Precipitation)**: To access GPM-IMERG rainfall data, register a free account at [NASA PPS](https://pps.gsfc.nasa.gov/). Once registered, configure your credentials in `src/collectors/pps_collector.py`.

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
python main.py --stage dynamic

# 5. Assembly the final training_dataset.parquet
python main.py --stage build
```

### 3. Model Training & Interpretability
```bash
# Run 2-Stage XGBoost with Leave-One-Event-Out Cross-Validation (LOOCV)
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