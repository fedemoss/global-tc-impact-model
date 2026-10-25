import os
from pathlib import Path
import geopandas as gpd

# Set the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Root for everything the pipeline reads or writes. Defaults to `data/` inside
# the repo, so a fresh clone works with no setup. Set TC_IMPACT_DATA_DIR to keep
# the data off a small home partition and out of the working tree -- on mldevbox
# that is /data/shared/fmoss/global-model.
DATA_DIR = Path(os.getenv("TC_IMPACT_DATA_DIR", BASE_DIR / "data"))

# Input and Output directories. Still individually overridable, for the case
# where inputs come from a shared read-only location but outputs do not.
INPUT_DIR = Path(os.getenv("TC_IMPACT_INPUT_DIR", DATA_DIR / "input"))
OUTPUT_DIR = Path(os.getenv("TC_IMPACT_OUTPUT_DIR", DATA_DIR / "output"))


# Data Source URLs
GADM_BASE_URL = "https://gadm.org/download_world.html"
GAUL_ADM2_URL = "https://storage.googleapis.com/fao-maps-catalog-data/boundaries/GAUL_2024_L2.zip"  # direct FAO catalog mirror; if it goes away, export via GEE (see README)
# Population and degree of urbanisation are time dependent: both are collected
# at several anchor years and interpolated to the year of each event, instead of
# freezing one year for the whole 2000-2022 record (see utils/time_interpolation.py).
WORLDPOP_URL_TEMPLATE = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
    "{year}/0_Mosaicked/ppp_{year}_1km_Aggregated.tif"
)
# One WorldPop release throughout, so the series is internally consistent.
# Events after 2020 are linearly extrapolated from the 2015->2020 slope.
POP_ANCHOR_YEARS = [2000, 2005, 2010, 2015, 2020]
LANDSLIDE_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/DR0045418/LS_RF_Mean_1980-2018_COG.tif"
STORM_SURGE_URL = "https://data.4tu.nl/file/4e291b8f-a37e-4378-8ca6-954a44fdc8fb/1263247c-4427-40eb-b497-a79f72caa267"
JRC_SMOD_URL_TEMPLATE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2022A/"
    "GHS_SMOD_{epoch}_GLOBE_R2022A_54009_1000/V1-0/"
    "GHS_SMOD_{epoch}_GLOBE_R2022A_54009_1000_V1_0.zip"
)
# E-epochs are observed, P2025 is GHSL's own projection. Keeping P2025 as the
# upper anchor means post-2020 events are interpolated rather than extrapolated,
# which matters because class fractions must stay inside [0, 1].
SMOD_EPOCH_YEARS = {
    "E2000": 2000,
    "E2005": 2005,
    "E2010": 2010,
    "E2015": 2015,
    "E2020": 2020,
    "P2025": 2025,
}
SRTM_BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
NASA_PPS_BASE_URL = "https://arthurhouhttps.pps.eosdis.nasa.gov/gpmdata/"
FLOOD_RISK_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP10/"
SHDI_URL = "https://globaldatalab.org/shdi/download/shdi/?levels=4&interpolation=0&extrapolation=0"

# --- IMERG rainfall product -----------------------------------------------
# Two routes to the same feature, `rainfall_max_24h` (max daily accumulation in
# mm over the landfall +/- 2 day window). They agree to correlation 0.9994; the
# choice is precision against download size.
#
#   "half_hourly" (default) - the 48 `3B-HHR-GIS ... total.accum` granules of
#       each day, summed. Units are 0.1 mm per half hour, so mm/day is
#       `raw / 10` summed over the 48. Resolution 0.1 mm, ~166 MB per storm.
#
#   "daily" - one `3B-DAY-GIS` granule per date. It is a RATE, not an
#       accumulation ("Unit=0.1(mm/hr) ... MaxPossibleNumHalfHour=48"), so
#       mm/day is `raw / 10 * 24`. Resolution 2.4 mm/day, ~1 MB per date, and
#       it exists on some days where the half-hourly files do not.
#
# Forgetting the x24 on the daily product is precisely how this feature ended up
# under-represented before, so the conversion lives in one function per product
# in process_rain_features.py.
IMERG_PRODUCT = os.getenv("TC_IMPACT_IMERG_PRODUCT", "half_hourly")
IMERG_PRODUCTS = ("half_hourly", "daily")

# Workers for the raster aggregation steps. These run one process per chunk of
# grid cells; lower it on a small machine (each worker holds a raster window).
RASTER_WORKERS = int(os.getenv("TC_IMPACT_RASTER_WORKERS", "8"))
# Grid cells per task. Keeps one large country from becoming the critical path.
RASTER_CHUNK_SIZE = int(os.getenv("TC_IMPACT_RASTER_CHUNK", "10000"))

# Tuned hyperparameters for the two-stage model, written by
# src/models/hyperparameter_search.py and picked up automatically by
# models/two_stage_xgb.py. If the file is absent the model falls back to the
# defaults hardcoded in that class, so a fresh clone still runs.
HYPERPARAMETERS_PATH = Path(
    os.getenv("TC_IMPACT_HYPERPARAMETERS", DATA_DIR / "model_hyperparameters.json")
)

# The same for the two-part GLM baseline, written by src/models/glm_search.py
# and read by models/two_stage_glm.py. Kept apart from the XGBoost file so that
# tuning one model never touches the other's configuration.
GLM_HYPERPARAMETERS_PATH = Path(
    os.getenv("TC_IMPACT_GLM_HYPERPARAMETERS", DATA_DIR / "glm_hyperparameters.json")
)

# FEATURES used in the final 2-stage XGBoost model
FEATURES = [
    "wind_speed", "rainfall_max_24h", "population", "coast_length_meters", 
    "with_coast", "mean_elev", "mean_slope", "mean_rug", "urban", 
    "rural", "water", "storm_tide_rp_0010", "landslide_risk_sum", 
    "N_events_5_years"
]


# Features discussed in the paper but excluded from global training.
# Documentation only: nothing reads this list. Exclusion is by omission from
# FEATURES above, which is the only list the model, the SHAP analysis and the
# hyperparameter search consume.
NON_CONTEMPLATED_FEATURES = [
    "flood_risk",     # Excluded: Basins < 500km2 missing in 22/72 countries
    "shdi",           # Note: SHDI used where available, but missing in 19/72 countries
    "track_distance", # Excluded as its highly correlated with windspeed
]

# ISO3 Country List
# ISO3_LIST = [
#     "ATG", "AUS", "BGD", "CAN", "CHN", "COL", "CRI", "CUB", "DJI", "DOM",
#     "FJI", "GLP", "GTM", "HND", "HTI", "IDN", "IND", "JPN", "KHM", "BRA",
#     "KOR", "LAO", "LKA", "MDG", "MEX", "MMR", "MOZ", "MTQ", "NCL", "NIC",
#     "NZL", "OMN", "PAK", "PAN", "PHL", "PNG", "PRK", "PRT", "SLV", "SOM",
#     "THA", "TLS", "TWN", "TZA", "USA", "VEN", "VNM", "VUT", "YEM", "ZAF",
#     "IRN", "MWI", "ZWE", "WSM", "TON", "BHS", "SLB", "FSM", "PYF", "BLZ",
#     "BRB", "GRD", "MUS"
# ]

# Territories dropped even at global scope. Antarctica accounts for roughly a
# quarter of the cells in the 0.1 degree land grid and is outside the range of
# tropical cyclones, so it contributes cost but no signal. Applied by
# resolve_iso3_list() and by the processors that read the grid directly.
EXCLUDED_ISO3 = {"ATA"}

# None means global scope: resolve_iso3_list() below falls back to every
# GID_0 in GADM. Set a list here to restrict the run to a few countries.
# ISO3_LIST = ["ATG", "FJI", "HTI"]
ISO3_LIST = None

# --- Output paths for the time-dependent exposure layers -------------------
# Kept here rather than in the processing modules so that dataset_builder can
# locate them without importing the GDAL/rasterstats stack.

def population_grid_path(iso3, year):
    """Per-country WorldPop aggregation for one anchor year."""
    return OUTPUT_DIR / "Worldpop" / "grid_data" / f"population_grid_{iso3}_{year}.csv"


def urbanization_grid_path(iso3, epoch):
    """Per-country GHS-SMOD aggregation for one anchor epoch (e.g. "E2010")."""
    return OUTPUT_DIR / "JRC" / "grid_data" / f"degree_of_urbanization_{iso3}_{epoch}.csv"


def resolve_iso3_list():
    """Return ISO3_LIST if set, otherwise all GID_0 codes from GADM.

    EXCLUDED_ISO3 is applied in both cases, so an explicit list cannot
    accidentally reintroduce a territory the model never wants.
    """
    if ISO3_LIST is not None:
        return [iso for iso in ISO3_LIST if iso not in EXCLUDED_ISO3]
    gadm_path = INPUT_DIR / "SHP" / "gadm_410.gdb"
    world = gpd.read_file(gadm_path, ignore_geometry=True)
    codes = world["GID_0"].dropna().unique().tolist()
    return sorted(c for c in codes if c not in EXCLUDED_ISO3)
