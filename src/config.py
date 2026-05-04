import os
from pathlib import Path

# Set the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input and Output directories
INPUT_DIR = Path(os.getenv("TC_IMPACT_INPUT_DIR", BASE_DIR / "data" / "input"))
OUTPUT_DIR = Path(os.getenv("TC_IMPACT_OUTPUT_DIR", BASE_DIR / "data" / "output"))


# Data Source URLs
GADM_BASE_URL = "https://gadm.org/download_world.html"
GAUL_ADM2_URL = "https://storage.googleapis.com/fao-maps-catalog-data/boundaries/GAUL_2024_L2.zip" # Check
WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif"
LANDSLIDE_URL = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/DR0045418/LS_RF_Mean_1980-2018_COG.tif"
STORM_SURGE_URL = "https://data.4tu.nl/file/4e291b8f-a37e-4378-8ca6-954a44fdc8fb/1263247c-4427-40eb-b497-a79f72caa267"
JRC_SMOD_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2022A/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000/V1-0/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000_V1_0.zip"
SRTM_BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
NASA_PPS_BASE_URL = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/"
FLOOD_RISK_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP10/"
SHDI_URL = "https://globaldatalab.org/shdi/download/shdi/?levels=4&interpolation=0&extrapolation=0"

# FEATURES used in the final 2-stage XGBoost model
FEATURES = [
    "wind_speed", "rainfall_max_24h", "population", "coast_length_meters", 
    "with_coast", "mean_elev", "mean_slope", "mean_rug", "urban", 
    "rural", "water", "storm_tide_rp_0010", "landslide_risk_sum", 
    "N_events_5_years"
]


# Non-contemplated features (Discussed in paper but excluded from global training)
NON_CONTEMPLATED_FEATURES = [
    "flood_risk",    # Excluded: Basins < 500km2 missing in 22/72 countries
    "shdi"           # Note: SHDI used where available, but missing in 19/72 countries
    "track_distance" # Excluded as its highly correlated with windspeed 
]

# ISO3 Country List
ISO3_LIST = [
    "ATG", "AUS", "BGD", "CAN", "CHN", "COL", "CRI", "CUB", "DJI", "DOM",
    "FJI", "GLP", "GTM", "HND", "HTI", "IDN", "IND", "JPN", "KHM", "BRA",
    "KOR", "LAO", "LKA", "MDG", "MEX", "MMR", "MOZ", "MTQ", "NCL", "NIC",
    "NZL", "OMN", "PAK", "PAN", "PHL", "PNG", "PRK", "PRT", "SLV", "SOM",
    "THA", "TLS", "TWN", "TZA", "USA", "VEN", "VNM", "VUT", "YEM", "ZAF",
    "IRN", "MWI", "ZWE", "WSM", "TON", "BHS", "SLB", "FSM", "PYF", "BLZ",
    "BRB", "GRD", "MUS"
]
