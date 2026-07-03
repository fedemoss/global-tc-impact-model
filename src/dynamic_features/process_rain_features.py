import datetime as dt
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from shapely.geometry import Polygon

from src.config import INPUT_DIR, OUTPUT_DIR, resolve_iso3_list

def _load_dotenv():
    """Load KEY=VALUE pairs from .env at the project root into os.environ (no-op if already set)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

# -------------------------------------------------------------------
# Configuration & Logging
# -------------------------------------------------------------------
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "process_rain.log"),
        logging.StreamHandler()
    ]
)

IMERG_SCALE_FACTOR = 10  # Stored value = mm * 10 (see TIFFTAG_IMAGEDESCRIPTION)
IMERG_FILL_VALUE = 29999

def get_date_list(df_meta, sid, days_to_landfall=2):
    metadata = df_meta.loc[df_meta.sid == sid].copy()
    metadata.loc[:, "landfalldate"] = pd.to_datetime(metadata["landfalldate"])
    start_date = metadata["landfalldate"] - dt.timedelta(days=days_to_landfall)
    end_date = metadata["landfalldate"] + dt.timedelta(days=days_to_landfall)
    return pd.date_range(start_date.iloc[0], end_date.iloc[0]).strftime("%Y%m%d").tolist()

def adjust_longitude(polygon):
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)

def _extract_grid_values(da_in, grid):
    """Read one pixel value (converted to mm) per grid cell, masking the IMERG fill value."""
    values = {}
    for _, row in grid.iterrows():
        minx, miny, maxx, maxy = row["bbox"]
        da_box = da_in.sel(x=slice(minx, maxx), y=slice(maxy, miny))
        if da_box.size > 0:
            raw_value = float(da_box.values[0, 0])
            if raw_value < IMERG_FILL_VALUE:
                values[row["id"]] = raw_value / IMERG_SCALE_FACTOR
    return values

def create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name):
    """
    Reads local half-hourly IMERG accumulation TIFFs downloaded by the PPS collector,
    sums them into a daily accumulated total (mm) per grid cell, and takes the max
    daily total across the +/- days_to_landfall window.
    """
    date_list = get_date_list(df_meta=df_meta, sid=sid, days_to_landfall=2)
    local_gpm_dir = INPUT_DIR / "gpm_data" / typhoon_name

    grid = grid_global[grid_global.iso3 == iso3].copy()
    grid["bbox"] = grid.geometry.apply(lambda geom: geom.bounds)

    file_df = pd.DataFrame()
    for date_str in date_list:
        if not local_gpm_dir.exists():
            continue

        day_files = sorted(local_gpm_dir.glob(f"*{date_str}*.tif"))
        if not day_files:
            continue

        daily_totals = pd.Series(0.0, index=grid["id"])
        daily_has_data = pd.Series(False, index=grid["id"])
        for file_path in day_files:
            da_in = rxr.open_rasterio(file_path, masked=True, chunks=True)
            da_in = da_in.rio.write_crs(4326).squeeze(drop=True)
            grid = grid.to_crs(da_in.rio.crs)

            for cell_id, accum_mm in _extract_grid_values(da_in, grid).items():
                daily_totals[cell_id] += accum_mm
                daily_has_data[cell_id] = True

        day_grid = grid[["id", "iso3"]].copy()
        day_grid["mean"] = daily_totals.reindex(grid["id"]).where(daily_has_data.reindex(grid["id"])).values
        day_grid["date"] = date_str
        file_df = pd.concat([file_df, day_grid], axis=0)

    if file_df.empty:
        raise FileNotFoundError(f"No local GPM data found for {typhoon_name} ({sid}) in {local_gpm_dir}")

    day_wide = pd.pivot(file_df, index=["id", "iso3"], columns=["date"], values=["mean"])
    day_wide.columns = day_wide.columns.droplevel(0)
    day_wide.reset_index(inplace=True)
    day_wide["rainfall_max_24h"] = day_wide.iloc[:, 2:].max(axis=1)
    day_wide["sid"] = sid

    return day_wide[["id", "iso3", "sid", "rainfall_max_24h"]]

def _process_storm(args):
    iso3, sid, typhoon_name, metadata_country, grid_global = args
    df_meta = metadata_country[metadata_country.sid == sid]
    try:
        df_rainfall = create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name)
        return df_rainfall.fillna(0), None
    except Exception as e:
        logging.error(f"Failed to process {iso3}, {sid}: {e}")
        return None, pd.DataFrame([{"iso3": iso3, "sid": sid}])

def process_country_rainfall(iso3, metadata_global, grid_global, out_dir):
    df_rainfall_total = []
    not_working_cases = []
    
    out_file = out_dir / f"rainfall_data_{iso3}.csv"
    if out_file.exists():
        logging.info(f"Skipping {iso3}: file already exists")
        return

    metadata_country = metadata_global[metadata_global.iso3 == iso3]
    if not metadata_country.empty:
        with ThreadPoolExecutor(max_workers=10) as executor:
            args_list = [
                (iso3, row.sid, row.typhoon, metadata_country, grid_global) 
                for _, row in metadata_country.drop_duplicates('sid').iterrows()
            ]
            results = executor.map(_process_storm, args_list)

        for df_rainfall, not_working_case in results:
            if df_rainfall is not None:
                df_rainfall_total.append(df_rainfall)
            if not_working_case is not None:
                not_working_cases.append(not_working_case)

        if not_working_cases:
            pd.concat(not_working_cases).to_csv(out_dir / f"nodata_rainfall_{iso3}.csv", mode="a", header=False, index=False)

        if df_rainfall_total:
            pd.concat(df_rainfall_total).to_csv(out_file, index=False)

def _load_metadata_global():
    """Load and concatenate all per-country metadata files from the IBTRACS output directory."""
    meta_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    meta_files = list(meta_dir.glob("metadata_*.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files found in {meta_dir}. Run process_wind_features first.")
    df = pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True)
    df = df.drop(columns=["DisNo."], errors="ignore").drop_duplicates()
    df["iso3"] = df["GID_0"]
    return df

def _load_grid_global():
    """Load the land-overlap grid and apply longitude adjustments."""
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["iso3"] = grid.GID_0
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    return grid

def run_single_storm(iso3, sid):
    """Download GPM data (if absent) and compute rainfall features for one storm."""
    from src.collectors.pps_collector import download_gpm_late_run

    meta_file = OUTPUT_DIR / "IBTRACS" / "standard" / f"metadata_{iso3}.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found for {iso3}. Run process_wind_features first.")

    df_meta = pd.read_csv(meta_file)
    storm = df_meta[df_meta.sid == sid]
    if storm.empty:
        raise ValueError(f"SID {sid} not found in metadata for {iso3}.")

    typhoon_name = storm.iloc[0]["typhoon"]
    date_list = get_date_list(df_meta, sid, days_to_landfall=2)

    logging.info(f"Downloading GPM data for {typhoon_name} ({date_list[0]} – {date_list[-1]})...")
    download_gpm_late_run(
        start_date=pd.to_datetime(date_list[0]),
        end_date=pd.to_datetime(date_list[-1]),
        typhoon_name=typhoon_name,
    )

    grid_global = _load_grid_global()
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_rainfall = create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name)
    df_rainfall = df_rainfall.fillna(0)
    out_file = out_dir / f"rainfall_data_{iso3}_{sid}.csv"
    df_rainfall.to_csv(out_file, index=False)
    logging.info(f"Saved: {out_file}")
    return df_rainfall

def _ensure_local_gpm_data(iso3, metadata_country):
    """
    Downloads any storm's GPM data that isn't already present locally.
    Sequential by design: NASA PPS isn't built for many parallel download
    sessions (single requests.Session, no cross-call rate-limit coordination
    in pps_collector.py) — unlike the local-only processing below, which is
    safe to run concurrently once the data is on disk.
    """
    from src.collectors.pps_collector import download_gpm_late_run

    for _, row in metadata_country.drop_duplicates("sid").iterrows():
        date_list = get_date_list(df_meta=metadata_country, sid=row.sid, days_to_landfall=2)
        local_gpm_dir = INPUT_DIR / "gpm_data" / row.typhoon
        already_local = local_gpm_dir.exists() and any(local_gpm_dir.glob(f"*{d}*.tif") for d in date_list)
        if already_local:
            continue
        try:
            logging.info(f"Downloading GPM data for {iso3}, {row.typhoon} ({row.sid})...")
            download_gpm_late_run(
                start_date=pd.to_datetime(date_list[0]),
                end_date=pd.to_datetime(date_list[-1]),
                typhoon_name=row.typhoon,
            )
        except Exception as e:
            logging.error(f"Failed to download GPM data for {iso3}, {row.sid} ({row.typhoon}): {e}")

def generate_all_rain_features(max_workers=4):
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading global grid and applying longitude adjustments...")
    grid_global = _load_grid_global()

    print("Loading global metadata...")
    metadata_global = _load_metadata_global()

    valid_iso3_list = [iso3 for iso3 in resolve_iso3_list() if iso3 in metadata_global["iso3"].unique()]

    print("Ensuring local GPM data is available for all storms (downloading missing storms)...")
    for iso3 in valid_iso3_list:
        if (out_dir / f"rainfall_data_{iso3}.csv").exists():
            continue  # already processed, no need to (re)download
        _ensure_local_gpm_data(iso3, metadata_global[metadata_global.iso3 == iso3])

    print(f"Starting rainfall processing for {len(valid_iso3_list)} countries...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_country_rainfall, iso3, metadata_global, grid_global, out_dir): iso3
            for iso3 in valid_iso3_list
        }
        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()
                print(f"Successfully processed rainfall for {iso3}")
            except Exception as e:
                print(f"Error processing {iso3}: {e}")

if __name__ == "__main__":
    run_single_storm(iso3="ATG", sid="2008287N15291")