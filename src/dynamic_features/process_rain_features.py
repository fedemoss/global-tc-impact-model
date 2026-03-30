import datetime as dt
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from shapely.geometry import Polygon

from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

logging.basicConfig(
    filename="rainfall_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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

def create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name):
    """
    Reads local TIFF files downloaded by the PPS collector to generate grid rainfall features.
    """
    date_list = get_date_list(df_meta=df_meta, sid=sid, days_to_landfall=2)
    local_gpm_dir = INPUT_DIR / "gpm_data" / typhoon_name
    
    raster_files = []
    found_dates = []
    
    for date_str in date_list:
        if not local_gpm_dir.exists():
            continue
            
        for file_path in local_gpm_dir.glob(f"*{date_str}*.tif"):
            raster_file = rxr.open_rasterio(file_path, masked=True, chunks=True)
            raster_file = raster_file.rio.write_crs(4326).squeeze(drop=True)
            raster_files.append(raster_file)
            found_dates.append(date_str)

    if not raster_files:
        raise FileNotFoundError(f"No local GPM data found for {typhoon_name} ({sid}) in {local_gpm_dir}")

    grid = grid_global[grid_global.iso3 == iso3].copy()
    grid["bbox"] = grid.geometry.apply(lambda geom: geom.bounds)
    
    file_df = pd.DataFrame()
    for i, da_in in enumerate(raster_files):
        da_in = da_in.rio.write_crs(4326)
        grid = grid.to_crs(da_in.rio.crs)
        grid["mean"] = np.nan

        for index, row in grid.iterrows():
            minx, miny, maxx, maxy = row["bbox"]
            da_box = da_in.sel(x=slice(minx, maxx), y=slice(miny, maxy))
            if da_box.size > 0:
                grid.at[index, "mean"] = da_box.values[0, 0]

        grid["date"] = found_dates[i]
        grid["mean"] /= 10  # Convert to mm/hr
        file_df = pd.concat([file_df, grid[["id", "iso3", "mean", "date"]]], axis=0)

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

def generate_all_rain_features(max_workers=4):
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading global grid and applying longitude adjustments...")
    grid_global = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid_global["iso3"] = grid_global.GID_0
    grid_global["geometry"] = grid_global["geometry"].apply(adjust_longitude)

    print("Loading global metadata...")
    metadata_global = pd.read_csv(INPUT_DIR / "IBTRACS" / "merged" / "meta_data.csv")
    metadata_global = metadata_global.drop("DisNo.", axis=1).drop_duplicates()
    metadata_global["iso3"] = metadata_global.GID_0

    # Filter global list to valid countries only
    valid_iso3_list = [iso3 for iso3 in ISO3_LIST if iso3 in metadata_global["iso3"].unique()]
    
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
    generate_all_rain_features(max_workers=4)