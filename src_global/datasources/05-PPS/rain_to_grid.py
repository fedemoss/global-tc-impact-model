#!/usr/bin/env python3
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

from src_global.utils import blob

# Set up logging
log_file = "rainfall_processing.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to get date list based on metadata of events
def get_date_list(df_meta, sid, DAYS_TO_LANDFALL=2):
    metadata = df_meta.loc[df_meta.sid == sid].copy()
    # Ensure 'landfalldate' is in datetime format
    metadata.loc[:, "landfalldate"] = pd.to_datetime(metadata["landfalldate"])

    # Calculate start and end dates
    start_date = metadata["landfalldate"] - dt.timedelta(days=DAYS_TO_LANDFALL)
    end_date = metadata["landfalldate"] + dt.timedelta(days=DAYS_TO_LANDFALL)

    # Generate date list in the format Y-m-d
    date_list = (
        pd.date_range(start_date.iloc[0], end_date.iloc[0])
        .strftime("%Y-%m-%d")
        .tolist()
    )

    return date_list


# IMERG v7 database has information until 2003
# IMERG v6 database has information from 2003
def adjust_metadata_to_imerg_dates(
    metadata,
    DAYS_TO_LANDFALL=2,
    low_limit_date="2003-03-11",
    high_limit_date=None,
):
    # Convert landfalldate to datetime format
    metadata["landfalldate_dt"] = pd.to_datetime(metadata["landfalldate"])
    # Start date for gathering data
    metadata["before_landfall"] = metadata["landfalldate_dt"] - dt.timedelta(
        days=DAYS_TO_LANDFALL
    )
    metadata["after_landfall"] = metadata["landfalldate_dt"] + dt.timedelta(
        days=DAYS_TO_LANDFALL
    )

    # Define the threshold date
    thres_date_low = dt.datetime.strptime(low_limit_date, "%Y-%m-%d").date()

    metadata_reduced = metadata[
        metadata["before_landfall"].dt.date >= thres_date_low
    ].reset_index(drop=True)
    # For setting upper thresholds
    if high_limit_date is not None:
        thres_date_high = dt.datetime.strptime(
            high_limit_date, "%Y-%m-%d"
        ).date()
        metadata_reduced = metadata_reduced[
            metadata_reduced["after_landfall"].dt.date <= thres_date_high
        ].reset_index(drop=True)
    return metadata_reduced


# Ajust geometries from countries that falls in the 0-meridian
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)

    # Adjust longitudes from [-180, 180) to [-360, 0)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)

    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


# PPS data to grid level
container_client = blob.prod_container_client


def create_rainfall_dataset(grid_global, df_meta, iso3, sid, version="v6"):
    # Date list
    date_list = get_date_list(df_meta=df_meta, sid=sid, DAYS_TO_LANDFALL=2)

    # Get rasters
    raster_files = []
    for date in date_list:
        blob_name = f"imerg/{version}/imerg-daily-late-{date}.tif"
        blob_client = container_client.get_blob_client(blob_name)
        cog_url = blob_client.url
        # Open raster
        raster_file = rxr.open_rasterio(cog_url, masked=True, chunks=True)
        raster_file = raster_file.rio.write_crs(4326)
        raster_file = raster_file.squeeze(drop=True)
        raster_files.append(raster_file)

    # Call grid and raster files
    grid = grid_global[grid_global.iso3 == iso3]
    grid = gpd.GeoDataFrame(grid, geometry="geometry")
    # Convert grid cells to GeoDataFrame with bounding boxes
    grid["bbox"] = grid.geometry.apply(lambda geom: geom.bounds)

    # For every raster file (date)
    file_df = pd.DataFrame()
    i = 0
    for da_in in raster_files:
        # Write CRS
        da_in = da_in.rio.write_crs(4326)

        # Reproject grid to match raster CRS if needed
        grid = grid.to_crs(da_in.rio.crs)

        # Initialize a column for pixel values
        grid["mean"] = np.nan

        # Extract the value of the raster for each grid cell
        for index, row in grid.iterrows():
            minx, miny, maxx, maxy = row["bbox"]

            # Select the raster data within the bounding box
            da_box = da_in.sel(x=slice(minx, maxx), y=slice(miny, maxy))
            # Assuming there's exactly one pixel per grid cell (there is), get its value
            pixel_value = da_box.values[0, 0]  # Adjust indices if needed
            grid.at[index, "mean"] = pixel_value

        grid["date"] = date_list[i]  # .strftime("%Y-%m-%d")
        # change values by dividing by 10 to mm/hr
        grid["mean"] /= 10
        file_df = pd.concat(
            [file_df, grid[["id", "iso3", "mean", "date"]]], axis=0
        )
        i += 1
    # Date as column, rainfall values as rows with index id, iso3
    day_wide = pd.pivot(
        file_df,
        index=["id", "iso3"],
        columns=["date"],
        values=["mean"],
    )
    # This for the headers 'Mean' and 'id'
    day_wide.columns = day_wide.columns.droplevel(0)
    day_wide.reset_index(inplace=True)
    # Max accumulated rainfall in the period selected
    day_wide["rainfall_max_24h"] = day_wide.iloc[:, 2:].max(axis=1)
    day_wide["sid"] = sid

    return day_wide[["id", "iso3", "sid", "rainfall_max_24h"]]



# # Iterate process
# def process_country_rainfall(
#     iso3, metadata_global, grid_global, out_dir, imerg_version
# ):
#     """Process rainfall dataset for a single country in parallel."""
#     df_rainfall_total = pd.DataFrame()
#     not_working_cases = pd.DataFrame()

#     logging.info(f"Processing country: {iso3}")

#     # Skip if country data already exists
#     if os.path.exists(f"{out_dir}/rainfall_data_{iso3}.csv"):
#         logging.info(f"Skipping {iso3}: file already exists")
#         return

#     metadata_country = metadata_global[metadata_global.iso3 == iso3]
#     if not metadata_country.empty:
#         for sid in metadata_country.sid.unique():
#             logging.info(f"Processing storm ID: {sid}")
#             df_meta = metadata_country[metadata_country.sid == sid]
#             try:
#                 df_rainfall = create_rainfall_dataset(
#                     grid_global=grid_global,
#                     df_meta=df_meta,
#                     iso3=iso3,
#                     sid=sid,
#                     version=imerg_version,
#                 )
#                 df_rainfall = df_rainfall.fillna(0)
#             except Exception as e:
#                 logging.error(f"Failed to process {iso3}, {sid}: {e}")
#                 not_working_case = pd.DataFrame([{"iso3": iso3, "sid": sid}])
#                 not_working_cases = pd.concat(
#                     [not_working_cases, not_working_case]
#                 )
#                 df_rainfall = pd.DataFrame()

#             df_rainfall_total = pd.concat([df_rainfall_total, df_rainfall])

#         if not not_working_cases.empty:
#             not_working_cases.to_csv(
#                 f"{out_dir}/nodata_rainfall_{iso3}.csv",
#                 mode="a",
#                 header=False,
#                 index=False,
#             )

#         df_rainfall_total.to_csv(
#             f"{out_dir}/rainfall_data_{iso3}.csv", index=False
#         )
#     else:
#         logging.info(f"{iso3} not present in metadata")


def _process_storm(args):
    """ Helper function to process a single storm event (sid). """
    iso3, sid, metadata_country, grid_global, imerg_version = args
    df_meta = metadata_country[metadata_country.sid == sid]
    
    try:
        df_rainfall = create_rainfall_dataset(
            grid_global=grid_global,
            df_meta=df_meta,
            iso3=iso3,
            sid=sid,
            version=imerg_version,
        )
        df_rainfall = df_rainfall.fillna(0)
        return df_rainfall, None
    except Exception as e:
        logging.error(f"Failed to process {iso3}, {sid}: {e}")
        return None, pd.DataFrame([{"iso3": iso3, "sid": sid}])

def process_country_rainfall(iso3, metadata_global, grid_global, out_dir, imerg_version):
    """Process rainfall dataset for a single country in parallel."""
    df_rainfall_total = []
    not_working_cases = []

    logging.info(f"Processing country: {iso3}")

    # Skip if country data already exists
    if os.path.exists(f"{out_dir}/rainfall_data_{iso3}.csv"):
        logging.info(f"Skipping {iso3}: file already exists")
        return

    metadata_country = metadata_global[metadata_global.iso3 == iso3]
    if not metadata_country.empty:
        with ThreadPoolExecutor(max_workers=10) as executor:
            args_list = [(iso3, sid, metadata_country, grid_global, imerg_version) for sid in metadata_country.sid.unique()]
            results = executor.map(_process_storm, args_list)

        # Collect results
        for df_rainfall, not_working_case in results:
            if df_rainfall is not None:
                df_rainfall_total.append(df_rainfall)
            if not_working_case is not None:
                not_working_cases.append(not_working_case)

        # Save results
        if not_working_cases:
            pd.concat(not_working_cases).to_csv(
                f"{out_dir}/nodata_rainfall_{iso3}.csv",
                mode="a",
                header=False,
                index=False,
            )

        if df_rainfall_total:
            pd.concat(df_rainfall_total).to_csv(
                f"{out_dir}/rainfall_data_{iso3}.csv", index=False
            )
    else:
        logging.info(f"{iso3} not present in metadata")


def iterate_rainfall_dataset(
    iso3_list,
    metadata_global,
    grid_global,
    out_dir,
    imerg_version="v6",
    max_workers=4,
):
    """Iterate over multiple countries using parallel processing."""
    os.makedirs(out_dir, exist_ok=True)

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:  # Adjust workers as needed
        futures = {
            executor.submit(
                process_country_rainfall,
                iso3,
                metadata_global,
                grid_global,
                out_dir,
                imerg_version,
            ): iso3
            for iso3 in iso3_list
        }

        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()  # Get the result to catch any exceptions
            except Exception as e:
                print(f"Error processing {iso3}: {e}")


if __name__ == "__main__":
    # Set data path and out dir
    data_path = "/data/big/fmoss/data"
    out_dir = f"{data_path}/PPS"
    # Load grid cells
    grid_global = gpd.read_file(
        f"{data_path}/GRID/merged/global_grid_land_overlap.gpkg"
    )
    grid_global.loc[:, "iso3"] = grid_global.GID_0

    # Apply the adjust_longitude function to each geometry in the DataFrame
    grid_global_transformed = grid_global.copy()
    grid_global_transformed["geometry"] = grid_global_transformed[
        "geometry"
    ].apply(adjust_longitude)

    # Load metadata (and clean DisNo. same-events cases)
    # It can happen that 2 events have the same sid and iso3 but different EM-DAT DisNo. code.
    # This is because multiple tracks of a same storms got reported affecitng differnet regions
    # Since the storms is still the same and the date range is also the same, rainfall data is going to be the same
    metadata_global = pd.read_csv(f"{data_path}/IBTRACS/merged/meta_data.csv")
    metadata_global = metadata_global.drop("DisNo.", axis=1).drop_duplicates()
    metadata_global.loc[:, "iso3"] = metadata_global.GID_0

    # country list
    # iso3_list = sorted(metadata_global.iso3.unique(), reverse=False)
    iso3_list = ["CHN", "USA"]
    # IMERG Late v7 only has information until 2003
    metadata_global_reduced = adjust_metadata_to_imerg_dates(
        metadata_global,
        DAYS_TO_LANDFALL=2,
        low_limit_date="2000-06-01",
        high_limit_date="2003-12-31",
    )
    print("Starting iteration for imerg v7")
    iterate_rainfall_dataset(
        iso3_list=iso3_list,
        metadata_global=metadata_global_reduced,
        grid_global=grid_global_transformed,
        imerg_version="v7",
        out_dir=f"{out_dir}/imerg_v7",
        max_workers=5,
    )

    # IMERG Late v6 only has information starting 2003
    metadata_global_reduced = adjust_metadata_to_imerg_dates(
        metadata_global,
        DAYS_TO_LANDFALL=2,
        low_limit_date="2003-03-11",
        high_limit_date=None,
    )
    print("Starting iteration for imerg v6")
    iterate_rainfall_dataset(
        iso3_list=iso3_list,
        metadata_global=metadata_global_reduced,
        grid_global=grid_global_transformed,
        imerg_version="v6",
        out_dir=f"{out_dir}/imerg_v6",
        max_workers=2,
    )
