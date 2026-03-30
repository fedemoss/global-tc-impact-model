#!/usr/bin/env python3
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from rasterstats import zonal_stats
from shapely.geometry import Polygon


# Define a function to adjust the longitude of a single polygon
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)

    # Adjust longitudes from [0, 360) to [-180, 180)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)

    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


# grid to grid transformed and CRS of src to transformed-grid CRS
def adjust_data(grid, src):
    grid_transformed = grid.copy()

    # Apply the adjust_longitude function to each geometry in the DataFrame
    grid_transformed["geometry"] = grid_transformed["geometry"].apply(
        adjust_longitude
    )
    grid_transformed = grid_transformed[
        ["id", "iso3", "Latitude", "Longitude", "geometry"]
    ]
    grid_transformed["Centroid"] = grid_transformed.centroid

    # Reproject raster to match grid CRS
    src_wgs84 = src.rio.reproject(grid_transformed.crs)
    return grid_transformed, src_wgs84


def calculate_urban_rural_water(grid, raster, out_dir, iso3, nodata_value=128):
    # Define the file path where the output will be saved
    file_path = f"{out_dir}/degree_of_urbanization_{iso3}.csv"

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File already exists for {iso3}, skipping computation.")
        return

    # Clip raster to grid extent (reduces data size)
    smod_raster_wgs84_clip = raster.rio.clip_box(*grid.total_bounds)

    # Mask out the nodata values in the raster
    smod_raster_wgs84_clip = smod_raster_wgs84_clip.where(
        smod_raster_wgs84_clip != nodata_value
    )

    # Compute zonal statistics for urban, rural, and water categories
    stats = zonal_stats(
        grid["geometry"],
        smod_raster_wgs84_clip.values[0],
        affine=smod_raster_wgs84_clip.rio.transform(),
        stats="count",
        categorical=True,
    )

    # Convert results to a DataFrame
    smod_grid_vals = pd.DataFrame(stats).fillna(0)

    # Map raster classes to urban, rural, and water
    smod_grid_vals["urban"] = (
        smod_grid_vals.get(21, 0)
        + smod_grid_vals.get(22, 0)
        + smod_grid_vals.get(23, 0)
    ) / smod_grid_vals.sum(axis=1)

    smod_grid_vals["rural"] = (
        smod_grid_vals.get(11, 0)
        + smod_grid_vals.get(12, 0)
        + smod_grid_vals.get(13, 0)
    ) / smod_grid_vals.sum(axis=1)

    smod_grid_vals["water"] = (smod_grid_vals.get(10, 0)) / smod_grid_vals.sum(
        axis=1
    )

    # Add IDs
    smod_grid_vals["id"] = grid["id"].values

    # Merge back with the grid
    df_urban_rural = smod_grid_vals[["id", "urban", "rural", "water"]]

    # Save the result to a CSV file
    df_urban_rural.to_csv(file_path, index=False)
    print(f"Processed {iso3} and saved to {file_path}")


def process_country(
    iso3, grid_transformed, src_wgs84, out_dir, nodata_value=128
):
    print(f"Processing country: {iso3}")
    # Filter grid for the specific country
    country_grid = grid_transformed[grid_transformed.iso3 == iso3]
    # Call the calculation function
    calculate_urban_rural_water(
        country_grid, src_wgs84, out_dir, iso3, nodata_value
    )


# def process_all_countries(grid_transformed, src_wgs84, out_dir, iso3_list, nodata_value=128):
#     # Prepare output directory if it does not exist
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     # Process each country in parallel
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         # Pass all necessary arguments to the process_country function
#         executor.map(lambda iso3: process_country(iso3, grid_transformed, src_wgs84, out_dir, nodata_value), iso3_list)


def process_all_countries(
    grid_transformed, src_wgs84, out_dir, iso3_list, nodata_value=128
):
    # Prepare output directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Process each country one by one
    for iso3 in iso3_list:
        process_country(
            iso3, grid_transformed, src_wgs84, out_dir, nodata_value
        )


if __name__ == "__main__":
    # Open dataset
    file_name = "GHS_SMOD_P2025_GLOBE_R2022A_54009_1000_V1_0.tif"
    src = rxr.open_rasterio(f"/data/big/fmoss/data/JRC/{file_name}")

    # Load grid cells
    grid_training = gpd.read_file(
        "/home/fmoss/GLOBAL MODEL/data/GRID/global_0.1_degree_grid_land_overlap.gpkg"
    )
    grid_world = gpd.read_file(
        "/home/fmoss/GLOBAL MODEL/data/GRID/other_countries_grid.gpkg"
    )
    global_grid = pd.concat([grid_training, grid_world])

    # Transform data
    grid_transformed, src_wgs84 = adjust_data(global_grid, src)

    # Out dir
    out_dir = "/data/big/fmoss/data/JRC/grid_data/"
    iso3_list = grid_transformed.iso3.unique()

    process_all_countries(grid_transformed, src_wgs84, out_dir, iso3_list)
