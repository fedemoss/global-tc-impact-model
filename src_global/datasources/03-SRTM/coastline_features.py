#!/usr/bin/env python3
import os
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
from shapely.geometry import Point, Polygon


def rasterize_coastline(coastline, output_tif, resolution=0.01):
    """
    Rasterizes a coastline GeoDataFrame and saves it as a TIFF file.

    Parameters:
        coastline (GeoDataFrame): A GeoDataFrame containing LINESTRING geometries representing coastlines.
        output_tif (str): The path to save the output raster.
        resolution (float): The spatial resolution of the raster in degrees.

    Returns:
        str: Path to the saved raster file.
    """

    # Get the bounding box of the geometries
    minx, miny, maxx, maxy = coastline.total_bounds

    # Calculate the number of rows and columns
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    # Define the transform
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Create an empty raster array
    raster = np.zeros((height, width), dtype=np.uint8)

    # Rasterize the LINESTRING geometries
    shapes = (
        (geom, 1) for geom in coastline.geometry
    )  # Assign a value of 1 to lines
    raster = rasterize(
        shapes, out_shape=(height, width), transform=transform, fill=0
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    # Save to TIFF
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=raster.dtype,
        crs=coastline.crs,  # Ensure the CRS is the same as the input
        transform=transform,
    ) as dst:
        dst.write(raster, 1)

    print(f"Raster saved at: {output_tif}")
    return output_tif


def create_coastline_tif(shp, out_dir):
    landmass = shp.dissolve()
    coastline = landmass.boundary  # Get the outer boundary (coastline only)
    # Convert to GeoDataFrame
    coastline = gpd.GeoDataFrame(
        geometry=coastline.explode(index_parts=False), crs=shp.crs
    )
    # Rasterize the coastline
    output_tif = os.path.join(out_dir, "coastline.tif")
    rasterize_coastline(coastline, output_tif, resolution=0.01)


def load_coastline_raster(raster_path):
    """
    Load the coastline raster once and return relevant metadata.

    Parameters:
        raster_path (str): Path to the coastline raster (coastline.tif).

    Returns:
        dict: Contains raster array, affine transform, CRS, and nodata value.
    """
    with rasterio.open(raster_path) as src:
        return {
            "raster_path": raster_path,
            "transform": src.transform,
            "crs": src.crs,
            "nodata": src.nodata,
            "pixel_size": abs(src.transform.a),  # Get pixel resolution
        }


def get_coast_features_from_raster(grid, raster_data):
    """
    Compute coastline intersection features using preloaded raster data.

    Parameters:
        grid (GeoDataFrame): Grid cells with geometries.
        raster_data (dict): Preloaded raster metadata.

    Returns:
        GeoDataFrame: Updated grid with 'coast_length' and 'with_coast' columns.
    """

    # Ensure grid is in the same CRS as the raster
    if grid.crs != raster_data["crs"]:
        grid = grid.to_crs(raster_data["crs"])

    # Make an explicit copy to avoid SettingWithCopyWarning
    grid = grid.copy()

    # Compute zonal statistics
    stats = zonal_stats(
        grid,
        raster_data["raster_path"],
        affine=raster_data["transform"],
        stats=["sum"],  # 'sum' will count the coastline pixels
        nodata=raster_data["nodata"],
    )

    # Assign values safely using `.loc`
    grid.loc[:, "coast_pixels"] = [
        s["sum"] if s["sum"] is not None else 0 for s in stats
    ]

    # Convert coastline pixels to meters
    grid.loc[:, "coast_length"] = (
        grid["coast_pixels"] * raster_data["pixel_size"]
    )

    # Binary flag for coastline presence
    grid.loc[:, "with_coast"] = np.where(grid["coast_length"] > 0, 1, 0)

    return grid[["id", "coast_length", "with_coast"]]


def process_country(grid, country, raster_data, out_dir):
    """
    Process each country and save the results to CSV if not already processed.

    Parameters:
        grid (GeoDataFrame): The original grid.
        country (str): The country ID.
        raster_data (dict): Preloaded raster metadata.
        out_dir (str): The output directory.
    """

    # Filter the grid for the current country
    grid_country = grid[grid.GID_0 == country]

    # Define output file path
    out_file = os.path.join(out_dir, f"coastline_data_{country}.csv")

    # Check if the file already exists
    if os.path.exists(out_file):
        print(f"File {out_file} already exists, skipping...")
        return

    # Compute the coastline features
    grid_with_coast = get_coast_features_from_raster(grid_country, raster_data)

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save the result to CSV
    grid_with_coast.to_csv(out_file, index=False)
    print(f"Processed and saved {out_file}")


def parallel_process_countries(grid, raster_data, out_dir, max_workers=4):
    """
    Run the country processing function in parallel for all countries.

    Parameters:
        grid (GeoDataFrame): The original grid.
        raster_data (dict): Preloaded raster metadata.
        out_dir (str): The output directory for saving results.
        max_workers (int): The maximum number of parallel workers.
    """

    # Get the list of unique countries
    country_list = grid.GID_0.unique()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to process each country
        futures = [
            executor.submit(
                process_country, grid, country, raster_data, out_dir
            )
            for country in country_list
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()


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


if __main__ == "__name__":
    # Load data
    data_path = "/data/big/fmoss/data/SRTM/tiles/"
    grid = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )
    shp = gpd.read_file("/data/big/fmoss/data/SHP/gadm_410.gdb")

    # Adjust data
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    # Create tif file
    tif_dir = "/data/big/fmoss/data/SRTM/"
    create_coastline_tif(shp, tif_dir)

    # Process to grid level
    coastline_tif = f"{tif_dir}/coastline.tif"  # Path to coastline raster
    raster_data = load_coastline_raster(coastline_tif)
    out_dir = "/data/big/fmoss/data/SRTM/grid_data_coastline"

    # Run the parallel processing
    parallel_process_countries(grid, raster_data, out_dir, max_workers=4)
