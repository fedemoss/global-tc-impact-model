#!/usr/bin/env python3
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from rasterio.merge import merge
from rasterstats import zonal_stats
from shapely.geometry import box

# def merge_tif_files(tif_paths):
#     """
#     Merges multiple TIFF files into a single mosaic and returns the merged data.

#     :param tif_paths: List of paths to the TIFF files to be merged.
#     :return: A tuple containing:
#         - merged_data (numpy array): The merged raster data from the first band.
#         - out_transform (Affine): The affine transform of the merged raster.
#         - out_meta (dict): The metadata of the merged raster.
#     """
#     # Open all TIFF files
#     src_files = [rasterio.open(tif) for tif in tif_paths]

#     # Ensure all files have the same CRS
#     crs_list = [src.crs for src in src_files]
#     if not all(crs == crs_list[0] for crs in crs_list):
#         raise ValueError("All input TIFF files must have the same CRS.")

#     # Merge the TIFF files into a single mosaic
#     mosaic, out_transform = merge(src_files)

#     # Define the metadata for the output file using the first raster
#     out_meta = src_files[0].meta.copy()
#     out_meta.update({
#         "driver": "GTiff",
#         "count": mosaic.shape[0],  # Number of bands
#         "height": mosaic.shape[1],  # Height of the merged raster
#         "width": mosaic.shape[2],   # Width of the merged raster
#         "transform": out_transform  # Updated transform
#     })

#     # Close all opened files
#     for src in src_files:
#         src.close()

#     return mosaic, out_transform, out_meta


def select_tif_files_by_country(
    tif_paths, grid_global, iso3, adjusted=True, buffer=3
):
    """
    Selects TIF files corresponding to a specified country based on ISO3 code and latitude/longitude in the filename.

    Args:
        tif_paths (list): List of TIF file paths to filter.
        grid_global (GeoDataFrame): GeoDataFrame containing country boundaries with ISO3 codes.
        iso3 (str): ISO3 code of the target country.
        adjusted (bool): Whether to use adjusted or raw depth files.
        buffer (int): Buffer (degrees) to extend the bounding box conservatively.

    Returns:
        list: A list of TIF file paths corresponding to the specified country.
    """
    # Get country boundaries
    # country_geometry = grid_global[grid_global.iso3 == iso3].geometry.unary_union.buffer(buffer)
    selected_files = []
    country_geometry = (
        grid_global[grid_global.iso3 == iso3]
        .geometry.union_all()
        .buffer(buffer)
    )

    # Regex patterns for different file types
    pattern = (
        r"ID\d+_(N|S)(\d+)_([EW])(\d+)_RP10_depth"
        + ("_reclass" if adjusted else "")
        + r"\.tif"
    )

    for tif_path in tif_paths:
        filename = os.path.basename(tif_path)
        match = re.search(pattern, filename)
        if match:
            lat_sign, lat_value, lon_sign, lon_value = match.groups()
            lat_value = int(lat_value) if lat_sign == "N" else -int(lat_value)
            lon_value = int(lon_value) if lon_sign == "E" else -int(lon_value)

            tif_point = box(
                lon_value - 0.5,
                lat_value - 0.5,
                lon_value + 0.5,
                lat_value + 0.5,
            ).centroid

            if country_geometry.contains(tif_point):
                selected_files.append(tif_path)

    return selected_files


def reproject_grid(grid, target_crs):
    """
    Reprojects a GeoDataFrame to the target CRS.

    :param grid: GeoDataFrame with the original CRS.
    :param target_crs: Target CRS to reproject into.
    :return: Reprojected GeoDataFrame.
    """
    if grid.crs != target_crs:
        grid = grid.to_crs(target_crs)
    return grid


# def save_mosaic_to_tif(mosaic, out_transform, out_meta, output_file):
#     """
#     Save the merged mosaic to a GeoTIFF file.

#     Parameters:
#         mosaic (numpy array): The merged raster data (3D array: bands x height x width).
#         out_transform (Affine): The affine transformation for the raster.
#         out_meta (dict): The metadata dictionary for the raster.
#         output_file (str): The output file path.
#     """
#     # Update metadata for the output file
#     out_meta.update({
#         "driver": "GTiff",
#         "height": mosaic.shape[1],  # Height of the raster
#         "width": mosaic.shape[2],  # Width of the raster
#         "transform": out_transform  # Georeferencing transform
#     })

#     # Write the mosaic to a GeoTIFF file
#     with rasterio.open(output_file, "w", **out_meta) as dest:
#         dest.write(mosaic)

# def aggregate_raster_to_grid(raster_path, grid):
#     """
#     Clips a raster to each grid cell and calculates maximum depth values per grid cell.

#     :param raster_path: Path to the input raster file.
#     :param grid: GeoDataFrame representing grid cells.
#     :return: GeoDataFrame with aggregated statistics added.
#     """
#     # Open the raster file
#     with rasterio.open(raster_path) as src:
#         # Ensure the grid and raster are in the same CRS
#         if grid.crs != src.crs:
#             grid = grid.to_crs(src.crs)

#         # Use zonal_stats to calculate max values within each grid cell
#         stats = zonal_stats(
#             grid,  # GeoDataFrame or geometry to aggregate over
#             raster_path,
#             stats=["max"],  # Compute the maximum value
#             nodata=src.nodata,
#             affine=src.transform,
#             all_touched=True  # Include all raster cells touched by the geometry
#         )

#     # Add max depth to the grid GeoDataFrame
#     grid["flood_risk"] = [stat["max"] if stat["max"] is not None else np.nan for stat in stats]

#     return grid

# def process_flood_risk(grid_global, iso, tif_paths_adj):
#     # Select tif files for the specified country
#     tif_paths_cat = select_tif_files_by_country(tif_paths_adj, grid_global, iso, adjusted=True, buffer=5)

#     # Filter and reproject the grid for the selected country
#     grid_country = grid_global[grid_global.iso3 == iso].copy()
#     grid_country_projected = reproject_grid(grid_country, "EPSG:4326")

#     # Check if there are tif files to process
#     if len(tif_paths_cat) > 0:
#         # Merge tif files
#         mosaic_adj, out_transform_adj, out_meta_adj = merge_tif_files(tif_paths_cat)

#         # Define output file path and create the directory if needed
#         output_file = f"./country_specific_flood_risk/{iso.lower()}_flood_risk_cat.tif"
#         output_dir = os.path.dirname(output_file)  # Extract directory path
#         os.makedirs(output_dir, exist_ok=True)     # Create directory

#         # Save the mosaic to a GeoTIFF file
#         save_mosaic_to_tif(mosaic_adj, out_transform_adj, out_meta_adj, output_file)

#         # Read and aggregate the raster data to the grid
#         with rasterio.open(output_file) as src:
#             raster = src.read(1)  # Read the first band
#             raster_crs = src.crs

#         # Aggregate raster data to the grid and fill NaN values
#         aggregated_grid = aggregate_raster_to_grid(output_file, grid_country_projected)# .fillna(0)
#     else:
#         # If no tif files, create an empty flood risk column
#         aggregated_grid = grid_country_projected.copy()
#         aggregated_grid['flood_risk'] = np.nan

#     return aggregated_grid


def process_single_tif(tif_path, grid):
    """
    Process a single TIF file, extracting max flood risk per grid cell.

    :param tif_path: Path to the TIF file.
    :param grid: GeoDataFrame of grid cells.
    :return: List of max flood risk values per grid cell.
    """
    with rasterio.open(tif_path) as src:
        # Ensure the grid and raster are in the same CRS
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        # Compute max flood depth per grid cell
        stats = zonal_stats(
            grid,
            tif_path,
            stats=["max"],
            nodata=src.nodata,
            affine=src.transform,
            all_touched=True,
        )

    return [
        stat["max"] if stat["max"] is not None else np.nan for stat in stats
    ]


def process_flood_risk(
    grid_global, iso, tif_paths_adj, parallel=True, n_jobs=-1
):
    """
    Compute flood risk at grid level without merging all TIF files at once.

    :param grid_global: GeoDataFrame containing all country grids.
    :param iso: ISO3 country code.
    :param tif_paths_adj: List of TIF file paths.
    :param parallel: Whether to use parallel processing.
    :param n_jobs: Number of parallel jobs (-1 uses all available cores).
    :return: Updated GeoDataFrame with flood risk.
    """
    # Select TIFF files for the given country
    tif_paths_cat = select_tif_files_by_country(
        tif_paths_adj, grid_global, iso, adjusted=True, buffer=3
    )

    # Extract country-specific grid and reproject it
    grid_country = grid_global[grid_global.iso3 == iso].copy()
    grid_country = reproject_grid(grid_country, "EPSG:4326")

    if not tif_paths_cat:
        # No TIF files found; return grid with NaN flood risk
        grid_country["flood_risk"] = np.nan
        return grid_country

    # Process TIF files (in parallel if enabled)
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_tif)(tif, grid_country)
            for tif in tif_paths_cat
        )
    else:
        results = [
            process_single_tif(tif, grid_country) for tif in tif_paths_cat
        ]

    # Convert results into a NumPy array and compute max per grid cell
    results = np.array(results)  # Shape: (num_tifs, num_grid_cells)
    grid_country["flood_risk"] = np.nanmax(
        results, axis=0
    )  # Max per grid cell across all TIFs

    return grid_country


def process_and_save_flood_risk(
    iso, grid_global, tif_paths_adj, output_dir, parallel=True, n_jobs=-1
):
    """
    Processes flood risk for a given ISO3 country code and saves the results to a CSV.

    :param iso: ISO3 country code.
    :param grid_global: GeoDataFrame containing all country grids.
    :param tif_paths_adj: List of available TIF file paths.
    :param output_dir: Directory to save the output files.
    :param parallel: Whether to use parallel processing for TIFFs.
    :param n_jobs: Number of parallel jobs (-1 uses all available cores).
    """
    out_path = os.path.join(output_dir, f"flood_risk_{iso}.csv")

    # # Skip processing if the output file already exists
    # if os.path.exists(out_path):
    #     print(f"Skipping {iso}, output already exists: {out_path}")
    #     return

    print(f"Processing: {iso}")

    # Compute flood risk
    flood_risk_iso = process_flood_risk(
        grid_global, iso, tif_paths_adj, parallel=parallel, n_jobs=n_jobs
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    flood_risk_iso[["id", "iso3", "flood_risk"]].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def process_multiple_countries(
    grid_global, output_dir, tif_path, max_workers=4, parallel=True, n_jobs=-1
):
    """
    Processes flood risk for multiple countries in parallel.

    :param grid_global: GeoDataFrame containing all country grids.
    :param output_dir: Directory to save the output files.
    :param select_tif_files_by_country: Function to retrieve available TIF files.
    :param max_workers: Number of parallel processes for countries.
    :param parallel: Whether to use parallel processing for TIFFs.
    :param n_jobs: Number of parallel jobs (-1 uses all available cores).
    """
    iso3_list = grid_global.iso3.unique()

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for iso in iso3_list:
            out_path = os.path.join(output_dir, f"flood_risk_{iso}.csv")

            # Check if output file exists before submitting
            if os.path.exists(out_path):
                print(f"Skipping {iso}, output already exists: {out_path}")
                continue

            # Submit task to process the country
            futures[
                executor.submit(
                    process_and_save_flood_risk,
                    iso=iso,
                    grid_global=grid_global,
                    tif_paths_adj=tif_path,
                    output_dir=output_dir,
                    parallel=parallel,
                    n_jobs=n_jobs,
                )
            ] = iso

        # Ensure all tasks complete and handle exceptions
        for future in futures:
            try:
                future.result()  # Wait for each thread to complete
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")


if __name__ == "__main__":
    downloaded_dir_adj = "/data/big/fmoss/data/FloodRisk/tiles"
    tif_paths_adj = [
        os.path.join(downloaded_dir_adj, filename)
        for filename in os.listdir(downloaded_dir_adj)
        if filename.endswith("_depth_reclass.tif")
    ]

    # Load global grid cells
    grid_training = gpd.read_file(
        "/home/fmoss/GLOBAL MODEL/data/GRID/grid_global.gpkg"
    )
    grid_world = gpd.read_file(
        "/home/fmoss/GLOBAL MODEL/data/GRID/other_countries_grid.gpkg"
    )
    grid_global = pd.concat([grid_training, grid_world])
    # Define output directory
    output_dir = "/data/big/fmoss/data/FloodRisk/grid_data"
    # Get the unique country codes
    iso3_list = grid_global.iso3.unique()
    # Process country
    process_multiple_countries(
        grid_global, output_dir, tif_paths_adj, max_workers=1, n_jobs=1
    )
