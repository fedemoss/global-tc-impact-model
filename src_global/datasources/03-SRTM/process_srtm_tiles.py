#!/usr/bin/env python3
import concurrent.futures
import logging
import os
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import open as rio_open
from rasterio.merge import merge
from rasterstats import zonal_stats
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors

# Set up the logger
logging.basicConfig(
    filename="process_srtm_files.log",  # Specify the log file name
    level=logging.ERROR,  # Log level set to ERROR to log only errors
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format for log messages
)


# Define a function to adjust the longitude of a single polygon
# the way srtm data likes it --> (-360, 0)
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


# Check border condition cases (these can cause problems if not detected)
def is_border_crossing(polygon):
    """Check if the polygon contains both positive and negative longitudes."""
    coords = list(polygon.exterior.coords)
    has_positive_lon = False
    has_negative_lon = False

    for lon, lat in coords:
        if lon > 0:
            has_positive_lon = True
        if lon < 0:
            has_negative_lon = True

        # If both conditions are met, the polygon crosses the meridian
        if has_positive_lon and has_negative_lon:
            return True

    return False


# Overlapping tiles / countries
def get_overlap_files(extent):
    """
    Identifies the .tif files overlapping with a country's geometry.

    Parameters:
        cextent: A list of total bounds of a geo file

    Returns:
        list: A list of .tif file names that overlap with the country.
    """
    # Generate a list of global .tif files
    tif_files = [
        f"srtm_{xx:02d}_{yy:02d}.tif"
        for xx in range(1, 73)  # xx ranges from 01 to 72
        for yy in range(1, 25)  # yy ranges from 01 to 24
    ]

    # Get total bounds of the country's geometry (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = extent

    # Global grid definitions
    lon_extent = 360  # Longitude range: 180W to 180E
    lat_extent = 120  # Latitude range: 60S to 60N
    lon_boxes = 72  # Number of longitude divisions
    lat_boxes = 24  # Number of latitude divisions

    lon_step = lon_extent / lon_boxes
    lat_step = lat_extent / lat_boxes

    # Calculate grid boundaries for each tif file
    overlapping_files = []
    for file in tif_files:
        # Parse xx and yy from file name
        parts = file.replace("srtm_", "").replace(".tif", "").split("_")
        xx = int(parts[0]) - 1  # Convert to 0-based index
        yy = int(parts[1]) - 1

        # Calculate the bounds of the grid cell corresponding to the file
        cell_minx = -180 + xx * lon_step
        cell_maxx = cell_minx + lon_step
        cell_maxy = 60 - yy * lat_step
        cell_miny = cell_maxy - lat_step

        # Check if the bounding box of the cell intersects the country's bounds
        if not (
            cell_maxx <= minx
            or cell_minx >= maxx
            or cell_maxy <= miny
            or cell_miny >= maxy
        ):
            overlapping_files.append(file)

    return overlapping_files


# Select country SRTM tiles for countries with grid cells
def get_country_tiles(grid):
    country_tiles = []
    for iso in grid.iso3.unique():
        # Particular cases
        if iso == "USA":
            usa_extent_1 = [172, 18.9, 180, 72.7]
            usa_extent_2 = [-180, 18.9, -67, 72.7]
            tiles = get_overlap_files(usa_extent_1) + get_overlap_files(
                usa_extent_2
            )
        elif iso == "FJI":
            fji_extent_1 = [176.8, -21.1, 180, -12.4]
            fji_extent_2 = [-180, -21.1, -178, -12.4]
            tiles = get_overlap_files(fji_extent_1) + get_overlap_files(
                fji_extent_2
            )
        elif iso == "NZL":
            nzl_extent_1 = [165.8, -52.7, 180, -29.2]
            nzl_extent_2 = [-180, -52.7, -176, -29.2]
            tiles = get_overlap_files(nzl_extent_1) + get_overlap_files(
                nzl_extent_2
            )
        elif iso == "KIR":
            kir_extent_1 = [176, -11, 180, 5]
            kir_extent_2 = [-180, -11, -174, 5]
            tiles = get_overlap_files(kir_extent_1) + get_overlap_files(
                kir_extent_2
            )
            continue
        elif iso == "RUS":
            rus_extent_1 = [25, 41.18886566, 180, 81.856247]
            rus_extent_2 = [-180, 41.18886566, -170, 81.856247]
            tiles = get_overlap_files(rus_extent_1) + get_overlap_files(
                rus_extent_2
            )
        # Regular cases
        else:
            country_grid = grid[grid.iso3 == iso]
            extent = country_grid.total_bounds
            tiles = get_overlap_files(extent)

        country_tiles.append({"iso3": iso, "tiles": tiles})
    return pd.DataFrame(country_tiles)


# Mosaic raster paths for the country based on all its raster tiles
def get_mosaic_raster_paths(row, data_path):
    """
    Get a list of raster file paths for the given row and data path.

    Args:
        row: Row containing a 'tiles' column with file names.
        data_path: Base path to the raster files.

    Returns:
        List of valid raster file paths.
    """

    # Helper function to validate tile paths
    def process_tile(tile):
        tile_path = os.path.join(data_path, tile)
        if os.path.exists(tile_path):
            return tile_path
        return None

    # Parallelize the processing of tiles
    tiles = row["tiles"]
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_tile, tiles)

    # Collect valid file paths, filtering out None (for missing tiles)
    mosaic_raster_paths = [
        tile_path for tile_path in results if tile_path is not None
    ]

    return mosaic_raster_paths


def calculate_zonal_stats(
    grid, raster_array, stats, nodata, affine, stat_type
):
    """
    Calculates zonal statistics for a given raster array and grid.
    """
    summary_stats = zonal_stats(
        grid,
        raster_array,
        stats=stats,
        nodata=nodata,
        all_touched=True,
        affine=affine,
    )
    summary_df = pd.DataFrame(summary_stats)
    return summary_df


def process_altitude(grid, altitude, affine):
    """
    Process altitude raster and return a DataFrame with mean altitude by grid.
    """
    altitude_df = calculate_zonal_stats(
        grid=grid,
        raster_array=altitude,
        stats=["mean"],
        nodata=-32768,
        affine=affine,
        stat_type="altitude",
    )
    del altitude  # Free memory after processing
    altitude_df["id"] = grid["id"].values
    return altitude_df


def process_slope(grid, slope_array, affine):
    """
    Process slope raster and return a DataFrame with mean and std of slope by grid.
    """
    slope_df = calculate_zonal_stats(
        grid=grid,
        raster_array=slope_array,
        stats=["mean", "std"],
        nodata=-9999,
        affine=affine,
        stat_type="slope",
    )
    del slope_array  # Free memory after processing
    slope_df["id"] = grid["id"].values
    return slope_df


def process_ruggedness(grid, tri_array, affine):
    """
    Process TRI raster and return a DataFrame with mean and std of ruggedness by grid.
    """
    ruggedness_df = calculate_zonal_stats(
        grid=grid,
        raster_array=tri_array,
        stats=["mean", "std"],
        nodata=-9999,
        affine=affine,
        stat_type="ruggedness",
    )
    del tri_array  # Free memory after processing
    ruggedness_df["id"] = grid["id"].values
    return ruggedness_df


# Main pipeline for 1 tif file
def process_single_raster(grid, raster_path):
    """
    Process a single raster file for altitude, slope, and ruggedness.

    Args:
        grid: The grid geometry to process against.
        raster_path: Path to the raster file.

    Returns:
        A tuple of DataFrames (altitude_df, slope_df, ruggedness_df).
    """
    # First of all, exclude border geometries (we'll fill them out later)
    grid["border"] = grid["geometry"].apply(is_border_crossing)
    grid = grid[grid.border == False]

    def process_altitude_task(raster_path):
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)  # Assuming data is in the first band
            raster_transform = src.transform
        return process_altitude(grid, raster_data, raster_transform)

    def process_slope_task(raster_path):
        slope_path = tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ).name
        subprocess.run(
            [
                "gdaldem",
                "slope",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "ZLEVEL=9",
                raster_path,
                slope_path,
                "-compute_edges",
            ],
            check=True,
            stdout=subprocess.PIPE,  # Suppress standard output
            stderr=subprocess.PIPE,  # Suppress standard erro
        )
        with rasterio.open(slope_path) as slope_rast:
            slope_array = slope_rast.read(1)
            return (
                process_slope(grid, slope_array, slope_rast.transform),
                slope_path,
            )

    def process_ruggedness_task(raster_path):
        tri_path = tempfile.NamedTemporaryFile(
            suffix=".tif", delete=False
        ).name
        subprocess.run(
            [
                "gdaldem",
                "TRI",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "ZLEVEL=9",
                raster_path,
                tri_path,
                "-compute_edges",
            ],
            check=True,
            stdout=subprocess.PIPE,  # Suppress standard output
            stderr=subprocess.PIPE,  # Suppress standard erro
        )
        with rasterio.open(tri_path) as tri_rast:
            tri_array = tri_rast.read(1)
            return (
                process_ruggedness(grid, tri_array, tri_rast.transform),
                tri_path,
            )

    try:
        # Use ThreadPoolExecutor to process altitude, slope, and ruggedness concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Start all three tasks simultaneously
            altitude_future = executor.submit(
                process_altitude_task, raster_path
            )
            slope_future = executor.submit(process_slope_task, raster_path)
            ruggedness_future = executor.submit(
                process_ruggedness_task, raster_path
            )

            # Wait for all tasks to complete
            altitude_df = altitude_future.result()
            slope_df, slope_path = slope_future.result()
            ruggedness_df, tri_path = ruggedness_future.result()

    finally:
        # Cleanup temporary files
        for temp_file in [slope_path, tri_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return altitude_df, slope_df, ruggedness_df


# Main pipeline fo all tif files for 1 country (easily expandible for multiple countries)
def main_processing_pipeline(grid, mosaic_raster_paths, num_workers=4):
    """
    Main pipeline to process altitude, slope, and ruggedness for each raster in parallel.

    Args:
        grid: The grid geometry to process against.
        mosaic_raster_paths: List of raster file paths.
        num_workers: Number of parallel workers to process rasters.

    Returns:
        A tuple of merged DataFrames (altitude_df, slope_df, ruggedness_df).
    """
    # Process each raster file in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            process_single_raster,
            [(grid, path) for path in mosaic_raster_paths],
        )

    # Merge results from all rasters
    altitude_df = pd.concat(
        [res[0].dropna() for res in results], ignore_index=True
    )
    slope_df = pd.concat(
        [res[1].dropna() for res in results], ignore_index=True
    )
    ruggedness_df = pd.concat(
        [res[2].dropna() for res in results], ignore_index=True
    )

    # Since there can be missing values, merge back with grid
    altitude_df = altitude_df.merge(grid, how="right")
    slope_df = slope_df.merge(grid, how="right")
    ruggedness_df = ruggedness_df.merge(grid, how="right")

    return altitude_df, slope_df, ruggedness_df


# Merge altitude, slope and ruggedness
def merge_terrain_data(grid, altitude_df, slope_df, ruggedness_df):
    """
    Merge the altitude, slope, and ruggedness DataFrames into the grid DataFrame with appropriate renaming.
    """
    # Renaming columns in each DataFrame
    df_slope = slope_df.rename({"mean": "mean_slope"}, axis=1)  # .fillna(0)
    df_elev = altitude_df.rename({"mean": "mean_elev"}, axis=1)  # .fillna(0)
    df_rugged = ruggedness_df.rename(
        {"mean": "mean_rug"}, axis=1
    )  # .fillna(0)

    # Merge slope and elevation DataFrames on 'grid_id'
    df_slope_elev = df_slope.merge(df_elev, on="id", how="left")

    # Merge slope, elevation, and ruggedness DataFrames on 'grid_id'
    df_terrain = df_slope_elev.merge(df_rugged, on="id", how="left")

    # Ensure the final DataFrame contains only the necessary columns
    df_terrain = df_terrain[["id", "mean_elev", "mean_slope", "mean_rug"]]

    # Merge the terrain DataFrame with the original grid DataFrame
    grid_terrain = grid.merge(df_terrain, on="id", how="left")

    grid_terrain = grid_terrain.to_crs(grid.crs)

    return grid_terrain


# For missing raster values
def spatial_interpolation(df, columns_to_interpolate):
    """
    Perform spatial interpolation for specified columns in a GeoDataFrame.

    Parameters:
        df (GeoDataFrame): Input GeoDataFrame with geometries and missing values.
        columns_to_interpolate (list): List of column names to interpolate.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with interpolated values.
    """

    # Extract centroids of the geometries
    df = df.copy()
    df["centroid"] = df.geometry.centroid
    coordinates = np.array([[point.x, point.y] for point in df.centroid])

    # Loop through columns to interpolate
    for column in columns_to_interpolate:
        # Check for missing values
        if df[column].isnull().any():
            # Find missing and non-missing indices
            missing_mask = df[column].isna().values
            non_missing_mask = ~missing_mask

            # Extract known values and their coordinates
            known_values = df.loc[non_missing_mask, column].values
            known_coords = coordinates[non_missing_mask]

            # Skip interpolation if there are no known values
            if len(known_coords) == 0:
                continue

            N = max(len(known_coords), 1)
            k = min(8, N)

            # Extract missing coordinates
            missing_coords = coordinates[missing_mask]

            # Skip interpolation if there are no missing coordinates
            if len(missing_coords) == 0:
                continue

            # Fit nearest neighbors model on known points
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(
                known_coords
            )
            distances, indices = nbrs.kneighbors(missing_coords)

            # Perform IDW (Inverse Distance Weighting) interpolation
            weights = 1 / (
                distances + 1e-10
            )  # Add a small value to avoid division by zero
            weights /= weights.sum(axis=1, keepdims=True)

            # Calculate interpolated values for missing points
            interpolated_values = (weights * known_values[indices]).sum(axis=1)

            # Update DataFrame with interpolated values
            df.loc[missing_mask, column] = interpolated_values

    # Drop the temporary centroid column
    df.drop(columns=["centroid"], inplace=True)

    return df


# Coast features
def get_coast_features(shp, grid):
    # dissolving polygons into one land mass
    dissolved_shp = shp.dissolve(by="GID_0")

    # Coastline
    coastline = dissolved_shp.boundary
    coastline.crs = grid.crs

    # Linestrings and MultiLinestrings
    grid_line_gdf = gpd.overlay(
        gpd.GeoDataFrame(
            coastline, geometry=coastline.geometry, crs=coastline.crs
        ).reset_index(),
        grid,
        how="intersection",
    )[["id", "geometry"]]
    # Coast line
    grid_line_gdf["coast_length"] = (
        grid_line_gdf["geometry"].to_crs(25394).length
    )  # 25394 gives me length in meters.
    grid_coast = grid[["id"]].merge(grid_line_gdf, on=["id"], how="left")

    # With coast? Binary
    grid_coast["with_coast"] = np.where(grid_coast["coast_length"] > 0, 1, 0)
    grid_coast["with_coast"].value_counts()

    return grid_coast[["id", "coast_length", "with_coast"]].fillna(0)


# Terrain + Coast features + apply fill missing values
def merge_srtm_features(df_coast, df_terrain, spatial_interpolation=True):
    # Merge data
    columns_to_fill = ["mean_elev", "mean_slope", "mean_rug"]
    if spatial_interpolation:
        df_terrain = spatial_interpolation(df_terrain, columns_to_fill)
    merged = df_terrain.merge(df_coast, on="id", how="left")
    merged = merged[
        [
            "id",
            "with_coast",
            "coast_length",
            "mean_elev",
            "mean_slope",
            "mean_rug",
        ]
    ]
    return merged


# Entire Process for 1 country
def data_extraction(iso, grid, shp, country_tiles, out_path, data_path):
    try:
        output_path = f"{out_path}/srtm_grid_data_{iso}.csv"
        # First check if file already exists (dont recompute existing directories)
        if os.path.exists(output_path):
            pass
        else:
            # Country grid and shapefile
            grid_country = grid[grid.iso3 == iso]
            shp_country = shp[shp.GID_0 == iso]

            # Get Altitude, Slope and Ruggedness data
            # mosaic_raster = country_tiles[country_tiles.iso3 == iso].apply(
            #     lambda row: get_mosaic_raster(row=row, data_path=data_path), axis=1).iloc[0]
            # altitude_df, slope_df, ruggedness_df = main_processing_pipeline(grid_country, mosaic_raster)

            # Get Altitude, Slope and Ruggedness data
            mosaic_raster_paths = (
                country_tiles[country_tiles.iso3 == iso]
                .apply(
                    lambda row: get_mosaic_raster_paths(
                        row=row, data_path=data_path
                    ),
                    axis=1,
                )
                .iloc[0]
            )

            if len(mosaic_raster_paths) == 0:
                df_terrain = grid_country.copy()
                df_terrain["mean_elev", "mean_slope", "mean_rug"] = np.nan
                # Get Coastline data
                df_coast = get_coast_features(shp_country, grid_country)
                # Merge everything + fill missing values with surrounding data
                df_srtm = merge_srtm_features(
                    df_coast, df_terrain, spatial_interpolation=False
                )
                df_srtm["iso3"] = iso

            else:
                (
                    altitude_df,
                    slope_df,
                    ruggedness_df,
                ) = main_processing_pipeline(
                    grid_country, mosaic_raster_paths, num_workers=4
                )
                # Terrain features all together
                df_terrain = merge_terrain_data(
                    grid_country, altitude_df, slope_df, ruggedness_df
                )
                # Get Coastline data
                df_coast = get_coast_features(shp_country, grid_country)

                # Merge everything + fill missing values with surrounding data
                df_srtm = merge_srtm_features(df_coast, df_terrain)
                df_srtm["iso3"] = iso

            # Ensure the directory exists or create it
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save and return the result
            df_srtm.to_csv(output_path, index=False)

            return df_srtm
    except Exception as e:
        # Log the error message to a file
        logging.error(f"Couldn't calculate terrain data for {iso}. Error: {e}")
        return None  # Return None if there's an error


# Entire Porcess for all countries
def iterate_srtm_data_extracting(grid, shp, out_path, data_path):
    # Tiles
    country_tiles = get_country_tiles(grid=grid)

    # Terrain data --> dataframe
    srtm_data = pd.DataFrame()

    # List of unique country ISO codes
    iso_list = country_tiles.iso3.unique()
    # iso_list = ['MEX', 'USA']

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks for each country ISO code
        future_to_iso = {
            executor.submit(
                data_extraction,
                iso,
                grid,
                shp,
                country_tiles,
                out_path,
                data_path,
            ): iso
            for iso in iso_list
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_iso):
            result = future.result()
            if result is not None:
                srtm_data = pd.concat([srtm_data, result], ignore_index=True)

    return srtm_data


if __name__ == "__main__":
    # Paths to tiles and output path
    out_path = "/data/big/fmoss/data/SRTM/grid_data"
    data_path = "/data/big/fmoss/data/SRTM/tiles/"
    # Load grid and shapefile data
    grid = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )
    shp = gpd.read_file("/home/fmoss/GLOBAL MODEL/data/SHP/gadm_410.gdb")
    # The process itself
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    grid.loc[:, "iso3"] = grid["GID_0"]

    srtm_data = iterate_srtm_data_extracting(grid, shp, out_path, data_path)
    # Save it
    srtm_data.to_csv(f"{out_path}/global_srtm_grid_data.csv", index=False)
