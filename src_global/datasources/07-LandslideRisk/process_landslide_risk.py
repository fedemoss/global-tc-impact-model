#!/usr/bin/env python3
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats


def aggregate_raster_to_grid(raster_path, grid):
    """
    Clips a raster to each grid cell and calculates maximum depth values per grid cell.

    :param raster_path: Path to the input raster file.
    :param grid: GeoDataFrame representing grid cells.
    :return: GeoDataFrame with aggregated statistics added.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Ensure the grid and raster are in the same CRS
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        # Use zonal_stats to calculate max values within each grid cell
        stats = zonal_stats(
            grid,  # GeoDataFrame or geometry to aggregate over
            raster_path,
            stats=["max", "sum", "median"],  # Compute the maximum value
            nodata=src.nodata,
            affine=src.transform,
            all_touched=True,  # Include all raster cells touched by the geometry
        )

    # Add max depth to the grid GeoDataFrame
    # grid["landslide_risk_max"] = [stat["max"] if stat["max"] is not None else np.nan for stat in stats]
    grid["landslide_risk_sum"] = [
        stat["sum"] if stat["sum"] is not None else np.nan for stat in stats
    ]
    # grid["landslide_risk_median"] = [stat["median"] if stat["median"] is not None else np.nan for stat in stats]
    return grid


if __name__ == "__main__":
    # Load global grid cells
    # grid_global = gpd.read_file('/home/fmoss/GLOBAL MODEL/data/GRID/grid_global.gpkg')
    # grid_global = gpd.read_file("/home/fmoss/GLOBAL MODEL/data/GRID/other_countries_grid.gpkg")
    grid_global = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )
    grid_global.loc[:, "iso3"] = grid_global["GID_0"]

    # Load landslide risk info
    raster_path = "/data/big/fmoss/data/LandSlides/landslide_data.tif"
    # To grid
    grid_landslide = aggregate_raster_to_grid(raster_path, grid_global)[
        ["id", "iso3", "landslide_risk_sum"]
    ]

    # Save it
    out_file_path = (
        "/data/big/fmoss/data/LandSlides/global_grid_landslide_risk.csv"
    )
    grid_landslide.to_csv(out_file_path, index=False)
