#!/usr/bin/env python3
import concurrent.futures
import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Set up logging
logging.basicConfig(
    filename="grid.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Country-grid data creation
def create_grid(shp, iso3):
    shp = shp[shp.GID_0 == iso3].reset_index(drop=True)

    # Calculate the bounding box of the GeoDataFrame
    bounds = shp.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds

    # Define a margin to expand the bounding box
    margin = 1

    # Create expanded bounding box coordinates
    lon_min = np.round(lon_min - margin)
    lat_min = np.round(lat_min - margin)
    lon_max = np.round(lon_max + margin)
    lat_max = np.round(lat_max + margin)

    # Define grid
    cell_size = 0.1
    cols = np.arange(lon_min, lon_max + cell_size, cell_size)
    rows = np.arange(lat_min, lat_max + cell_size, cell_size)

    # Reverse rows using slicing
    rows = rows[::-1]

    polygons = [
        Polygon(
            [
                (x, y),
                (x + cell_size, y),
                (x + cell_size, y - cell_size),
                (x, y - cell_size),
            ]
        )
        for x in cols
        for y in rows
    ]

    # Create grid GeoDataFrame
    grid = gpd.GeoDataFrame({"geometry": polygons}, crs=shp.crs)
    grid["id"] = grid.index + 1
    grid["iso3"] = iso3

    # Add centroids directly
    grid["Longitude"] = grid["geometry"].centroid.x
    grid["Latitude"] = grid["geometry"].centroid.y
    grid["Centroid"] = (
        grid["Longitude"].round(2).astype(str)
        + "W_"
        + grid["Latitude"].round(2).astype(str)
    )

    # Centroids GeoDataFrame
    grid_centroids = grid.copy()
    grid_centroids["geometry"] = grid_centroids["geometry"].centroid

    # Intersection of grid and shapefile
    adm2_grid_intersection = gpd.overlay(shp, grid, how="identity")
    adm2_grid_intersection = adm2_grid_intersection.dropna(subset=["id"])

    # Filter grid that intersects
    grid_land_overlap = grid.loc[grid["id"].isin(adm2_grid_intersection["id"])]

    # Centroids of intersection
    grid_land_overlap_centroids = grid_centroids.loc[
        grid["id"].isin(adm2_grid_intersection["id"])
    ]

    # Grids by municipality (Spatial join)
    grid_muni = gpd.sjoin(shp, grid_land_overlap, how="inner")

    # Parallelized calculation of intersection areas
    def calculate_intersection_area(row):
        grid_cell = grid_land_overlap.loc[
            grid_land_overlap.id == row.id, "geometry"
        ]
        municipality_polygon = row.geometry

        # Ensure valid geometries
        if not grid_cell.empty and municipality_polygon.is_valid:
            intersection_area = (
                grid_cell.iloc[0].intersection(municipality_polygon).area
            )
            return intersection_area
        return 0

    # Use ThreadPoolExecutor or ProcessPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        intersection_areas = list(
            executor.map(
                calculate_intersection_area, grid_muni.itertuples(index=False)
            )
        )

    # Add intersection area to grid_muni
    grid_muni["intersection_area"] = intersection_areas

    # Sort by intersection area and drop duplicates
    grid_muni = grid_muni.sort_values("intersection_area", ascending=False)
    grid_muni_total = grid_muni.drop_duplicates(subset="id", keep="first")
    grid_muni_total = grid_muni_total[["id", "GID_0", "GID_1", "GID_2"]]

    # Change ID naming with iso3
    grid_muni_total["id"] = grid_muni_total["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid["id"] = grid["id"].apply(lambda x: iso3 + "_" + str(x))
    grid_land_overlap["id"] = grid_land_overlap["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid_centroids["id"] = grid_centroids["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )
    grid_land_overlap_centroids["id"] = grid_land_overlap_centroids[
        "id"
    ].apply(lambda x: iso3 + "_" + str(x))
    adm2_grid_intersection["id"] = adm2_grid_intersection["id"].apply(
        lambda x: iso3 + "_" + str(x)
    )

    return (
        grid,
        grid_land_overlap,
        grid_centroids,
        grid_land_overlap_centroids,
        grid_muni_total,
    )


# Parallel processing for multiple countries
def process_multiple_countries(shp, out_path):
    os.makedirs(out_path, exist_ok=True)
    countries = shp["GID_0"].unique()

    def process_country(iso3):
        try:
            # Check if output files already exist
            if (
                os.path.exists(f"{out_path}/grid_{iso3}.gpkg")
                and os.path.exists(f"{out_path}/grid_land_overlap_{iso3}.gpkg")
                and os.path.exists(f"{out_path}/grid_centroids_{iso3}.gpkg")
                and os.path.exists(
                    f"{out_path}/grid_land_overlap_centroids_{iso3}.gpkg"
                )
                and os.path.exists(
                    f"{out_path}/grid_municipality_info_{iso3}.csv"
                )
            ):
                print(f"Files for {iso3} already exist, skipping processing.")
                return

            # Call the create_grid function to process data for the country
            (
                grid,
                grid_land_overlap,
                grid_centroids,
                grid_land_overlap_centroids,
                grid_muni_total,
            ) = create_grid(shp, iso3)

            # Save to files
            grid.to_file(
                f"{out_path}/grid_{iso3}.gpkg", layer="grid", driver="GPKG"
            )
            grid_land_overlap.to_file(
                f"{out_path}/grid_land_overlap_{iso3}.gpkg",
                layer="grid_land_overlap",
                driver="GPKG",
            )
            grid_centroids.to_file(
                f"{out_path}/grid_centroids_{iso3}.gpkg",
                layer="grid_centroids",
                driver="GPKG",
            )
            grid_land_overlap_centroids.to_file(
                f"{out_path}/grid_land_overlap_centroids_{iso3}.gpkg",
                layer="grid_land_overlap_centroids",
                driver="GPKG",
            )
            grid_muni_total.to_csv(
                f"{out_path}/grid_municipality_info_{iso3}.csv", index=False
            )

            # Free memory
            (
                grid,
                grid_land_overlap,
                grid_centroids,
                grid_land_overlap_centroids,
                grid_muni_total,
            ) = [None] * 5

        except Exception as e:
            # Log the error along with the country ISO3 code
            logging.error(f"Error processing {iso3}: {e}")

    # Parallel processing with max_workers=4
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_country, countries)


if __name__ == "__main__":
    # Load global shapefile
    global_shp = gpd.read_file(
        "/home/fmoss/GLOBAL MODEL/data/SHP/gadm_410.gdb"
    )
    out_path = "/data/big/fmoss/data/GRID/"
    # Define grid level for every country available in the impact data
    process_multiple_countries(global_shp, out_path)
