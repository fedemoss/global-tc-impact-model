#!/usr/bin/env python3
import os

import geopandas as gpd
import numpy as np
import pandas as pd

data_path = "/data/big/fmoss/data/SHDI"
file = "SHDI-SGDI-Total 8.0.csv"


def compute_min_shdi_per_region(gadm, shdi):
    """
    Assigns each SHDI polygon to a GADM region using a spatial join,
    computes the minimum SHDI per GID_1 region, and merges it back into the GADM structure.

    Parameters:
    gadm (GeoDataFrame): GADM administrative level 1 regions.
    shdi (GeoDataFrame): SHDI polygons with SHDI values.

    Returns:
    GeoDataFrame: GADM regions with the minimum SHDI values per GID_1.
    """
    # Make copies to avoid modifying original data
    gadm = gpd.GeoDataFrame(gadm, geometry="geometry").copy()
    shdi = gpd.GeoDataFrame(shdi, geometry="geometry").copy()

    # Ensure CRS matches before spatial operations
    gadm = gadm.to_crs(shdi.crs)

    # Perform spatial join: assigning each fine-grained SHDI polygon to a coarser GADM region
    shdi_joined = gpd.sjoin(shdi, gadm, how="right", predicate="touches")

    # Compute min SHDI per GID_1 region
    shdi_min = shdi_joined.groupby("GID_1")["shdi"].min().reset_index()

    # Merge the min SHDI values back into the original GADM structure
    gadm_merged = gadm.merge(shdi_min, on="GID_1", how="left")

    return gadm_merged


def interpolate_missing_shdi(admin_shdi_data):
    """
    Removes countries where all SHDI values are NaN and applies IDW interpolation
    within each administrative region to estimate missing SHDI values.

    Parameters:
    admin_shdi_data (GeoDataFrame): Administrative regions with SHDI values.

    Returns:
    GeoDataFrame: Administrative regions with interpolated SHDI values.
    """
    df = gpd.GeoDataFrame(admin_shdi_data, geometry="geometry").copy()
    valid_countries = df.groupby("country_id")["shdi"].transform(
        lambda x: x.notna().any()
    )
    df_filtered = df[valid_countries].copy()

    def apply_idw(group):
        known_points = group.dropna(subset=["shdi"])[["geometry", "shdi"]]

        if known_points.empty:
            return group

        known_coords = np.array(
            [
                (geom.centroid.x, geom.centroid.y)
                for geom in known_points.geometry
            ]
        )
        known_values = np.array(known_points["shdi"].values, dtype=float)

        missing_points = group[group["shdi"].isna()][["geometry"]]
        missing_coords = np.array(
            [
                (geom.centroid.x, geom.centroid.y)
                for geom in missing_points.geometry
            ]
        )

        if missing_points.empty:
            return group

        tree = cKDTree(known_coords)
        k = min(3, len(known_coords))

        if k == 0:
            return group

        distances, indices = tree.query(missing_coords, k=k)

        if k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        neighbor_values = np.array([known_values[idx] for idx in indices])

        def idw(dist, vals):
            vals = np.array(vals, dtype=float)
            weights = 1 / np.maximum(dist, 1e-5)
            return np.sum(weights * vals, axis=1) / np.sum(weights, axis=1)

        group.loc[group["shdi"].isna(), "shdi"] = idw(
            distances, neighbor_values
        )

        return group

    df_interpolated = df_filtered.groupby(
        "country_id", group_keys=False
    ).apply(apply_idw)
    return df_interpolated


if __name__ == "__main__":
    # Load shdi
    shdi = pd.read_csv(f"{data_path}/{file}", low_memory=False)[
        ["iso_code", "year", "GDLCODE", "level", "shdi"]
    ]
    shdi = shdi[shdi.year == 2020].reset_index(drop=True)

    # Load shdi associated shapefile and create shdi_gdf
    shp_file = "GDL Shapefiles V6.4.zip"
    gdf = gpd.read_file(f"{data_path}/{shp_file}")
    shdi_gpd = shdi.merge(gdf)

    # Load GADM shp
    gadm_shp_adm1 = gpd.read_file("/data/big/fmoss/data/SHP/GADM_adm1.gpkg")

    # Load grid cells
    grid_cells = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )

    # Merge GADM and SHDI associated geodata
    gadm_merged = compute_min_shdi_per_region(gadm_shp_adm1, shdi_gpd)
    gadm_merged.to_file(
        "/data/big/fmoss/data/SHDI/GADM_adm1_SHDI.gpkg", driver="GPKG"
    )

    # Interpolate missing SHDI values
    gadm_interpolated = interpolate_missing_shdi(gadm_merged)

    # To grid level
    grid_shdi = grid_cells.merge(
        gadm_interpolated[["GID_0", "GID_1", "shdi"]],
        on=["GID_0", "GID_1"],
        how="left",
    ).sort_values("GID_0", ignore_index=True)
    os.mkdir("/data/big/fmoss/data/SHDI/grid_data/")
    grid_shdi[["GID_0", "GID_1", "id", "shdi"]].to_csv(
        "/data/big/fmoss/data/SHDI/grid_data/shdi_global_grid_index.csv",
        index=False,
    )
