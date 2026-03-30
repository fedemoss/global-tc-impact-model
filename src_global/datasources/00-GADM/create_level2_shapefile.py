#!/usr/bin/env python3
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src_global.utils import blob, constant

PROJECT_PREFIX = "global_model"


# Function to group dataset on specific level
def group_shp(global_shp_red):
    # Group by the level
    grouped = global_shp_red.groupby(["GID_0", "GID_1", "GID_2"])

    # Aggregate geometries using unary_union
    agg_geometries = grouped["geometry"].agg(lambda x: x.unary_union)

    # Create a new GeoDataFrame with the aggregated geometries
    agg_df = gpd.GeoDataFrame(agg_geometries, geometry="geometry")

    # Reset index to get level as a column again
    agg_df.reset_index(inplace=True)

    # To GPD
    agg_df_shp = gpd.GeoDataFrame(agg_df, geometry="geometry")
    return agg_df_shp


def create_adm2_shp(iso3_list):
    # Load data
    global_shp = blob.load_gpkg(name=f"{PROJECT_PREFIX}/SHP/gadm_410-gpkg.zip")

    # Process data
    global_shp_red = global_shp[
        ["GID_0", "GID_1", "GID_2", "GID_3", "GID_4", "GID_5", "geometry"]
    ]
    global_shp_red = global_shp_red[global_shp_red["GID_0"].isin(iso3_list)]

    # Group to level 2 (if possible)
    agg_df_shp_adm2 = group_shp(global_shp_red[global_shp_red.GID_2 != ""])
    # For the missing data (countries with no adm2 data), keep the level 1 info
    agg_df_shp_adm1 = global_shp_red[global_shp_red.GID_2 == ""][
        ["GID_0", "GID_1", "GID_2", "geometry"]
    ].reset_index()
    # Concatenate
    agg_df_shp = pd.concat([agg_df_shp_adm1, agg_df_shp_adm2]).reset_index(
        drop=True
    )

    datasets = {
        "/SHP/global_shapefile_GID_adm2.gpkg": agg_df_shp,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Save and upload GeoPackage datasets
        for filename, gdf in datasets.items():
            local_file_path = (
                temp_dir_path / filename.split("/")[-1]
            )  # Save in temp_dir with filename only
            gdf.to_file(local_file_path, driver="GPKG")
            blob_name = f"{PROJECT_PREFIX}{filename}"

            with open(local_file_path, "rb") as file:
                data = file.read()
                blob.upload_blob_data(
                    blob_name=blob_name, data=data, prod_dev="dev"
                )


if __name__ == "__main__":
    iso3_list = constant.iso3_list
    create_adm2_shp(iso3_list)
