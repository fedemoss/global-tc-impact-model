#!/usr/bin/env python3
import ast
import io
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd


# Function to load and clean emdat dataset
def clean_emdat(impact_data):
    impact_data_global = (
        impact_data[
            [
                "DisNo.",
                "Total Affected",
                "Start Year",
                "Start Month",
                "iso3",
                "Country",
                "Admin Units",
                "sid",
                "Event Name",
            ]
        ]
        .dropna(subset="Total Affected")
        .sort_values(["iso3", "sid", "Start Year"])
        .reset_index(drop=True)
    )

    return impact_data_global


# Fuction to propertly create the "affected regions" column
def get_item(x):
    try:
        list_locations = ast.literal_eval(x)
        adm1_pcodes = []
        adm2_pcodes = []
        adm1 = False
        adm2 = False

        for el in list_locations:
            try:
                adm1_pcode = el["adm1_code"]
                adm1_pcodes.append(adm1_pcode)
                adm1 = True
            except KeyError:
                adm2_pcode = el["adm2_code"]
                adm2_pcodes.append(adm2_pcode)
                adm2 = True

        if adm1:
            level = "ADM1"
            return {"level": level, "affected_regions": adm1_pcodes}
        elif adm2:
            level = "ADM2"
            return {"level": level, "affected_regions": adm2_pcodes}
    except:
        level = "ADM0"
        return {"level": level, "affected_regions": []}


# Fuction to propertly explode the emdata databaset on the "affected regions" column
def process_impact_data(guil_shp, impact_data):
    """For the ADM1 level events"""

    # Explode impact data
    adm1_impact_exploded = (
        impact_data[impact_data.level == "ADM1"]
        .explode("regions_affected")
        .reset_index(drop=True)
    )
    # Add geometry feature by merging with gaul2015 shapefile
    adm1_impact_complete = guil_shp.merge(
        adm1_impact_exploded,
        left_on="ADM1_CODE",
        right_on="regions_affected",
        how="right",
    )[
        adm1_impact_exploded.columns.to_list()
        + ["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"]
    ].drop(
        "regions_affected", axis=1
    )  # .drop_duplicates(['sid', 'ADM1_CODE'])
    adm1_impact_complete = gpd.GeoDataFrame(
        adm1_impact_complete, geometry="geometry"
    )

    # Calculate centroids of the geometries in adm2_impact_complete
    adm1_impact_complete["centroid"] = adm1_impact_complete.geometry.centroid

    """For the ADM2 level events"""
    # Explode impact data
    adm2_impact_exploded = impact_data[impact_data.level == "ADM2"].explode(
        "regions_affected"
    )
    # Add geometry feature by merging with gaul2015 shapefile
    adm2_impact_complete = guil_shp.merge(
        adm2_impact_exploded,
        left_on="ADM2_CODE",
        right_on="regions_affected",
        how="right",
    )[
        adm2_impact_exploded.columns.to_list()
        + ["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"]
    ].drop(
        "regions_affected", axis=1
    )
    adm2_impact_complete = gpd.GeoDataFrame(
        adm2_impact_complete, geometry="geometry"
    )

    # Calculate centroids of the geometries in adm2_impact_complete
    adm2_impact_complete["centroid"] = adm2_impact_complete.geometry.centroid

    """For the ADM0 level events"""
    adm0_impact_complete = impact_data[impact_data.level == "ADM0"]

    # Concatenate
    impact_data_complete_geo = pd.concat(
        [adm1_impact_complete, adm2_impact_complete, adm0_impact_complete]
    )
    # Clean
    impact_data_complete_geo = impact_data_complete_geo.drop(
        [
            "ADM1_CODE",
            "ADM2_CODE",
            "ADM0_NAME",
            "geometry",
            "regions_affected",
        ],
        axis=1,
    )
    impact_data_complete_geo = impact_data_complete_geo.rename(
        {"centroid": "geometry"}, axis=1
    )

    return impact_data_complete_geo


# Function to manually add missing sid when it's possible
def add_missing_sid(df_impact):
    # Manually create list of sid's
    missing_sid_subset = df_impact[df_impact.sid.isna()][
        ["Event Name", "DisNo.", "Start Year", "GID_0"]
    ].drop_duplicates()
    # I want at least to have an EVENT NAME
    missing_sid_subset = missing_sid_subset.dropna(
        subset="Event Name"
    ).reset_index(drop=True)
    # Create list by going into https://ncics.org/ibtracs/index.php?
    missing_sid_list = [
        "2022295N13093",  # Sitrang
        "2022254N24143",  # Nanmadol
        "2022020S13059",  # Ana
        "2022025S11091",  # Batsirai
        "2022042S12063",  # Dumako
        "2022047S15073",  # Emnati
        "2022020S13059",  # Ana
        "2022042S12063",  # Dumako
        None,  # Ineng
        "2022099N11128",  # Megi
        "2022232N18131",  # Ma-on
        "2022285N17140",  # Nesat
        "2022285N12116",  # Sonca
        "2022338N05100",  # Mandous
        "2022065S16055",  # Gombe
        "2022110S12051",  # Jasmine
        "2022020S13059",  # Ana
        None,  # Winnie
        "2022008S17173",  # Cody
        "2022180N15130",  # Aere
        "2022263N18137",  # Talas
        "2022239N22150",  # Hinnamnor
        "2022065S16055",  # Gombe
        "2022025S11091",  # Batsirai
        "2022299N11134",  # Nalgae
        "2022285N17140",  # Nesat
        "2022020S13059",  # Ana
    ]
    # Updated Dataframe of missing ones
    missing_sid_subset["sid"] = missing_sid_list
    missing_sid_subset = missing_sid_subset[["DisNo.", "sid"]].dropna()

    # Merge to get the missing sid values
    merge_sid = df_impact.merge(
        missing_sid_subset, on="DisNo.", how="left", suffixes=("", "_missing")
    )
    # Fill the NaN values in the sid column with the corresponding values from the merged DataFrame
    merge_sid["sid"] = merge_sid["sid"].fillna(merge_sid["sid_missing"])
    # Drop the temporary 'sid_missing' column
    df_impact_complete_fixed = merge_sid.drop(columns=["sid_missing"])
    # Drop all the rows with no sid column --> no TC matching is a no-go
    df_impact_complete_fixed = df_impact_complete_fixed.dropna(
        subset="sid"
    ).reset_index(drop=True)
    return df_impact_complete_fixed


# Function to geolocate the emdat database (GID codes by GDAM shapefile)
def geolocate_impact(impact_data, GID_shapefile, GUIL_shapefile):
    global_shp = GID_shapefile
    guil_shp = GUIL_shapefile[
        ["ADM2_CODE", "ADM1_CODE", "ADM0_NAME", "geometry"]
    ]

    # Create "affected regions" column
    impact_data[["level", "regions_affected"]] = impact_data[
        "Admin Units"
    ].apply(lambda x: pd.Series(get_item(x)))
    impact_data = impact_data.drop(columns="Admin Units")
    # Explode on this column
    impact_data_complete_geo = process_impact_data(guil_shp, impact_data)

    # Create geometry column
    geo_impact_data = gpd.GeoDataFrame(
        impact_data_complete_geo, geometry="geometry"
    )

    # Set the CRS for geo_impact_data if not already set
    geo_impact_data.set_crs(
        epsg=4326, inplace=True, allow_override=True
    )  # Assuming WGS84 (epsg:4326)

    # Ensure both GeoDataFrames use the same CRS
    geo_impact_data.to_crs(global_shp.crs, inplace=True)

    # We have 2 cases
    geo_impact_data_adm0 = geo_impact_data[geo_impact_data.level == "ADM0"]
    geo_impact_data_subnational = geo_impact_data[
        geo_impact_data.level != "ADM0"
    ]

    # Add 2nd geometry column
    global_shp["shp_geometry"] = global_shp["geometry"]

    # Merge with shapefile
    merged_gdf = gpd.sjoin(
        geo_impact_data_subnational, global_shp, how="left", predicate="within"
    )

    # Identify points that were not matched
    unmatched_points = merged_gdf[merged_gdf["index_right"].isna()]

    # Find the nearest polygon for unmatched points
    unmatched_points = unmatched_points.drop(columns=["index_right"])
    global_shp["centroid"] = global_shp.centroid

    # Calculate distance to each polygon centroid and find the nearest
    nearest_polygons = gpd.GeoDataFrame(
        global_shp, geometry="centroid"
    ).sjoin_nearest(
        unmatched_points[geo_impact_data.columns],
        how="left",
        distance_col="distance",
    )
    # Keep the nearest point
    nearest_polygons = nearest_polygons.sort_values(
        "distance"
    ).drop_duplicates(subset="index_right", keep="first")
    # If the country does not match, dont consider the event. (it just affected 10 datapoints, quite good)
    nearest_polygons = nearest_polygons[
        nearest_polygons["GID_0"] == nearest_polygons["iso3"]
    ]
    nearest_polygons = nearest_polygons.drop(columns=["centroid", "distance"])

    # Merge back the geometry and other columns from agg_df_shp
    merged_gdf = merged_gdf.dropna(
        subset=["GID_0", "GID_1"]
    )  # .drop('centroid', axis=1)
    merged_gdf = pd.concat([merged_gdf, nearest_polygons], ignore_index=True)
    merged_gdf["geometry"] = merged_gdf["shp_geometry"]
    merged_gdf = merged_gdf.drop(["shp_geometry", "index_right"], axis=1)

    # Add adm0 impact
    merged_gdf = pd.concat([merged_gdf, geo_impact_data_adm0])
    # adm0 events dont have gid0 info, but this is esentially the iso3
    merged_gdf["GID_0"] = merged_gdf["GID_0"].fillna(merged_gdf.iso3)

    # Create "reduced" dataset with only relevant information
    reduced_impact_dataset = merged_gdf[
        [
            "Event Name",
            "DisNo.",
            "sid",
            "Total Affected",
            "level",
            "Start Year",
            "Start Month",
            # "Country",
            "GID_0",
            "GID_1",
            "GID_2",
        ]
    ].drop_duplicates()
    reduced_shp = global_shp[["GID_0", "GID_1", "GID_2"]]

    # Iterate through every event to get non affected areas
    df_impact_complete = pd.DataFrame()
    for event in reduced_impact_dataset["DisNo."].unique():
        df_event = reduced_impact_dataset[
            reduced_impact_dataset["DisNo."] == event
        ]
        country = df_event.GID_0.unique()[0]
        level = df_event.level.unique()[0]
        df_loc = reduced_shp[reduced_shp.GID_0 == country]
        # Merge
        if level == "ADM1":
            df_event = df_event.drop("GID_2", axis=1)
            reduced_merged = pd.merge(
                df_event, df_loc, on=["GID_0", "GID_1"], how="right"
            )
        elif level == "ADM2":
            reduced_merged = pd.merge(
                df_event, df_loc, on=["GID_0", "GID_1", "GID_2"], how="right"
            )
        elif level == "ADM0":
            df_event = df_event.drop(["GID_1", "GID_2"], axis=1)
            reduced_merged = pd.merge(
                df_event, df_loc, on=["GID_0"], how="right"
            )

        # Sort values for ffill
        reduced_merged = reduced_merged.sort_values(
            by="Total Affected", ascending=False
        )
        # Fill nans
        reduced_merged["DisNo."] = reduced_merged["DisNo."].ffill()
        reduced_merged["sid"] = reduced_merged["sid"].ffill()
        reduced_merged["Start Year"] = reduced_merged["Start Year"].ffill()
        reduced_merged["Start Month"] = reduced_merged["Start Month"].ffill()
        reduced_merged["Event Name"] = reduced_merged["Event Name"].ffill()
        reduced_merged["level"] = reduced_merged["level"].ffill()
        # reduced_merged["Country"] = reduced_merged["Country"].ffill()
        reduced_merged["Total Affected"].fillna(
            0, inplace=True
        )  # Fill 'Total Affected' with 0
        # Sort back by GID codes
        reduced_merged = reduced_merged.drop_duplicates().sort_values(
            ["GID_1", "GID_2"]
        )
        df_impact_complete = pd.concat([df_impact_complete, reduced_merged])
    # Reset index
    df_impact_complete = df_impact_complete.reset_index(drop=True)
    # Add missing sid data
    df_impact_complete_fixed = add_missing_sid(df_impact_complete)

    return df_impact_complete_fixed


if __name__ == "__main__":
    # Clean EMDAT dataset
    emdat_data = pd.read_csv(
        "/home/fmoss/GLOBAL MODEL/data/EMDAT/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
    )
    impact_data = clean_emdat(emdat_data)
    # Create impact dataset at adm2 level for event
    GID_shapefile = gpd.read_file("/data/big/fmoss/data/SHP/GADM_adm2.gpkg")
    GUIL_shapefile = gpd.read_file(
        "/data/big/fmoss/data/SHP/global_shapefile_GUIL_adm2.gpkg"
    )
    df_impact = geolocate_impact(impact_data, GID_shapefile, GUIL_shapefile)
    out_path = "/data/big/fmoss/data/EMDAT/impact_data.csv"
    os.makedirs("/data/big/fmoss/data/EMDAT", exist_ok=True)
    df_impact.to_csv(out_path, index=False)
