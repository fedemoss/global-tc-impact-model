#!/usr/bin/env python3
import os
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# Function to add population data to municipality info
def add_pop_info_to_impact_data(
    iso3, global_ids_mun, global_pop, impact_global
):
    # Select country datasets
    ids_mun_country = global_ids_mun[global_ids_mun.GID_0 == iso3]
    pop_df_country = global_pop[global_pop.iso3 == iso3]
    impact_country = impact_global[impact_global.GID_0 == iso3]

    # Merge pop and id data
    df_merge_country = ids_mun_country.merge(pop_df_country, on="id")

    # Level 1
    population_sum = (
        df_merge_country.groupby("GID_1")["population"].sum().reset_index()
    )
    gid_0_data = (
        df_merge_country.groupby("GID_1")
        .first()
        .reset_index()[["GID_1", "GID_0"]]
    )
    df_merge_gid1 = pd.merge(gid_0_data, population_sum, on="GID_1")

    # Level 2
    population_sum = (
        df_merge_country.groupby("GID_2")["population"].sum().reset_index()
    )
    gid_0_data = (
        df_merge_country.groupby("GID_2")
        .first()
        .reset_index()[["GID_2", "GID_1", "GID_0"]]
    )
    df_merge_gid2 = pd.merge(gid_0_data, population_sum, on="GID_2")

    # Add information as new columns
    impact_plus = impact_country.merge(
        df_merge_gid1[["population", "GID_1"]], on="GID_1", how="left"
    ).rename({"population": "population_gid1"}, axis=1)

    impact_plus = impact_plus.merge(
        df_merge_gid2[["population", "GID_2"]], on="GID_2", how="left"
    ).rename({"population": "population_gid2"}, axis=1)
    return impact_plus, df_merge_country


# Function to transform impact data to grid-based impact data
def impact_to_grid(impact_merged_adm):
    df_events = impact_merged_adm.copy()
    # Iterate for every event
    impact_data_grid_no_weather = pd.DataFrame()
    for typhoon_id in df_events["DisNo."].unique():
        # Select event
        df_event = df_events[df_events["DisNo."] == typhoon_id]
        df_event_dmg_with_pop = df_event[
            (df_event["population"] > 1) & (df_event["Total Affected"] != 0)
        ].copy()
        # Grid cells id of affected regions
        ids_dmg = df_event_dmg_with_pop.id
        # Total pop of country
        TOTAL_POP = df_event["population"].sum()
        # Total pop of region affected
        TOTAL_POP_REG = df_event_dmg_with_pop["population"].sum()

        # Perc of total pop affected in the area affected
        perc_dmg = (
            100
            * df_event_dmg_with_pop["Total Affected"].unique()
            / df_event_dmg_with_pop["population"].sum()
        )
        # # If, for some reason, there are >100% dmg in some cells, set the dmg to 100%
        # if perc_dmg > 1:
        #     perc_dmg = 1

        """     Different grid disagraggation definitions       """

        # Total pop affected by grid
        # df_event_dmg_with_pop.loc[:, 'affected_pop_grid'] = df_event_dmg_with_pop['population'] * perc_dmg / 100
        # # % of affection with respect to the total pop of the country
        # df_event_dmg_with_pop.loc[:, 'perc_affected_pop_grid_country'] = 100 * df_event_dmg_with_pop['Total Affected'] / TOTAL_POP
        # # % of affection with respect to the total pop of the region affected
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_region"] = (
            100 * df_event_dmg_with_pop["Total Affected"] / TOTAL_POP_REG
        )
        # # % of affection with respect to the total pop of grid
        # df_event_dmg_with_pop.loc[:, 'perc_affected_pop_grid_grid'] = 100 * df_event_dmg_with_pop['Total Affected'] / df_event_dmg_with_pop['population']
        # % of affection based on classic aproximated definition.
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid"] = (
            100
            * df_event_dmg_with_pop["population"]
            * df_event_dmg_with_pop["Total Affected"]
            / TOTAL_POP_REG**2
        )
        df_event = df_event.merge(df_event_dmg_with_pop, how="left").fillna(0)
        impact_data_grid_no_weather = pd.concat(
            [impact_data_grid_no_weather, df_event]
        )
    return impact_data_grid_no_weather


# Functions to create grid-based-impact dataset for a specific country
def process_impact_level(level, impact_plus, pop_grid):
    try:
        impact_level = impact_plus[impact_plus.level == level]
        impact_merged = impact_level.merge(
            pop_grid, on=["GID_0", "GID_1", "GID_2"]
        )[
            [
                "DisNo.",
                "sid",
                "level",
                "population",
                "id",
                "GID_0",
                "GID_1",
                "GID_2",
                "Total Affected",
            ]
        ].drop_duplicates()
        return impact_to_grid(impact_merged)
    except:
        return pd.DataFrame()


def create_impact_grid_dataset(
    iso3, global_ids_mun, global_pop, impact_global
):
    # Population and impact data
    impact_plus, pop_grid = add_pop_info_to_impact_data(
        iso3, global_ids_mun, global_pop, impact_global
    )

    # Process different levels
    impact_data_grid_adm0 = process_impact_level("ADM0", impact_plus, pop_grid)
    impact_data_grid_adm1 = process_impact_level("ADM1", impact_plus, pop_grid)
    impact_data_grid_adm2 = process_impact_level("ADM2", impact_plus, pop_grid)

    # Concatenate results
    impact_data_grid = pd.concat(
        [impact_data_grid_adm0, impact_data_grid_adm1, impact_data_grid_adm2]
    )

    return impact_data_grid


# Function to get impact to grid level for every country
def iterate_grid_impact(iso3_list, global_ids_mun, global_pop, impact_global):
    impact_data_grid_global = pd.DataFrame()
    for iso3 in iso3_list:
        impact_data_grid = create_impact_grid_dataset(
            iso3, global_ids_mun, global_pop, impact_global
        )
        impact_data_grid_global = pd.concat(
            [impact_data_grid_global, impact_data_grid]
        )
    # Reset index
    impact_data_grid_global = impact_data_grid_global.reset_index(drop=True)

    # Fix > 100% affected pop regions for specific events
    impact_data_grid_global.loc[
        impact_data_grid_global.perc_affected_pop_grid_region > 100,
        "perc_affected_pop_grid_region",
    ] = 100

    return impact_data_grid_global


if __name__ == "__main__":
    # Load municipality info dataset
    global_ids_mun = pd.read_csv(
        "/data/big/fmoss/data/GRID/merged/global_grid_municipality_info.csv"
    )
    # Load population dataset
    global_pop = pd.read_csv(
        "/data/big/fmoss/data/Worldpop/grid_data/merged/global_grid_worldpop.csv"
    )
    # Load impact data
    impact_global = pd.read_csv("/data/big/fmoss/data/EMDAT/impact_data.csv")

    # Get impact data to grid level
    iso3_list = global_ids_mun.GID_0.unique()
    impact_data_grid_global = iterate_grid_impact(
        iso3_list=iso3_list,
        global_ids_mun=global_ids_mun,
        global_pop=global_pop,
        impact_global=impact_global,
    )
    # Save data
    impact_data_grid_global.to_csv(
        "/data/big/fmoss/data/EMDAT/global_grid_impact_data.csv",
        ignore_index=True,
    )
