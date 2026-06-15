import os
import ast
import logging
import gc
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from src.config import INPUT_DIR, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------
# Helper Utilities
# -------------------------------------------------------------------

def clean_emdat(df):
    """Filters and cleans the raw EM-DAT dataset for tropical cyclones."""
    columns = [
        "DisNo.", "Total Affected", "Start Year", "Start Month",
        "iso3", "Country", "Admin Units", "sid", "Event Name"
    ]
    return (
        df[columns]
        .dropna(subset=["Total Affected"])
        .sort_values(["iso3", "sid", "Start Year"])
        .reset_index(drop=True)
    )

def parse_admin_units(x):
    """Parses the 'Admin Units' string to extract ADM levels and PCodes."""
    try:
        list_locations = ast.literal_eval(x)
        adm1_pcodes, adm2_pcodes = [], []
        adm1_found = adm2_found = False

        for el in list_locations:
            if "adm1_code" in el:
                adm1_pcodes.append(el["adm1_code"])
                adm1_found = True
            elif "adm2_code" in el:
                adm2_pcodes.append(el["adm2_code"])
                adm2_found = True

        if adm1_found:
            return {"level": "ADM1", "regions_affected": adm1_pcodes}
        elif adm2_found:
            return {"level": "ADM2", "regions_affected": adm2_pcodes}
        return {"level": "ADM0", "regions_affected": []}
    except Exception:
        return {"level": "ADM0", "regions_affected": []}

def process_impact_geometries(gaul_shp, impact_data):
    """Maps impact records to GAUL geometries and ensures result is hashable for deduplication."""
    results = []
    for level in ["ADM1", "ADM2"]:
        subset = impact_data[impact_data.level == level].copy()
        if subset.empty:
            continue

        exploded = subset.explode("regions_affected").reset_index(drop=True)
        code_col = f"{level}_CODE"

        merged = gaul_shp.merge(exploded, left_on=code_col, right_on="regions_affected", how="right")

        cols_to_keep = exploded.columns.tolist() + ["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"]
        merged = merged[cols_to_keep].drop(columns=["regions_affected"], errors="ignore")
        merged = gpd.GeoDataFrame(merged, geometry="geometry")
        merged["centroid"] = merged.geometry.centroid
        results.append(merged)

    adm0_subset = impact_data[impact_data.level == "ADM0"].copy()
    if not adm0_subset.empty:
        results.append(adm0_subset.drop(columns=["regions_affected"], errors="ignore"))

    combined = pd.concat(results, ignore_index=True)
    combined = combined.drop(["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"], axis=1, errors="ignore")

    if "centroid" in combined.columns:
        combined = combined.rename(columns={"centroid": "geometry"})

    return combined

def add_missing_sid(df_impact):
    """Patches known missing SIDs for 2022 events and drops rows with no SID."""
    missing_sid_subset = df_impact[df_impact.sid.isna()][
        ["Event Name", "DisNo.", "Start Year", "GID_0"]
    ].drop_duplicates()
    missing_sid_subset = missing_sid_subset.dropna(subset="Event Name").reset_index(drop=True)

    missing_sid_list = [
        "2022295N13093",  # Sitrang
        "2022254N24143",  # Nanmadol
        "2022020S13059",  # Ana
        "2022025S11091",  # Batsirai
        "2022042S12063",  # Dumako
        "2022047S15073",  # Emnati
        "2022020S13059",  # Ana
        "2022042S12063",  # Dumako
        None,             # Ineng
        "2022099N11128",  # Megi
        "2022232N18131",  # Ma-on
        "2022285N17140",  # Nesat
        "2022285N12116",  # Sonca
        "2022338N05100",  # Mandous
        "2022065S16055",  # Gombe
        "2022110S12051",  # Jasmine
        "2022020S13059",  # Ana
        None,             # Winnie
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

    # Only apply patch if list length matches (guards against future EM-DAT updates)
    if len(missing_sid_subset) == len(missing_sid_list):
        missing_sid_subset["sid"] = missing_sid_list
        missing_sid_subset = missing_sid_subset[["DisNo.", "sid"]].dropna()

        merge_sid = df_impact.merge(
            missing_sid_subset, on="DisNo.", how="left", suffixes=("", "_missing")
        )
        merge_sid["sid"] = merge_sid["sid"].fillna(merge_sid["sid_missing"])
        df_impact = merge_sid.drop(columns=["sid_missing"])

    return df_impact.dropna(subset="sid").reset_index(drop=True)


# -------------------------------------------------------------------
# Primary Pipeline Functions
# -------------------------------------------------------------------

def preprocess_emdat_events(iso3_filter=None):
    """Main function to clean, geolocate, and spatially expand the EM-DAT database."""
    emdat_path = INPUT_DIR / "EMDAT" / "emdat.csv"
    gadm_path  = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    gaul_path  = INPUT_DIR / "SHP" / "global_shapefile_GAUL_adm2.gpkg"
    out_path   = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    impact_data = clean_emdat(pd.read_csv(emdat_path))
    if iso3_filter:
        impact_data = impact_data[impact_data["iso3"] == iso3_filter].reset_index(drop=True)
        logging.info(f"Filtered to {iso3_filter}: {len(impact_data)} rows, {impact_data['DisNo.'].nunique()} events")

    # Load GAUL shapefile — uses ADM codes that match EM-DAT's Admin Units field
    gaul_shp = gpd.read_file(gaul_path)[["ADM2_CODE", "ADM1_CODE", "ADM0_NAME", "geometry"]]

    impact_data[["level", "regions_affected"]] = impact_data["Admin Units"].apply(
        lambda x: pd.Series(parse_admin_units(x))
    )
    impact_data = impact_data.drop(columns="Admin Units")

    geo_impact_data = process_impact_geometries(gaul_shp, impact_data)
    del gaul_shp
    gc.collect()

    global_shp = gpd.read_file(gadm_path)
    geo_impact_data = gpd.GeoDataFrame(geo_impact_data, geometry="geometry", crs="EPSG:4326")
    geo_impact_data.to_crs(global_shp.crs, inplace=True)

    subnational = geo_impact_data[geo_impact_data.level != "ADM0"].copy()
    national    = geo_impact_data[geo_impact_data.level == "ADM0"].copy()

    global_shp["shp_geometry"] = global_shp["geometry"]
    merged_gdf = gpd.sjoin(subnational, global_shp, how="left", predicate="within")

    unmatched = merged_gdf[merged_gdf["index_right"].isna()].drop(columns=["index_right"])
    if not unmatched.empty:
        global_shp["centroid"] = global_shp.centroid
        nearest = gpd.GeoDataFrame(global_shp, geometry="centroid").sjoin_nearest(
            unmatched[geo_impact_data.columns.tolist()], how="left", distance_col="distance"
        )
        # Keep only the closest match per unmatched point
        nearest = nearest.sort_values("distance").drop_duplicates(subset="index_right", keep="first")
        nearest = nearest[nearest["GID_0"] == nearest["iso3"]]
        merged_gdf = pd.concat([
            merged_gdf.dropna(subset=["GID_0", "GID_1"]),
            nearest.drop(columns=["centroid", "distance"])
        ])

    merged_gdf["geometry"] = merged_gdf["shp_geometry"]
    merged_gdf = pd.concat([
        merged_gdf.drop(["shp_geometry", "index_right"], axis=1, errors="ignore"),
        national
    ])
    merged_gdf["GID_0"] = merged_gdf["GID_0"].fillna(merged_gdf.iso3)

    reduced_shp = global_shp[["GID_0", "GID_1", "GID_2"]].copy()
    del global_shp
    gc.collect()

    final_rows = []
    for event_id in merged_gdf["DisNo."].unique():
        df_event = merged_gdf[merged_gdf["DisNo."] == event_id].copy()
        country = df_event.GID_0.unique()[0]
        level   = df_event.level.unique()[0]
        df_loc  = reduced_shp[reduced_shp.GID_0 == country]

        if level == "ADM1":
            merged = pd.merge(df_event.drop("GID_2", axis=1, errors="ignore"), df_loc, on=["GID_0", "GID_1"], how="right")
        elif level == "ADM2":
            merged = pd.merge(df_event, df_loc, on=["GID_0", "GID_1", "GID_2"], how="right")
        else:
            merged = pd.merge(df_event.drop(["GID_1", "GID_2"], axis=1, errors="ignore"), df_loc, on=["GID_0"], how="right")

        merged = merged.sort_values(by="Total Affected", ascending=False)
        cols_to_fill = ["DisNo.", "sid", "Start Year", "Start Month", "Event Name", "level"]
        for col in cols_to_fill:
            merged[col] = merged[col].ffill()

        merged["Total Affected"] = merged["Total Affected"].fillna(0)
        final_rows.append(merged.drop_duplicates().sort_values(["GID_1", "GID_2"]))

    result = pd.concat(final_rows, ignore_index=True)
    del final_rows
    gc.collect()

    # Patch known missing SIDs and remove rows with no TC match
    result = add_missing_sid(result)

    result.drop(columns=["geometry"], errors="ignore").to_csv(out_path, index=False)

def calculate_grid_impact(iso3_filter=None):
    """
    Disaggregates EM-DAT impact data to the grid level based on regional population.

    The function calculates the 'perc_affected_pop_grid_region' metric by determining
    the total population within the administrative units specifically affected by
    an event and distributing the impact accordingly.
    """
    in_path  = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    out_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    pop_dir  = OUTPUT_DIR / "Worldpop" / "grid_data"

    # Path to the administrative mapping (created during SHDI processing)
    grid_admin_dir = OUTPUT_DIR / "SHDI" / "grid_data"

    if not in_path.exists():
        return

    df_events = pd.read_csv(in_path)
    if iso3_filter:
        df_events = df_events[df_events["GID_0"] == iso3_filter].reset_index(drop=True)
    impact_data_grid = []

    for event_id in df_events["DisNo."].unique():
        # 1. Prepare event metadata
        df_event = df_events[df_events["DisNo."] == event_id].copy()
        iso   = df_event["GID_0"].iloc[0]
        level = df_event["level"].iloc[0]  # ADM1 or ADM2

        # 2. Load on-demand grid and administrative mapping
        pop_file       = pop_dir / f"population_grid_{iso}.csv"
        admin_map_file = grid_admin_dir / f"shdi_grid_{iso}.csv"

        if not (pop_file.exists() and admin_map_file.exists()):
            continue

        df_pop   = pd.read_csv(pop_file)
        df_admin = pd.read_csv(admin_map_file)[["id", "GID_1", "GID_2"]]

        # Combine grid population with administrative IDs
        df_grid = df_pop.merge(df_admin, on="id", how="left")

        # 3. Merge grid with event based on administrative sub-levels
        # Drop GID_2 from the event side for ADM1 merges — df_grid already carries it
        # from the admin map; keeping both causes GID_2_x/GID_2_y suffix collision.
        merge_keys = ["GID_1"] if level == "ADM1" else ["GID_1", "GID_2"]
        event_drop = ["iso3"] + (["GID_2"] if level == "ADM1" else [])
        df_merged = df_grid.merge(
            df_event.drop(columns=event_drop, errors="ignore"),
            on=merge_keys,
            how="left"
        )

        # Fill event metadata for grid cells not matched to any affected admin unit
        meta_cols = ["DisNo.", "sid", "Start Year", "Start Month", "Event Name", "level", "GID_0"]
        for col in meta_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(df_event[col].iloc[0])
        df_merged["Total Affected"] = df_merged["Total Affected"].fillna(0)

        # 4. Calculate Regional Metrics
        affected_mask = (df_merged["Total Affected"] > 0) & (df_merged["population"] > 0)
        df_affected = df_merged[affected_mask].copy()

        if df_affected.empty:
            continue

        # total_pop_reg: sum of population in all affected subnational units
        total_pop_reg = df_affected.drop_duplicates(subset=["id"])["population"].sum()

        # perc_affected_pop_grid_region: (100 * Total Affected) / Total Regional Population
        df_affected["perc_affected_pop_grid_region"] = (
            100 * df_affected["Total Affected"]
        ) / total_pop_reg

        # 5. Finalize and append
        df_final = df_merged.merge(
            df_affected[["id", "perc_affected_pop_grid_region"]],
            on="id",
            how="left"
        ).fillna({"perc_affected_pop_grid_region": 0})

        impact_data_grid.append(df_final)

        del df_pop, df_admin, df_grid, df_merged, df_affected

    if impact_data_grid:
        pd.concat(impact_data_grid, ignore_index=True).to_csv(out_path, index=False)

def process_emdat_events(iso3_filter=None):
    preprocess_emdat_events(iso3_filter=iso3_filter)
    calculate_grid_impact(iso3_filter=iso3_filter)

if __name__ == "__main__":
    # Example (case study)
    process_emdat_events(iso3_filter="ATG")
