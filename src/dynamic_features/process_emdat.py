import ast
import gc
import logging
import os

import geopandas as gpd
import pandas as pd

from src.config import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

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

def process_impact_geometries(guil_shp, impact_data):
    """Maps impact records to geometries and ensures result is hashable for deduplication."""
    results = []
    for level in ["ADM1", "ADM2"]:
        subset = impact_data[impact_data.level == level].copy()
        if subset.empty:
            continue
            
        exploded = subset.explode("regions_affected").reset_index(drop=True)
        code_col = f"{level}_CODE"
        
        merged = guil_shp.merge(exploded, left_on=code_col, right_on="regions_affected", how="right")
        
        cols_to_keep = exploded.columns.tolist() + ["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"]
        merged = merged[cols_to_keep].drop(columns=["regions_affected"], errors="ignore")
        merged = gpd.GeoDataFrame(merged, geometry="geometry")
        merged["centroid"] = merged.geometry.centroid
        results.append(merged)

    adm0_subset = impact_data[impact_data.level == "ADM0"].copy()
    if not adm0_subset.empty:
        results.append(adm0_subset.drop(columns=["regions_affected"], errors="ignore"))

    combined = pd.concat(results, ignore_index=True)
    combined = combined.drop(["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"], axis=1, errors='ignore')
    
    if "centroid" in combined.columns:
        combined = combined.rename(columns={"centroid": "geometry"})
    
    return combined

# -------------------------------------------------------------------
# Primary Pipeline Functions
# -------------------------------------------------------------------

def preprocess_emdat_events():
    """Main function to clean, geolocate, and spatially expand the EM-DAT database."""
    emdat_path = INPUT_DIR / "EMDAT" / "emdat.csv"
    gadm_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    guil_path = INPUT_DIR / "SHP" / "global_shapefile_GUIL_adm2.zip"
    out_path = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    impact_data = clean_emdat(pd.read_csv(emdat_path))
    
    # Process GUIL
    guil_shp = gpd.read_file(f"zip://{guil_path}")
    guil_shp = guil_shp[["gaul2_code", "gaul1_code", "gaul0_name", "geometry"]].rename(
        columns={"gaul2_code": "ADM2_CODE", "gaul1_code": "ADM1_CODE", "gaul0_name": "ADM0_NAME"}
    )

    impact_data[["level", "regions_affected"]] = impact_data["Admin Units"].apply(
        lambda x: pd.Series(parse_admin_units(x))
    )
    impact_data = impact_data.drop(columns="Admin Units")

    geo_impact_data = process_impact_geometries(guil_shp, impact_data)
    del guil_shp # Free memory
    gc.collect()

    global_shp = gpd.read_file(gadm_path)
    geo_impact_data = gpd.GeoDataFrame(geo_impact_data, geometry="geometry", crs="EPSG:4326")
    geo_impact_data.to_crs(global_shp.crs, inplace=True)

    subnational = geo_impact_data[geo_impact_data.level != "ADM0"].copy()
    national = geo_impact_data[geo_impact_data.level == "ADM0"].copy()

    global_shp["shp_geometry"] = global_shp["geometry"]
    merged_gdf = gpd.sjoin(subnational, global_shp, how="left", predicate="within")

    unmatched = merged_gdf[merged_gdf["index_right"].isna()].drop(columns=["index_right"])
    if not unmatched.empty:
        global_shp["centroid"] = global_shp.centroid
        nearest = gpd.GeoDataFrame(global_shp, geometry="centroid").sjoin_nearest(
            unmatched[geo_impact_data.columns.tolist()], how="left", distance_col="distance"
        )
        nearest = nearest[nearest["GID_0"] == nearest["iso3"]]
        merged_gdf = pd.concat([merged_gdf.dropna(subset=["GID_0"]), nearest.drop(columns=["centroid", "distance"])])

    merged_gdf["geometry"] = merged_gdf["shp_geometry"]
    merged_gdf = pd.concat([merged_gdf.drop(["shp_geometry", "index_right"], axis=1), national])
    merged_gdf["GID_0"] = merged_gdf["GID_0"].fillna(merged_gdf.iso3)

    reduced_shp = global_shp[["GID_0", "GID_1", "GID_2"]].copy()
    del global_shp
    gc.collect()

    final_rows = []
    for event_id in merged_gdf["DisNo."].unique():
        df_event = merged_gdf[merged_gdf["DisNo."] == event_id].copy()
        country = df_event.GID_0.unique()[0]
        level = df_event.level.unique()[0]
        df_loc = reduced_shp[reduced_shp.GID_0 == country]

        if level == "ADM1":
            merged = pd.merge(df_event.drop("GID_2", axis=1, errors='ignore'), df_loc, on=["GID_0", "GID_1"], how="right")
        elif level == "ADM2":
            merged = pd.merge(df_event, df_loc, on=["GID_0", "GID_1", "GID_2"], how="right")
        else: 
            merged = pd.merge(df_event.drop(["GID_1", "GID_2"], axis=1, errors='ignore'), df_loc, on=["GID_0"], how="right")

        merged = merged.sort_values(by="Total Affected", ascending=False)
        cols_to_fill = ["DisNo.", "sid", "Start Year", "Start Month", "Event Name", "level"]
        for col in cols_to_fill:
            merged[col] = merged[col].ffill()
        
        merged["Total Affected"] = merged["Total Affected"].fillna(0)
        final_rows.append(merged.drop_duplicates().sort_values(["GID_1", "GID_2"]))

    pd.concat(final_rows, ignore_index=True).to_csv(out_path, index=False)
    del final_rows
    gc.collect()

def calculate_grid_impact():
    """
    Disaggregates EM-DAT impact data to the grid level based on regional population.
    
    The function calculates the 'perc_affected_pop_grid_region' metric by determining
    the total population within the administrative units specifically affected by 
    an event and distributing the impact accordingly.
    """
    in_path = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    out_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    pop_dir = OUTPUT_DIR / "Worldpop" / "grid_data"
    
    # Path to the administrative mapping (created during SHDI processing)
    grid_admin_dir = OUTPUT_DIR / "SHDI" / "grid_data"

    if not in_path.exists():
        return

    df_events = pd.read_csv(in_path)
    impact_data_grid = []

    for event_id in df_events["DisNo."].unique():
        # 1. Prepare event metadata
        df_event = df_events[df_events["DisNo."] == event_id].copy()
        iso = df_event["GID_0"].iloc[0]
        level = df_event["level"].iloc[0]  # ADM1 or ADM2

        # 2. Load on-demand grid and administrative mapping
        pop_file = pop_dir / f"population_grid_{iso}.csv"
        admin_map_file = grid_admin_dir / f"shdi_grid_{iso}.csv"
        
        if not (pop_file.exists() and admin_map_file.exists()):
            continue
            
        df_pop = pd.read_csv(pop_file)
        df_admin = pd.read_csv(admin_map_file)[["id", "GID_1", "GID_2"]]
        
        # Combine grid population with administrative IDs
        df_grid = df_pop.merge(df_admin, on="id", how="inner")
        
        # 3. Merge grid with event based on administrative sub-levels
        # This prevents duplicate assignments across subnational units
        merge_keys = ["GID_1"] if level == "ADM1" else ["GID_1", "GID_2"]
        df_merged = df_grid.merge(
            df_event.drop(columns=["iso3"], errors="ignore"), 
            on=merge_keys, 
            how="inner"
        )

        # 4. Calculate Regional Metrics
        # Identify grid cells within units where impact was reported
        affected_mask = (df_merged["Total Affected"] > 0) & (df_merged["population"] > 0)
        df_affected = df_merged[affected_mask].copy()

        if df_affected.empty:
            continue

        # total_pop_reg: sum of population in all affected subnational units
        total_pop_reg = df_affected.drop_duplicates(subset=["id"])["population"].sum()

        # perc_affected_pop_grid_region: (100 * Total Affected) / Total Regional Population
        # This represents the impact intensity relative to the exposed regional pool
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
        
        # Clear local memory for next iteration
        del df_pop, df_admin, df_grid, df_merged, df_affected

    if impact_data_grid:
        pd.concat(impact_data_grid, ignore_index=True).to_csv(out_path, index=False)

def process_emdat_events():
    preprocess_emdat_events()
    calculate_grid_impact()

if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_emdat_events()