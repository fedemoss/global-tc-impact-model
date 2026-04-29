import os
import ast
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

# Configure logging for spatial processing
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -------------------------------------------------------------------
# Impact Disaggregation to the adm1 level
# -------------------------------------------------------------------

def clean_emdat(df):
    """Filters and cleans the raw EM-DAT dataset for tropical cyclones."""
    logging.info("Cleaning EM-DAT source data...")
    columns = [
        "DisNo.", "Total Affected", "Start Year", "Start Month",
        "iso3", "Country", "Admin Units", "sid", "Event Name"
    ]
    # Drop rows without 'Total Affected' and sort for consistency
    df_clean = (
        df[columns]
        .dropna(subset=["Total Affected"])
        .sort_values(["iso3", "sid", "Start Year"])
        .reset_index(drop=True)
    )
    return df_clean

def parse_admin_units(x):
    """Parses the 'Admin Units' string to extract ADM levels and PCodes."""
    try:
        list_locations = ast.literal_eval(x)
        adm1_pcodes = []
        adm2_pcodes = []
        adm1_found = False
        adm2_found = False

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
        else:
            return {"level": "ADM0", "regions_affected": []}
    except Exception:
        return {"level": "ADM0", "regions_affected": []}

def process_impact_geometries(guil_shp, impact_data):
    """Maps impact records to geometries based on ADM level and calculates centroids."""
    logging.info("Processing ADM level geometries and centroids...")
    results = []

    # Process ADM1 and ADM2
    for level in ["ADM1", "ADM2"]:
        subset = impact_data[impact_data.level == level].copy()
        if subset.empty:
            continue
            
        exploded = subset.explode("regions_affected").reset_index(drop=True)
        code_col = f"{level}_CODE"
        
        merged = guil_shp.merge(
            exploded, left_on=code_col, right_on="regions_affected", how="right"
        )
        
        # Keep relevant columns and calculate centroid
        cols_to_keep = exploded.columns.tolist() + ["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"]
        merged = merged[cols_to_keep].drop("regions_affected", axis=1)
        merged = gpd.GeoDataFrame(merged, geometry="geometry")
        merged["centroid"] = merged.geometry.centroid
        results.append(merged)

    # Process ADM0 (National level)
    adm0_subset = impact_data[impact_data.level == "ADM0"].copy()
    if not adm0_subset.empty:
        results.append(adm0_subset)

    # Concatenate and clean
    combined = pd.concat(results, ignore_index=True)
    combined = combined.drop(["ADM1_CODE", "ADM2_CODE", "ADM0_NAME", "geometry"], axis=1, errors='ignore')
    
    if "centroid" in combined.columns:
        combined = combined.rename(columns={"centroid": "geometry"})
    
    return combined

def add_missing_sid(df):
    """Manually supplements missing IBTrACS SIDs for the 2022 season."""
    logging.info("Applying manual SID matching for 2022 events...")
    
    # Mapping based on manual IBTrACS lookup (Sitrang, Nanmadol, Batsirai, etc.)
    manual_sid_map = {
        "2022-0690-BGD": "2022295N13093", "2022-0590-JPN": "2022254N24143",
        "2022-0046-MDG": "2022020S13059", "2022-0071-MDG": "2022025S11091",
        "2022-0104-MDG": "2022042S12063", "2022-0113-MDG": "2022047S15073",
        "2022-0046-MOZ": "2022020S13059", "2022-0104-MOZ": "2022042S12063",
        "2022-0217-PHL": "2022099N11128", "2022-0518-PHL": "2022232N18131",
        "2022-0660-PHL": "2022285N17140", "2022-0658-PHL": "2022285N12116",
        "2022-0785-THA": "2022338N05100", "2022-0149-MWI": "2022065S16055",
        "2022-0261-MWI": "2022110S12051", "2022-0046-MWI": "2022020S13059",
        "2022-0013-FJI": "2022008S17173", "2022-0402-JPN": "2022180N15130",
        "2022-0604-JPN": "2022263N18137", "2022-0545-KOR": "2022239N22150",
        "2022-0149-MOZ": "2022065S16055", "2022-0071-MUS": "2022025S11091",
        "2022-0715-PHL": "2022299N11134", "2022-0660-VNM": "2022285N17140",
        "2022-0046-ZWE": "2022020S13059"
    }

    df["sid"] = df["sid"].fillna(df["DisNo."].map(manual_sid_map))
    return df.dropna(subset=["sid"]).reset_index(drop=True)

def preprocess_emdat_events():
    """Main function to clean, geolocate, and spatially expand the EM-DAT database."""
    # IO Paths from config
    emdat_path = INPUT_DIR / "EMDAT" / "emdat.csv"
    gadm_path = INPUT_DIR / "SHP" / "GADM_adm2.gpkg"
    guil_path = INPUT_DIR / "SHP" / "global_shapefile_GUIL_adm2.gpkg"
    out_path = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    os.makedirs(out_path.parent, exist_ok=True)

    # 1. Initialization
    emdat_raw = pd.read_csv(emdat_path)
    impact_data = clean_emdat(emdat_raw)
    
    global_shp = gpd.read_file(gadm_path)
    guil_shp = gpd.read_file(guil_path)[["ADM2_CODE", "ADM1_CODE", "ADM0_NAME", "geometry"]]

    # 2. Extract Admin Units
    impact_data[["level", "regions_affected"]] = impact_data["Admin Units"].apply(
        lambda x: pd.Series(parse_admin_units(x))
    )
    impact_data = impact_data.drop(columns="Admin Units")

    # 3. Geometric Processing
    geo_impact_data = process_impact_geometries(guil_shp, impact_data)
    geo_impact_data = gpd.GeoDataFrame(geo_impact_data, geometry="geometry", crs="EPSG:4326")
    geo_impact_data.to_crs(global_shp.crs, inplace=True)

    # 4. Spatial Join with GADM
    subnational = geo_impact_data[geo_impact_data.level != "ADM0"].copy()
    national = geo_impact_data[geo_impact_data.level == "ADM0"].copy()

    global_shp["shp_geometry"] = global_shp["geometry"]
    merged_gdf = gpd.sjoin(subnational, global_shp, how="left", predicate="within")

    # Nearest neighbor for points outside GADM polygons
    unmatched = merged_gdf[merged_gdf["index_right"].isna()].drop(columns=["index_right"])
    if not unmatched.empty:
        global_shp["centroid"] = global_shp.centroid
        nearest = gpd.GeoDataFrame(global_shp, geometry="centroid").sjoin_nearest(
            unmatched[geo_impact_data.columns.tolist()], how="left", distance_col="distance"
        )
        # Keep nearest and ensure country match
        nearest = nearest.sort_values("distance").drop_duplicates(subset="index_right", keep="first")
        nearest = nearest[nearest["GID_0"] == nearest["iso3"]]
        nearest = nearest.drop(columns=["centroid", "distance"])
        
        merged_gdf = merged_gdf.dropna(subset=["GID_0", "GID_1"])
        merged_gdf = pd.concat([merged_gdf, nearest], ignore_index=True)

    merged_gdf["geometry"] = merged_gdf["shp_geometry"]
    merged_gdf = pd.concat([merged_gdf.drop(["shp_geometry", "index_right"], axis=1), national])
    merged_gdf["GID_0"] = merged_gdf["GID_0"].fillna(merged_gdf.iso3)

    # 5. Spatial Expansion (Zero-Filling non-impacted Admin Units)
    logging.info("Expanding events to all country administrative units...")
    reduced_shp = global_shp[["GID_0", "GID_1", "GID_2"]]
    final_rows = []

    for event_id in merged_gdf["DisNo."].unique():
        df_event = merged_gdf[merged_gdf["DisNo."] == event_id].copy()
        country = df_event.GID_0.unique()[0]
        level = df_event.level.unique()[0]
        df_loc = reduced_shp[reduced_shp.GID_0 == country]

        if level == "ADM1":
            merged = pd.merge(df_event.drop("GID_2", axis=1), df_loc, on=["GID_0", "GID_1"], how="right")
        elif level == "ADM2":
            merged = pd.merge(df_event, df_loc, on=["GID_0", "GID_1", "GID_2"], how="right")
        else: # ADM0
            merged = pd.merge(df_event.drop(["GID_1", "GID_2"], axis=1), df_loc, on=["GID_0"], how="right")

        # Forward fill event metadata and fill targets with zero
        merged = merged.sort_values(by="Total Affected", ascending=False)
        cols_to_fill = ["DisNo.", "sid", "Start Year", "Start Month", "Event Name", "level"]
        for col in cols_to_fill:
            # Sort to put impacted rows at top before filling
            merged = merged.sort_values(by="Total Affected", ascending=False)
            merged[col] = merged[col].ffill()
        
        merged["Total Affected"] = merged["Total Affected"].fillna(0)
        final_rows.append(merged.drop_duplicates().sort_values(["GID_1", "GID_2"]))

    df_final = pd.concat(final_rows, ignore_index=True)
    df_final = add_missing_sid(df_final)

    df_final.to_csv(out_path, index=False)
    logging.info(f"Master Impact Registry saved to {out_path}")

# -------------------------------------------------------------------
# Impact Disaggregation to the grid level
# -------------------------------------------------------------------
def calculate_grid_impact():
    """
    Disaggregates national or regional TC impact data to the grid level 
    based on exposed population.
    """
    # Load processed EMDAT db
    in_path = OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv"
    out_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    df_events = pd.read_csv(in_path)

    impact_data_grid = pd.DataFrame()
    for typhoon_id in df_events["DisNo."].unique():
        df_event = df_events[df_events["DisNo."] == typhoon_id]
        df_event_dmg_with_pop = df_event[
            (df_event["population"] > 1) & (df_event["Total Affected"] != 0)
        ].copy()
        
        if df_event_dmg_with_pop.empty:
            continue

        total_pop_reg = df_event_dmg_with_pop["population"].sum()

        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid_region"] = (
            100 * df_event_dmg_with_pop["Total Affected"] / total_pop_reg
        )
        
        df_event_dmg_with_pop.loc[:, "perc_affected_pop_grid"] = (
            100 * df_event_dmg_with_pop["population"] * df_event_dmg_with_pop["Total Affected"] / (total_pop_reg ** 2)
        )
        
        df_event = df_event.merge(df_event_dmg_with_pop, how="left").fillna(0)
        impact_data_grid = pd.concat([impact_data_grid, df_event])

    impact_data_grid.to_csv(out_path, index=False)
    logging.info(f"Master Gridded Impact Registry saved to {out_path}")

def process_emdat_events():
    """ Main pipeline """
    preprocess_emdat_events()
    calculate_grid_impact()
    

if __name__ == "__main__":
    process_emdat_events()