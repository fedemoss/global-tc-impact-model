import pandas as pd
import geopandas as gpd
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

def group_shp(gdf_subset):
    """Aggregates geometries based on GID hierarchy using unary_union."""
    # Group by GID levels
    grouped = gdf_subset.groupby(["GID_0", "GID_1", "GID_2"])
    
    # Aggregate geometries
    agg_geometries = grouped["geometry"].agg(lambda x: x.unary_union)
    
    # Reconstruct GeoDataFrame
    agg_df = gpd.GeoDataFrame(agg_geometries, geometry="geometry").reset_index()
    return agg_df

def process_gadm_adm2():
    """
    Creates a unified global ADM2-level shapefile for the study countries.
    Falls back to ADM1 for countries where ADM2 is unavailable.
    """
    print("Processing GADM to unified sub-national levels...")
    raw_path = INPUT_DIR / "SHP" / "gadm_410.gdb"
    out_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    
    if not raw_path.exists():
        print("Error: Raw GADM file not found.")
        return

    # Load the global dataset (ADM0 layer usually contains all sub-units in GeoPackages)
    # We load the full hierarchy for filtering
    global_shp = gpd.read_file(raw_path)
    
    # Filter for countries in the study list
    global_shp = global_shp[global_shp["GID_0"].isin(ISO3_LIST)]
    
    # 1. Process ADM2 Data
    # Keep rows where GID_2 is populated
    adm2_data = global_shp[global_shp.GID_2.notna() & (global_shp.GID_2 != "")].copy()
    agg_adm2 = group_shp(adm2_data[["GID_0", "GID_1", "GID_2", "geometry"]])
    
    # 2. Process ADM1 Fallback
    # Keep rows where GID_2 is missing (countries with only ADM1 data)
    adm1_data = global_shp[global_shp.GID_2.isna() | (global_shp.GID_2 == "")].copy()
    agg_adm1 = adm1_data[["GID_0", "GID_1", "GID_2", "geometry"]].reset_index(drop=True)
    
    # 3. Combine and Save
    final_gdf = pd.concat([agg_adm1, agg_adm2], ignore_index=True)
    
    # Ensure CRS is consistent
    final_gdf = final_gdf.set_crs("EPSG:4326", allow_override=True)
    
    final_gdf.to_file(out_path, driver="GPKG")
    print(f"Unified GID ADM2 shapefile saved to {out_path}")

if __name__ == "__main__":
    process_gadm_adm2()