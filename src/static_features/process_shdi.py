import pandas as pd
import geopandas as gpd
import numpy as np
from src.config import INPUT_DIR, OUTPUT_DIR

def process_shdi_to_grid(iso, grid_enriched, shdi_df, out_dir):
    """
    Maps SHDI values to grid cells using enriched administrative attributes.
    """
    out_path = out_dir / f"shdi_grid_{iso}.csv"
    target_year = '2019'
    
    shdi_country = shdi_df[shdi_df["ISO_Code"] == iso].copy()
    
    if shdi_country.empty:
        grid_enriched["shdi"] = np.nan
    else:
        # Determine national fallback value
        nat_row = shdi_country[shdi_country["Level"] == "National"]
        nat_val = nat_row[target_year].values[0] if not nat_row.empty else np.nan

        # Prepare subnational mapping
        # SHDI uses 'Region' names which correspond to GADM 'NAME_1'
        subnat_map = shdi_country[shdi_country["Level"] == "Subnat"][[ 'Region', target_year ]]
        subnat_map = subnat_map.rename(columns={target_year: 'shdi_val'})

        # Merge based on Region name (Requires NAME_1 to be in the grid)
        if "NAME_1" in grid_enriched.columns:
            grid_enriched = grid_enriched.merge(
                subnat_map, 
                left_on="NAME_1", 
                right_on="Region", 
                how="left"
            )
            grid_enriched["shdi"] = grid_enriched["shdi_val"].fillna(nat_val)
        else:
            grid_enriched["shdi"] = nat_val

    # Final cleanup and export
    final = grid_enriched[["id", "iso3", "shdi"]].copy()
    final["GID_1"] = grid_enriched.get("GID_1", None)
    final["GID_2"] = grid_enriched.get("GID_2", None)
    
    final.to_csv(out_path, index=False)


def process_all_shdi():
    """
    Orchestrates the mapping of SHDI data to the global grid by first 
    enriching the grid with administrative boundaries.
    """
    shdi_file = INPUT_DIR / "SHDI" / "GDL-Subnational-HDI-data.csv"
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    adm2_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    
    if not shdi_file.exists():
        return

    # Load core datasets
    shdi_df = pd.read_csv(shdi_file)
    grid_global = gpd.read_file(grid_path)
    
    # Load administrative boundaries (ADM2)
    # Ensure this file contains NAME_1 or similar for SHDI name matching
    adm2_shp = gpd.read_file(adm2_path)

    out_dir = OUTPUT_DIR / "SHDI" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    iso_list = grid_global.iso3.unique()
    
    for iso in iso_list:
        # 1. Filter global datasets to the specific country
        grid_country = grid_global[grid_global.iso3 == iso].copy()
        adm_country = adm2_shp[adm2_shp.GID_0 == iso].copy()
        
        if adm_country.empty:
            # Fallback if no administrative shapes exist for this ISO
            process_shdi_to_grid(iso, grid_country, shdi_df, out_dir)
            continue

        # 2. Spatial Join: Attach ADM info to grid cells
        # We use 'predicate=intersects' or 'within' 
        # Centroid join is often cleaner for grid cells to avoid double-counting
        grid_country["centroid"] = grid_country.geometry.centroid
        enriched_grid = gpd.sjoin(
            grid_country.set_geometry("centroid"), 
            adm_country, 
            how="left", 
            predicate="within"
        )
        
        # Restore the original polygon geometry
        enriched_grid = enriched_grid.set_geometry("geometry").drop(columns=["centroid", "index_right"])

        # 3. Process SHDI using the now-enriched grid (which contains GID_1, NAME_1, etc.)
        process_shdi_to_grid(iso, enriched_grid, shdi_df, out_dir)

if __name__ == "__main__":
    process_all_shdi()
