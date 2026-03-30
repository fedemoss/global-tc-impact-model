import pandas as pd
import geopandas as gpd
import numpy as np
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

def process_shdi_to_grid(iso, grid_global, shdi_df, out_dir):
    """Maps subnational SHDI values to grid cells based on ISO and region names."""
    out_path = out_dir / f"shdi_grid_{iso}.csv"
    if out_path.exists(): return

    # Filter for the specific country and the latest available year (e.g., 2019)
    shdi_country = shdi_df[(shdi_df.iso_code == iso) & (shdi_df.year == 2019)].copy()
    grid_country = grid_global[grid_global.iso3 == iso].copy()

    if shdi_country.empty:
        grid_country["shdi"] = np.nan
    else:
        # SHDI uses its own region names; we merge on GADM names provided in the grid
        # Note: Some fuzzy matching may be required if GADM and SHDI names differ slightly
        grid_country = grid_country.merge(
            shdi_country[['region', 'shdi']], 
            left_on='NAME_1', right_on='region', how='left'
        )

    grid_country[["id", "iso3", "shdi"]].to_csv(out_path, index=False)

if __name__ == "__main__":
    shdi_file = INPUT_DIR / "SHDI" / "SHDI_Complete_v10.csv"
    if shdi_file.exists():
        shdi_df = pd.read_csv(shdi_file)
        grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
        grid["iso3"] = grid["GID_0"]
        
        out_dir = OUTPUT_DIR / "SHDI" / "grid_data"
        out_dir.mkdir(parents=True, exist_ok=True)

        for iso in ISO3_LIST:
            process_shdi_to_grid(iso, grid, shdi_df, out_dir)