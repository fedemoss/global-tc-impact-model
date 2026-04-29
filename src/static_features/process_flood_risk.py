import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

def select_tiles_by_country(tif_paths, grid_country, buffer=3):
    """Filters global flood tiles to only those overlapping the country boundary."""
    selected_files = []
    country_geom = grid_country.geometry.union_all().buffer(buffer)
    pattern = r"ID\d+_(N|S)(\d+)_([EW])(\d+)_RP10_depth_reclass\.tif"

    for path in tif_paths:
        match = re.search(pattern, os.path.basename(path))
        if match:
            lat_s, lat_v, lon_s, lon_v = match.groups()
            lat = int(lat_v) if lat_s == "N" else -int(lat_v)
            lon = int(lon_v) if lon_s == "E" else -int(lon_v)
            # Check if tile center is within buffered country geometry
            if country_geom.contains(box(lon-0.5, lat-0.5, lon+0.5, lat+0.5).centroid):
                selected_files.append(path)
    return selected_files

def process_single_country_flood(iso, grid_global, tile_paths, out_dir):
    """Calculates max flood risk per grid cell across all relevant tiles."""
    out_path = out_dir / f"flood_risk_{iso}.csv"
    if out_path.exists(): return

    grid_country = grid_global[grid_global.iso3 == iso].copy()
    relevant_tiles = select_tiles_by_country(tile_paths, grid_country)
    
    if not relevant_tiles:
        grid_country["flood_risk"] = np.nan
    else:
        # Collect max values per grid cell from each overlapping tile
        tile_results = []
        for tif in relevant_tiles:
            with rasterio.open(tif) as src:
                stats = zonal_stats(grid_country, tif, stats=["max"], nodata=src.nodata, all_touched=True)
                tile_results.append([s["max"] for s in stats])
        
        grid_country["flood_risk"] = np.nanmax(np.array(tile_results), axis=0)

    grid_country[["id", "iso3", "flood_risk"]].to_csv(out_path, index=False)

def process_all_flood():
    tile_dir = INPUT_DIR / "FloodRisk" / "tiles"
    tile_paths = [tile_dir / f for f in os.listdir(tile_dir) if f.endswith("_depth_reclass.tif")]

    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["iso3"] = grid["GID_0"]

    out_dir = OUTPUT_DIR / "FloodRisk" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_iso3 = [iso for iso in ISO3_LIST if iso in grid.iso3.unique()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        for iso in valid_iso3:
            executor.submit(process_single_country_flood, iso, grid, tile_paths, out_dir)

if __name__ == "__main__":
    process_all_flood()

