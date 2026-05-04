#!/usr/bin/env python3
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors

from src.config import INPUT_DIR, OUTPUT_DIR

# -------------------------------------------------------------------
# Configuration & Logging
# -------------------------------------------------------------------
LOG_DIR = OUTPUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "process_srtm.log"),
        logging.StreamHandler()
    ]
)

# -------------------------------------------------------------------
# Geometry Utilities
# -------------------------------------------------------------------

def adjust_longitude(polygon):
    """Adjusts polygon longitude to [-360, 0) as required by SRTM data."""
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)


def is_border_crossing(polygon):
    """Identifies if a polygon straddles the meridian."""
    coords = list(polygon.exterior.coords)
    has_pos = any(lon > 0 for lon, lat in coords)
    has_neg = any(lon < 0 for lon, lat in coords)
    return has_pos and has_neg

# -------------------------------------------------------------------
# Raster Tile Management
# -------------------------------------------------------------------

def get_overlap_files(extent):
    """Identifies SRTM .tif tiles overlapping with a bounding box."""
    tif_files = [f"srtm_{xx:02d}_{yy:02d}.tif" for xx in range(1, 73) for yy in range(1, 25)]
    minx, miny, maxx, maxy = extent
    lon_step, lat_step = 5.0, 5.0  # SRTM tiles are 5x5 degrees

    overlapping_files = []
    for file in tif_files:
        parts = file.replace("srtm_", "").replace(".tif", "").split("_")
        xx, yy = int(parts[0]) - 1, int(parts[1]) - 1
        cell_minx = -180 + xx * lon_step
        cell_maxx = cell_minx + lon_step
        cell_maxy = 60 - yy * lat_step
        cell_miny = cell_maxy - lat_step

        if not (cell_maxx <= minx or cell_minx >= maxx or cell_maxy <= miny or cell_miny >= maxy):
            overlapping_files.append(file)
    return overlapping_files

# -------------------------------------------------------------------
# Terrain Analysis
# -------------------------------------------------------------------

def run_terrain_task(grid_subset, raster_path):
    """
    Calculates altitude, slope, and ruggedness for a single tile.
    Executes sequentially within a process to avoid CPU over-subscription.
    """
    # Remove border crossing polygons for this specific raster calculation
    grid_clean = grid_subset[~grid_subset["geometry"].apply(is_border_crossing)].copy()
    if grid_clean.empty:
        return None

    def run_gdaldem(mode, r_path):
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        subprocess.run(["gdaldem", mode, "-co", "COMPRESS=DEFLATE", str(r_path), tmp, "-compute_edges"], 
                       check=True, capture_output=True)
        return tmp

    slope_path = run_gdaldem("slope", raster_path)
    tri_path = run_gdaldem("TRI", raster_path)

    try:
        with rasterio.open(raster_path) as src:
            alt = pd.DataFrame(zonal_stats(grid_clean, src.read(1), stats="mean", nodata=-32768, affine=src.transform))
        with rasterio.open(slope_path) as src:
            slope = pd.DataFrame(zonal_stats(grid_clean, src.read(1), stats="mean", nodata=-9999, affine=src.transform))
        with rasterio.open(tri_path) as src:
            rug = pd.DataFrame(zonal_stats(grid_clean, src.read(1), stats="mean", nodata=-9999, affine=src.transform))
        
        # Consolidate results
        res = pd.DataFrame({
            "id": grid_clean["id"].values,
            "mean_elev": alt["mean"].values,
            "mean_slope": slope["mean"].values,
            "mean_rug": rug["mean"].values
        })
        return res
    finally:
        for p in [slope_path, tri_path]:
            if os.path.exists(p): os.remove(p)

# -------------------------------------------------------------------
# Spatial Processing
# -------------------------------------------------------------------

def spatial_interpolation(df, columns):
    """Inverse Distance Weighting interpolation for missing grid values."""
    df = df.copy()
    if not df[columns].isnull().any().any():
        return df

    df["centroid"] = df.geometry.centroid
    coords = np.array([[p.x, p.y] for p in df.centroid])

    for col in columns:
        missing = df[col].isna()
        known = ~missing
        if not known.any() or not missing.any(): continue
        
        nbrs = NearestNeighbors(n_neighbors=min(8, known.sum()), algorithm="auto").fit(coords[known])
        dist, idx = nbrs.kneighbors(coords[missing])
        weights = 1 / (dist + 1e-10)
        weights /= weights.sum(axis=1, keepdims=True)
        df.loc[missing, col] = (weights * df.loc[known, col].values[idx]).sum(axis=1)
    
    return df.drop(columns=["centroid"])

def get_coast_features(shp_country, grid_country):
    """Calculates coastline binary flag and length in meters."""
    dissolved = shp_country.dissolve(by="GID_0")
    coastline = dissolved.boundary
    
    # Intersection overlay
    coast_gdf = gpd.GeoDataFrame(geometry=coastline, crs=grid_country.crs)
    grid_line = gpd.overlay(coast_gdf, grid_country, how="intersection")
    
    # Calculate length in meters (EPSG:25394 is a placeholder for metric projections)
    grid_line["coast_length_meters"] = grid_line.to_crs(epsg=25394).length
    
    merged = grid_country[["id"]].merge(grid_line[["id", "coast_length_meters"]], on="id", how="left").fillna(0)
    merged["with_coast"] = (merged["coast_length_meters"] > 0).astype(int)
    return merged

# -------------------------------------------------------------------
# Execution Logic
# -------------------------------------------------------------------

def process_country(iso, global_grid, global_shp, out_path, data_path):
    """Processes a single country's terrain and coastal data."""
    output_file = out_path / f"srtm_grid_data_{iso}.csv"
    if output_file.exists():
        return logging.info(f"Skipping {iso}: Output exists.")

    logging.info(f"Processing {iso}...")
    
    # Subset data to country level once to save memory in parallel workers
    grid_c = global_grid[global_grid.iso3 == iso].copy()
    shp_c = global_shp[global_shp.GID_0 == iso].copy()
    
    if grid_c.empty:
        return logging.warning(f"No grid data for {iso}.")

    # Identify and validate tiles
    extent = grid_c.total_bounds
    tiles = get_overlap_files(extent)
    tile_paths = [data_path / t for t in tiles if (data_path / t).exists()]

    if not tile_paths:
        logging.warning(f"No tiles found for {iso}.")
        df_terrain = grid_c[["id", "geometry"]].copy()
        for col in ["mean_elev", "mean_slope", "mean_rug"]: df_terrain[col] = np.nan
    else:
        # Process tiles in parallel. Limit Pool size to prevent CPU overload.
        workers = min(len(tile_paths), cpu_count() // 2)
        with Pool(processes=workers) as pool:
            tile_results = pool.starmap(run_terrain_task, [(grid_c, p) for p in tile_paths])
        
        # Merge tile results (averaging overlaps)
        df_terrain_raw = pd.concat([r for r in tile_results if r is not None])
        df_terrain_avg = df_terrain_raw.groupby("id").mean().reset_index()
        df_terrain = grid_c[["id", "geometry"]].merge(df_terrain_avg, on="id", how="left")
        
        # Interpolate missing values (e.g. border cells or small gaps)
        df_terrain = spatial_interpolation(df_terrain, ["mean_elev", "mean_slope", "mean_rug"])

    # Coastal processing
    df_coast = get_coast_features(shp_c, grid_c)
    
    # Final assembly
    final = df_terrain.merge(df_coast, on="id", how="left")
    final["iso3"] = iso
    final.drop(columns="geometry", errors="ignore").to_csv(output_file, index=False)
    logging.info(f"Completed {iso}.")

def process_all_srtm():
    """Entry point for global SRTM processing."""
    out_path = OUTPUT_DIR / "SRTM" / "grid_data"
    data_path = INPUT_DIR / "SRTM" / "tiles"
    out_path.mkdir(parents=True, exist_ok=True)

    logging.info("Loading global datasets...")
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    grid["GID_0"] = grid["iso3"]

    shp = gpd.read_file(INPUT_DIR / "SHP" / "gadm_410.gdb")
    
    # Iterate countries sequentially to prevent process explosion
    # Multi-processing is handled at the tile level within each country
    ISO3_LIST = grid.iso3.unique()
    for iso in ISO3_LIST:
        if iso in grid.iso3.unique():
            try:
                process_country(iso, grid, shp, out_path, data_path)
            except Exception as e:
                logging.error(f"Failed to process {iso}: {e}")

    # Compile global results
    logging.info("Compiling global SRTM dataset...")
    all_csvs = list(out_path.glob("srtm_grid_data_*.csv"))
    global_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    global_df.to_csv(out_path / "global_srtm_grid_data.csv", index=False)
    logging.info("Global SRTM processing complete.")

if __name__ == "__main__":
    process_all_srtm()