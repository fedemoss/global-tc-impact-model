#!/usr/bin/env python3
import logging
import os
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from sklearn.neighbors import NearestNeighbors

from src.config import INPUT_DIR, OUTPUT_DIR
from src.utils.geo_utils import (
    GLOBAL_METRIC_EPSG,
    adjust_longitude,
    is_antimeridian_crossing,
)

logger = logging.getLogger(__name__)

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
    # Remove antimeridian-crossing polygons for this specific raster calculation
    grid_clean = grid_subset[~grid_subset["geometry"].apply(is_antimeridian_crossing)].copy()
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

    coast_gdf = gpd.GeoDataFrame(geometry=coastline, crs=grid_country.crs)
    grid_line = gpd.overlay(coast_gdf, grid_country, how="intersection")

    # World Cylindrical Equal Area — preserves length-scale globally and avoids
    # the local-only EPSG:25394 (Philippines) used previously.
    grid_line["coast_length_meters"] = grid_line.to_crs(epsg=GLOBAL_METRIC_EPSG).length

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
        logger.info(f"Skipping {iso}: Output exists.")
        return

    logger.info(f"Processing {iso}...")

    grid_c = global_grid[global_grid.iso3 == iso].copy()
    shp_c = global_shp[global_shp.GID_0 == iso].copy()

    if grid_c.empty:
        logger.warning(f"No grid data for {iso}.")
        return

    extent = grid_c.total_bounds
    tiles = get_overlap_files(extent)
    tile_paths = [data_path / t for t in tiles if (data_path / t).exists()]

    if not tile_paths:
        logger.warning(f"No tiles found for {iso}.")
        df_terrain = grid_c[["id", "geometry"]].copy()
        for col in ["mean_elev", "mean_slope", "mean_rug"]:
            df_terrain[col] = np.nan
    else:
        workers = max(1, min(len(tile_paths), cpu_count() // 2))
        with Pool(processes=workers) as pool:
            tile_results = pool.starmap(run_terrain_task, [(grid_c, p) for p in tile_paths])

        valid_results = [r for r in tile_results if r is not None]
        if valid_results:
            df_terrain_raw = pd.concat(valid_results)
            df_terrain_avg = df_terrain_raw.groupby("id").mean().reset_index()
            df_terrain = grid_c[["id", "geometry"]].merge(df_terrain_avg, on="id", how="left")
            df_terrain = spatial_interpolation(df_terrain, ["mean_elev", "mean_slope", "mean_rug"])
        else:
            df_terrain = grid_c[["id", "geometry"]].copy()
            for col in ["mean_elev", "mean_slope", "mean_rug"]:
                df_terrain[col] = np.nan

    df_coast = get_coast_features(shp_c, grid_c)

    final = df_terrain.merge(df_coast, on="id", how="left")
    final["iso3"] = iso
    final.drop(columns="geometry", errors="ignore").to_csv(output_file, index=False)
    logger.info(f"Completed {iso}.")


def process_all_srtm():
    """Entry point for global SRTM processing."""
    out_path = OUTPUT_DIR / "SRTM" / "grid_data"
    data_path = INPUT_DIR / "SRTM" / "tiles"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading global datasets...")
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")

    # Only wrap polygons that actually carry longitudes > 180 — leave the rest
    # untouched so they continue to align with the [-180, 180] SRTM tiles.
    needs_wrap = grid.geometry.apply(
        lambda g: any(lon > 180 for lon, _ in g.exterior.coords) if g is not None else False
    )
    if needs_wrap.any():
        logger.info(f"Wrapping longitudes for {int(needs_wrap.sum())} antimeridian-crossing polygons.")
        grid.loc[needs_wrap, "geometry"] = grid.loc[needs_wrap, "geometry"].apply(adjust_longitude)
    grid["GID_0"] = grid["iso3"]

    shp = gpd.read_file(INPUT_DIR / "SHP" / "gadm_410.gdb")

    for iso in grid.iso3.unique():
        try:
            process_country(iso, grid, shp, out_path, data_path)
        except Exception as e:
            logger.error(f"Failed to process {iso}: {e}", exc_info=True)

    logger.info("Compiling global SRTM dataset...")
    all_csvs = list(out_path.glob("srtm_grid_data_*.csv"))
    global_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    global_df.to_csv(out_path / "global_srtm_grid_data.csv", index=False)
    logger.info("Global SRTM processing complete.")


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_srtm()