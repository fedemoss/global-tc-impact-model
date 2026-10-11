#!/usr/bin/env python3
import logging
import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from sklearn.neighbors import NearestNeighbors

from src.config import EXCLUDED_ISO3, INPUT_DIR, OUTPUT_DIR
from src.utils.geo_utils import adjust_longitude, is_antimeridian_crossing

logger = logging.getLogger(__name__)


def require_gdaldem():
    """Fail loudly, once, if the gdaldem binary is not on PATH.

    Terrain features shell out to `gdaldem` (a PATH lookup), while the rest of
    the module only needs importable Python packages. Running the pipeline with
    the env's interpreter but without its bin/ on PATH therefore imports fine
    and then fails per country, which previously logged the same
    FileNotFoundError 260 times while the stage still reported success. Checking
    up front turns that into one actionable error before any work is done.
    """
    if shutil.which("gdaldem") is None:
        raise RuntimeError(
            "gdaldem not found on PATH. It ships with GDAL; if GDAL came from "
            "conda, activate the environment (or add its bin/ to PATH) so the "
            "binary resolves, not just the Python packages. See the README's "
            "Prerequisites section."
        )


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


# Select country SRTM tiles for countries with grid cells
def get_country_tiles(grid):
    country_tiles  = []
    for iso in grid.iso3.unique():
        # Particular cases
        if iso == 'USA':
            usa_extent_1 = [172, 18.9, 180, 72.7]
            usa_extent_2 = [-180, 18.9, -67, 72.7]
            tiles = get_overlap_files(usa_extent_1) + get_overlap_files(usa_extent_2)
        elif iso == 'FJI':
            fji_extent_1 = [176.8, -21.1, 180, -12.4] 
            fji_extent_2 = [-180, -21.1, -178, -12.4] 
            tiles = get_overlap_files(fji_extent_1) + get_overlap_files(fji_extent_2)
        elif iso == 'NZL':
            nzl_extent_1 = [165.8, -52.7, 180, -29.2]
            nzl_extent_2 = [-180, -52.7, -176, -29.2]
            tiles = get_overlap_files(nzl_extent_1) + get_overlap_files(nzl_extent_2)
        elif iso == 'KIR':
            kir_extent_1 = [176,  -17.852114,  180,    4.699492]
            kir_extent_2 = [-180,  -17.852114,  -174,    4.699492]
            tiles = get_overlap_files(kir_extent_1) + get_overlap_files(kir_extent_2)
        elif iso == 'RUS':
            rus_extent_1 = [25, 41.18886566,  180,   81.856247]
            rus_extent_2 = [-180, 41.18886566,  -170,   81.856247]
            tiles = get_overlap_files(rus_extent_1) + get_overlap_files(rus_extent_2)
        # Regular cases
        else:
            country_grid = grid[grid.iso3==iso]
            extent = country_grid.total_bounds
            tiles =  get_overlap_files(extent)
        
        country_tiles.append({'iso3': iso, 'tiles': tiles })
    return pd.DataFrame(country_tiles)

# -------------------------------------------------------------------
# Terrain Analysis
# -------------------------------------------------------------------

def derived_tile_path(raster_path, mode):
    """Cache location for a gdaldem product of one SRTM tile."""
    out_dir = INPUT_DIR / "SRTM" / "derived"
    return out_dir / f"{Path(raster_path).stem}_{mode.lower()}.tif"


def ensure_derived_tile(raster_path, mode):
    """Return the cached gdaldem product for a tile, building it if absent.

    slope and TRI depend only on the tile, never on which country is being
    processed, but tiles are shared between neighbouring countries. Computing
    them inside the per-country loop therefore re-decompressed the DEM and
    re-ran gdaldem once per (country, tile) pair. Caching makes it once per
    tile, which is the number of times the work actually needs doing.

    The write is atomic (temp file in the same directory, then os.replace) so
    that workers racing on a shared border tile cannot observe a partial file.
    """
    out_path = derived_tile_path(raster_path, mode)
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tif", dir=str(out_path.parent))
    os.close(fd)
    try:
        subprocess.run(
            ["gdaldem", mode, "-co", "COMPRESS=DEFLATE", str(raster_path), tmp,
             "-compute_edges"],
            check=True, capture_output=True,
        )
        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return out_path


def run_terrain_task(grid_subset, raster_path):
    """
    Calculates altitude, slope, and ruggedness for a single tile.
    Executes sequentially within a process to avoid CPU over-subscription.

    `grid_subset` is expected to be free of antimeridian-crossing polygons
    already (process_country filters once per country rather than once per
    tile). Cells are clipped to the tile's own bounds before the zonal pass,
    since a cell outside them cannot intersect the raster: without that, every
    tile scanned the entire country grid, making the cost n_tiles x n_cells.
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds

    grid_clean = grid_subset.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    if grid_clean.empty:
        return None

    slope_path = ensure_derived_tile(raster_path, "slope")
    tri_path = ensure_derived_tile(raster_path, "TRI")

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

    # NOTE: EPSG:6933 was proposed here as a "global" replacement, but it is an
    # equal-AREA projection and does not preserve length: its east-west scale
    # error runs from -13% at the equator to +51% at 55N. Measured against a
    # local UTM zone, EPSG:25394 reproduces Antigua's coastline to 0.3% while
    # EPSG:6933 is off by 9%. Keep 25394 until this is replaced by a per-country
    # UTM or a geodesic length, which is the only correct global fix.
    grid_line["coast_length_meters"] = grid_line.to_crs(epsg=25394).length

    merged = grid_country[["id"]].merge(grid_line[["id", "coast_length_meters"]], on="id", how="left").fillna(0)
    merged["with_coast"] = (merged["coast_length_meters"] > 0).astype(int)
    return merged

# -------------------------------------------------------------------
# Execution Logic
# -------------------------------------------------------------------


def process_country(iso, global_grid, global_shp, tiles_df, out_path, data_path):
    """Processes a single country's terrain and coastal data."""
    output_file = out_path / f"srtm_grid_data_{iso}.csv"
    if output_file.exists():
        logger.info(f"Skipping {iso}: Output exists.")
        return

    logger.info(f"Processing {iso}...")

    grid_c = global_grid[global_grid.iso3 == iso].copy()
    shp_c = global_shp[global_shp.GID_0 == iso].copy()

    # Drop antimeridian-crossing polygons once for the whole country. This used
    # to run inside every tile task, rescanning the full country grid n_tiles
    # times to reach the same answer. Kept separate from grid_c because the
    # coastal step below still wants the unfiltered cells.
    grid_terrain = grid_c[~grid_c["geometry"].apply(is_antimeridian_crossing)]

    # Retrieve pre-calculated tiles from tiles_df ---
    iso_tiles_data = tiles_df[tiles_df['iso3'] == iso]
    
    if iso_tiles_data.empty:
        tiles = []
    else:
        # Extract the list of tiles for this specific ISO
        tiles = iso_tiles_data['tiles'].values[0] 

    # Validate tiles against the filesystem
    tile_paths = [data_path / t for t in tiles if (data_path / t).exists()]

    if not tile_paths:
        logger.warning(f"No tiles found for {iso}.")
        df_terrain = grid_c[["id", "geometry"]].copy()
        for col in ["mean_elev", "mean_slope", "mean_rug"]: 
            df_terrain[col] = np.nan
    else:
        # Build the gdaldem derivatives up front, in parallel across all cores.
        # They are cached per tile, so a tile shared with an already-processed
        # neighbour costs nothing here.
        workers = max(1, min(len(tile_paths), cpu_count()))
        with Pool(processes=workers) as pool:
            pool.starmap(
                ensure_derived_tile,
                [(p, mode) for p in tile_paths for mode in ("slope", "TRI")],
            )
            tile_results = pool.starmap(
                run_terrain_task, [(grid_terrain, p) for p in tile_paths]
            )

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
    require_gdaldem()

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
    
    # Pre-calculate country tiles to handle antimeridian cases 
    logging.info("Pre-calculating SRTM tile overlaps for all countries...")
    tiles_df = get_country_tiles(grid)
    
    # Iterate countries sequentially to prevent process explosion
    # Multi-processing is handled at the tile level within each country.
    # This reads the grid rather than resolve_iso3_list() (GADM is already
    # open and re-reading it is expensive), so EXCLUDED_ISO3 is applied here.
    ISO3_LIST = [iso for iso in grid.iso3.unique() if iso not in EXCLUDED_ISO3]
    for iso in ISO3_LIST:
        try:
            # Pass tiles_df as an argument to the modified process_country function
            process_country(iso, grid, shp, tiles_df, out_path, data_path)
        except Exception as e:
            logger.error(f"Failed to process {iso}: {e}", exc_info=True)

    logger.info("Compiling global SRTM dataset...")
    all_csvs = list(out_path.glob("srtm_grid_data_*.csv"))
    
    if all_csvs:
        global_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        global_df.to_csv(out_path / "global_srtm_grid_data.csv", index=False)
        logging.info("Global SRTM processing complete.")
    else:
        logging.warning("No CSV files found to compile. Check for processing errors.")

if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_srtm()