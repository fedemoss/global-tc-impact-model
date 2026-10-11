import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import box

from src.config import EXCLUDED_ISO3, INPUT_DIR, OUTPUT_DIR, RASTER_WORKERS

logger = logging.getLogger(__name__)

# Each GLOFAS tile filename encodes the tile's lower-left corner; tiles are
# 10°×10° in WGS84.
_TILE_SIZE_DEG = 10.0
_TILE_PATTERN = re.compile(r"ID\d+_(N|S)(\d+)_([EW])(\d+)_RP10_depth_reclass\.tif")


def _tile_bounds(filename):
    """Return (minx, miny, maxx, maxy) for a GLOFAS tile filename, or None.

    The latitude in the filename is the tile's NORTHERN (top) edge and the
    longitude its WESTERN (left) edge -- verified against the georeferencing of
    all 271 tiles, which match this convention to within 0.03 degrees. Reading
    the latitude as the southern edge shifts every tile one full tile north and
    selects tiles that do not cover the country at all.
    """
    match = _TILE_PATTERN.search(os.path.basename(filename))
    if not match:
        return None
    lat_s, lat_v, lon_s, lon_v = match.groups()
    maxy = int(lat_v) if lat_s == "N" else -int(lat_v)
    minx = int(lon_v) if lon_s == "E" else -int(lon_v)
    return (minx, maxy - _TILE_SIZE_DEG, minx + _TILE_SIZE_DEG, maxy)


def select_tiles_by_country(tif_paths, grid_country, buffer_deg=0.5):
    """Select flood tiles whose extent intersects the country grid bounds."""
    if grid_country.empty:
        return []
    minx, miny, maxx, maxy = grid_country.total_bounds
    country_box = box(minx - buffer_deg, miny - buffer_deg,
                      maxx + buffer_deg, maxy + buffer_deg)

    selected = []
    for path in tif_paths:
        bounds = _tile_bounds(str(path))
        if bounds is None:
            continue
        if box(*bounds).intersects(country_box):
            selected.append(path)
    return selected


def process_single_country_flood(iso, grid_country, tile_paths, out_dir):
    """Calculates max flood risk per grid cell across all relevant tiles.

    `grid_country` is this country's cells only. It used to receive the whole
    global grid and filter inside, which meant every worker held all 2.2M cells
    and -- worse -- the zonal pass below ran over the full country grid once per
    tile. Cells are now clipped to each tile's bounds first, since a cell
    outside them cannot contribute to that tile's maximum anyway.
    """
    out_path = out_dir / f"flood_risk_{iso}.csv"
    if out_path.exists():
        logger.info(f"Skipping {iso}: file already exists")
        return

    if grid_country.empty:
        logger.warning(f"No grid cells for {iso}; skipping flood risk.")
        return

    relevant_tiles = select_tiles_by_country(tile_paths, grid_country)

    if not relevant_tiles:
        logger.warning(f"No flood tiles overlap {iso}; writing NaNs.")
        grid_country["flood_risk"] = np.nan
    else:
        # Running maximum per cell id. np.fmax propagates the non-NaN operand,
        # which is what the previous np.nanmax over the stacked per-tile rows
        # did, but without materialising n_tiles x n_cells values.
        running = pd.Series(np.nan, index=grid_country["id"].values, dtype=float)
        for tif in relevant_tiles:
            with rasterio.open(tif) as src:
                bounds = src.bounds
                sub = grid_country.cx[bounds.left:bounds.right,
                                      bounds.bottom:bounds.top]
                if sub.empty:
                    continue
                stats = zonal_stats(
                    sub, tif, stats=["max"],
                    nodata=src.nodata, all_touched=True,
                )
            vals = pd.Series(
                [s["max"] if s["max"] is not None else np.nan for s in stats],
                index=sub["id"].values, dtype=float,
            )
            running = pd.Series(
                np.fmax(running.values, vals.reindex(running.index).values),
                index=running.index,
            )

        grid_country["flood_risk"] = running.reindex(grid_country["id"].values).values

    grid_country[["id", "iso3", "flood_risk"]].to_csv(out_path, index=False)
    logger.info(f"Flood risk written for {iso} ({len(relevant_tiles)} tiles).")


def process_all_flood():
    tile_dir = INPUT_DIR / "FloodRisk" / "tiles"
    if not tile_dir.exists():
        logger.error(f"Flood tile directory not found: {tile_dir}")
        return
    tile_paths = [tile_dir / f for f in os.listdir(tile_dir) if f.endswith("_depth_reclass.tif")]
    if not tile_paths:
        logger.error(f"No flood tiles found in {tile_dir}")
        return

    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["GID_0"] = grid["iso3"]

    out_dir = OUTPUT_DIR / "FloodRisk" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_iso3 = [iso for iso in grid.iso3.unique() if iso not in EXCLUDED_ISO3]
    logger.info(f"Processing flood risk for {len(valid_iso3)} countries...")

    # Processes, not threads: zonal_stats spends its time in Python-level
    # per-cell work that holds the GIL, so the previous ThreadPoolExecutor ran
    # four workers at roughly one core's throughput. Each task is handed only
    # its own country's cells, so the global grid is never pickled.
    with ProcessPoolExecutor(max_workers=RASTER_WORKERS) as executor:
        futures = {
            executor.submit(
                process_single_country_flood,
                iso, grid[grid.iso3 == iso].copy(), tile_paths, out_dir,
            ): iso
            for iso in valid_iso3
        }
        for future in as_completed(futures):
            iso = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing flood risk for {iso}: {e}", exc_info=True)


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_flood()
