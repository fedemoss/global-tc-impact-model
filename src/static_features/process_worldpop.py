"""Aggregate the WorldPop 1km mosaics to grid-cell level, one file per anchor year.

Output: OUTPUT_DIR/Worldpop/grid_data/population_grid_{iso3}_{year}.csv

Population is time dependent in this pipeline: every anchor year is aggregated
here, and dataset_builder interpolates between them to the year of each event
(see utils/time_interpolation.py).

Two defects of the single-year version are fixed.

* The raster handle used to be opened once and shared across worker threads.
  GDAL datasets are not thread-safe, which produces spurious "Corrupted LZW
  table" / "TIFFReadEncodedTile() failed" errors and silently drops whole
  countries. Each worker now opens its own handle, and the work is split across
  processes rather than threads - the per-cell masking is Python-level and holds
  the GIL, so a thread pool ran slower than a single thread.
* Population was summed as `out_image[out_image > 0]`. That works for the
  2005-2020 rasters, whose nodata is -3.4e38 and so drops out of a "> 0" filter
  by itself, but ppp_2000 encodes nodata as *positive* +3.4e38: a single such
  pixel would have swamped the entire population of a cell. Nodata is now
  excluded explicitly, which is correct whichever sign a year happens to use.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

from src.utils.geo_utils import adjust_longitude
from src.config import (
    INPUT_DIR, OUTPUT_DIR, POP_ANCHOR_YEARS, RASTER_CHUNK_SIZE, RASTER_WORKERS,
    population_grid_path, resolve_iso3_list,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def worldpop_raster_path(year):
    return INPUT_DIR / "Worldpop" / f"ppp_{year}_1km_Aggregated.tif"




def calculate_population(geometry, raster):
    """Population inside one grid cell.

    WorldPop is 30 arcsec, so a 0.1 degree cell covers roughly 144 pixels and
    the value is their sum - not a single pixel lookup.
    """
    out_image, _ = mask(raster, [geometry], crop=True)
    out_image = out_image[0].astype("float64")
    valid = (out_image > 0) & (out_image != raster.nodata) & np.isfinite(out_image)
    return float(out_image[valid].sum())


def _process_chunk(args):
    iso3, chunk_id, chunk_grid, raster_path = args
    try:
        # One handle per worker: GDAL datasets are not safe to share
        with rasterio.open(raster_path) as raster:
            grid = chunk_grid.to_crs(raster.crs)
            population = [calculate_population(geom, raster) for geom in grid["geometry"]]
        return iso3, chunk_id, pd.DataFrame({
            "id": grid["id"].to_numpy(),
            "iso3": grid["iso3"].to_numpy(),
            "population": population,
        }), None
    except Exception as e:
        logging.error(f"Error processing {iso3} chunk {chunk_id}: {e}")
        return iso3, chunk_id, None, str(e)


def _write_country(iso3, year, chunks):
    df = pd.concat([chunks[k] for k in sorted(chunks)], ignore_index=True)
    out_path = population_grid_path(iso3, year)
    tmp = out_path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.rename(out_path)
    return out_path


def process_worldpop_year(grid, year, iso3_list):
    """Aggregate one anchor year for every country still missing it."""
    raster_path = worldpop_raster_path(year)
    if not raster_path.exists():
        logging.warning(f"[{year}] raster missing at {raster_path}, skipping year")
        return

    out_dir = OUTPUT_DIR / "Worldpop" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, expected, size = [], {}, {}
    for iso3 in iso3_list:
        if population_grid_path(iso3, year).exists():
            continue
        grid_country = grid[grid.iso3 == iso3].reset_index(drop=True)
        if grid_country.empty:
            continue
        size[iso3] = len(grid_country)
        n = 0
        for start in range(0, len(grid_country), RASTER_CHUNK_SIZE):
            tasks.append((iso3, n, grid_country.iloc[start:start + RASTER_CHUNK_SIZE], raster_path))
            n += 1
        expected[iso3] = n

    if not tasks:
        print(f"[{year}] all countries already processed")
        return

    # Largest countries first so they do not end up running alone at the end
    tasks.sort(key=lambda t: (-size[t[0]], t[0], t[1]))
    print(f"[{year}] {len(expected)} countries, {len(tasks)} chunks")

    collected = {iso3: {} for iso3 in expected}
    failed = set()
    with ProcessPoolExecutor(max_workers=RASTER_WORKERS) as executor:
        for iso3, chunk_id, df, error in executor.map(_process_chunk, tasks):
            if error is not None:
                failed.add(iso3)
                continue
            collected[iso3][chunk_id] = df
            if iso3 not in failed and len(collected[iso3]) == expected[iso3]:
                path = _write_country(iso3, year, collected.pop(iso3))
                print(f"[{year}] {iso3} -> {path.name}")

    if failed:
        logging.error(f"[{year}] countries left incomplete: {sorted(failed)}")


def process_all_worldpop():
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    if "iso3" not in grid.columns:
        grid["iso3"] = grid["GID_0"]
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    grid = grid[["id", "iso3", "geometry"]]

    iso3_list = [iso for iso in resolve_iso3_list() if iso in set(grid.iso3)]
    print(f"Grid loaded: {len(grid)} cells, {len(iso3_list)} countries")

    for year in POP_ANCHOR_YEARS:
        process_worldpop_year(grid, year, iso3_list)


if __name__ == "__main__":
    process_all_worldpop()
