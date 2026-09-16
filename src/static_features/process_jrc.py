"""Aggregate GHS-SMOD to grid-cell level, one file per epoch.

Output: OUTPUT_DIR/JRC/grid_data/degree_of_urbanization_{iso3}_{epoch}.csv

Like population, the degree of urbanisation is time dependent here: every anchor
epoch is aggregated, and dataset_builder interpolates between them to the year
of each event.

Notes on correctness, all of which bit the earlier single-epoch pipeline:

* `zonal_stats(stats="count", categorical=True)` returns the per-class pixel
  counts *and* a "count" column holding their total. Summing the whole frame
  therefore double counts, and every fraction comes out at about half its true
  value. Only the documented SMOD classes take part in the denominator here.
* `urban` must include class 30, the Urban Centre class - the densest
  settlements, 639k pixels globally in E2020, more than classes 22 and 23 put
  together. Leaving it out drops exactly the dense cores where cyclone impact
  concentrates. The two halves are also written separately so the previous
  definition stays reconstructible.
* Nodata in GHS-SMOD R2022A is -200, not 128. Rather than hardcode a fill value,
  anything outside the eight documented classes is dropped, which also handles
  the fill introduced by the reprojection.

The Mollweide -> WGS84 reprojection is the expensive step, so each epoch is
reprojected once and cached under INPUT_DIR/JRC/reprojected (~17 MB per epoch).
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterstats import zonal_stats
from shapely.geometry import Polygon

from src.config import (
    INPUT_DIR, OUTPUT_DIR, RASTER_CHUNK_SIZE, RASTER_WORKERS, SMOD_EPOCH_YEARS,
    resolve_iso3_list, urbanization_grid_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# GHS-SMOD degree of urbanisation classes (R2022A)
WATER_CLASSES = [10]
RURAL_CLASSES = [11, 12, 13]           # very low density / low density / rural cluster
URBAN_CLUSTER_CLASSES = [21, 22, 23]   # suburban / semi-dense / dense cluster
URBAN_CENTRE_CLASSES = [30]            # urban centre
VALID_CLASSES = WATER_CLASSES + RURAL_CLASSES + URBAN_CLUSTER_CLASSES + URBAN_CENTRE_CLASSES

OUT_COLUMNS = ["id", "urban", "rural", "water", "urban_cluster", "urban_centre", "n_valid_pixels"]


def smod_raster_path(epoch):
    return INPUT_DIR / "JRC" / f"GHS_SMOD_{epoch}_GLOBE_R2022A_54009_1000_V1_0.tif"


def smod_reprojected_path(epoch):
    return INPUT_DIR / "JRC" / "reprojected" / f"GHS_SMOD_{epoch}_wgs84.tif"


from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST
from src.utils.geo_utils import adjust_longitude

logger = logging.getLogger(__name__)



def ensure_reprojected(epoch, crs):
    """Reproject one epoch to the grid CRS once and keep it on disk."""
    out_path = smod_reprojected_path(epoch)
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{epoch}] reprojecting to {crs}...")
    src = rxr.open_rasterio(smod_raster_path(epoch))
    src_wgs84 = src.rio.reproject(crs)

    tmp = out_path.with_suffix(".tif.part")
    src_wgs84.rio.to_raster(tmp, driver="GTiff", tiled=True, compress="LZW")
    tmp.rename(out_path)
    del src, src_wgs84
    return out_path


def class_counts_to_fractions(counts):
    """Per-class pixel counts -> class fractions, over the valid classes only."""
    class_cols = [c for c in counts.columns if c in VALID_CLASSES]
    classified = counts[class_cols] if class_cols else counts[[]]
    total = classified.sum(axis=1)

    def share(classes):
        present = [c for c in classes if c in class_cols]
        if not present:
            return pd.Series(0.0, index=counts.index)
        return classified[present].sum(axis=1).div(total).fillna(0.0)

    fractions = pd.DataFrame(index=counts.index)
    fractions["water"] = share(WATER_CLASSES)
    fractions["rural"] = share(RURAL_CLASSES)
    fractions["urban_cluster"] = share(URBAN_CLUSTER_CLASSES)
    fractions["urban_centre"] = share(URBAN_CENTRE_CLASSES)
    fractions["urban"] = fractions["urban_cluster"] + fractions["urban_centre"]
    # Cells whose pixels are all nodata carry no information; total == 0 there
    fractions["n_valid_pixels"] = total
    return fractions


def calculate_urban_rural_water(grid, raster_file):
    """Class fractions for one block of grid cells."""
    stats = zonal_stats(grid["geometry"], str(raster_file), stats="count", categorical=True)
    counts = pd.DataFrame(stats).fillna(0)
    fractions = class_counts_to_fractions(counts)
    fractions.insert(0, "id", grid["id"].to_numpy())
    return fractions[OUT_COLUMNS]


def _process_chunk(args):
    iso3, chunk_id, chunk_grid, raster_file = args
    try:
        return iso3, chunk_id, calculate_urban_rural_water(chunk_grid, raster_file), None
    except Exception as e:
        logging.error(f"Error processing {iso3} chunk {chunk_id}: {e}")
        return iso3, chunk_id, None, str(e)


def _write_country(iso3, epoch, chunks):
    df = pd.concat([chunks[k] for k in sorted(chunks)], ignore_index=True)
    out_path = urbanization_grid_path(iso3, epoch)
    tmp = out_path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.rename(out_path)
    return out_path


def process_jrc_epoch(grid, epoch, iso3_list):
    if not smod_raster_path(epoch).exists():
        logging.warning(f"[{epoch}] raster missing, skipping epoch")
        return

    out_dir = OUTPUT_DIR / "JRC" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    raster_file = ensure_reprojected(epoch, grid.crs)

    tasks, expected, size = [], {}, {}
    for iso3 in iso3_list:
        if urbanization_grid_path(iso3, epoch).exists():
            continue
        grid_country = grid[grid.iso3 == iso3].reset_index(drop=True)
        if grid_country.empty:
            continue
        size[iso3] = len(grid_country)
        n = 0
        for start in range(0, len(grid_country), RASTER_CHUNK_SIZE):
            tasks.append((iso3, n, grid_country.iloc[start:start + RASTER_CHUNK_SIZE], raster_file))
            n += 1
        expected[iso3] = n

    if not tasks:
        print(f"[{epoch}] all countries already processed")
        return

    tasks.sort(key=lambda t: (-size[t[0]], t[0], t[1]))
    print(f"[{epoch}] {len(expected)} countries, {len(tasks)} chunks")

    collected = {iso3: {} for iso3 in expected}
    failed = set()
    with ProcessPoolExecutor(max_workers=RASTER_WORKERS) as executor:
        for iso3, chunk_id, df, error in executor.map(_process_chunk, tasks):
            if error is not None:
                failed.add(iso3)
                continue
            collected[iso3][chunk_id] = df
            if iso3 not in failed and len(collected[iso3]) == expected[iso3]:
                path = _write_country(iso3, epoch, collected.pop(iso3))
                print(f"[{epoch}] {iso3} -> {path.name}")

    if failed:
        logging.error(f"[{epoch}] countries left incomplete: {sorted(failed)}")


def process_all_jrc():
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    if "iso3" not in grid.columns:
        grid["iso3"] = grid["GID_0"]
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    grid = grid[["id", "iso3", "geometry"]]

    iso3_list = [iso for iso in resolve_iso3_list() if iso in set(grid.iso3)]
    print(f"Grid loaded: {len(grid)} cells, {len(iso3_list)} countries")

    for epoch in SMOD_EPOCH_YEARS:
        process_jrc_epoch(grid, epoch, iso3_list)


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_jrc()
