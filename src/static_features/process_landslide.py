import logging

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats

from src.config import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def aggregate_raster_to_grid(raster_path, grid):
    with rasterio.open(raster_path) as src:
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        stats = zonal_stats(
            grid, raster_path, stats=["sum"],
            nodata=src.nodata, affine=src.transform, all_touched=True
        )

    grid["landslide_risk_sum"] = [stat["sum"] if stat["sum"] is not None else np.nan for stat in stats]
    return grid


def process_all_landslide():
    grid_global = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid_global["GID_0"] = grid_global["iso3"]

    raster_path = INPUT_DIR / "LandSlides" / "landslide_data.tif"

    logger.info("Processing Landslide Risk data...")
    grid_landslide = aggregate_raster_to_grid(raster_path, grid_global)[["id", "iso3", "landslide_risk_sum"]]

    out_folder = OUTPUT_DIR / "LandSlides"
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file_path = out_folder / "global_grid_landslide_risk.csv"

    grid_landslide.to_csv(out_file_path, index=False)
    logger.info(f"Saved Landslide Risk data to {out_file_path}")


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_landslide()