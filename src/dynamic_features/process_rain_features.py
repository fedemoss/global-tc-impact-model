import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterstats import zonal_stats

from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST
from src.utils.geo_utils import adjust_longitude

logger = logging.getLogger(__name__)


def get_date_list(df_meta, sid, days_to_landfall=2):
    metadata = df_meta.loc[df_meta.sid == sid].copy()
    metadata.loc[:, "landfalldate"] = pd.to_datetime(metadata["landfalldate"])
    start_date = metadata["landfalldate"] - dt.timedelta(days=days_to_landfall)
    end_date = metadata["landfalldate"] + dt.timedelta(days=days_to_landfall)
    return pd.date_range(start_date.iloc[0], end_date.iloc[0]).strftime("%Y%m%d").tolist()


def _zonal_mean_for_raster(grid, raster):
    """Compute the mean raster value within each grid polygon."""
    affine = raster.rio.transform()
    arr = raster.values
    if arr.ndim == 3:
        arr = arr[0]
    nodata = raster.rio.nodata
    stats = zonal_stats(
        grid.geometry, arr, affine=affine, stats=["mean"],
        nodata=nodata, all_touched=True,
    )
    return [s["mean"] if s["mean"] is not None else np.nan for s in stats]


def create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name):
    """Generate per-grid daily rainfall features from local GPM IMERG TIFFs."""
    date_list = get_date_list(df_meta=df_meta, sid=sid, days_to_landfall=2)
    local_gpm_dir = INPUT_DIR / "gpm_data" / typhoon_name

    if not local_gpm_dir.exists():
        raise FileNotFoundError(f"GPM directory missing for {typhoon_name}: {local_gpm_dir}")

    grid = grid_global[grid_global.iso3 == iso3].copy()
    if grid.empty:
        raise ValueError(f"No grid cells found for {iso3}")

    file_df = []
    for date_str in date_list:
        for file_path in local_gpm_dir.glob(f"*{date_str}*.tif"):
            raster = rxr.open_rasterio(file_path, masked=True).rio.write_crs(4326).squeeze(drop=True)
            grid_for_raster = grid.to_crs(raster.rio.crs) if grid.crs != raster.rio.crs else grid

            means = _zonal_mean_for_raster(grid_for_raster, raster)
            df_day = pd.DataFrame({
                "id": grid_for_raster["id"].values,
                "iso3": grid_for_raster["iso3"].values,
                # Source unit is 1/10 mm/hr per IMERG late-run gis convention.
                "mean": np.array(means, dtype=float) / 10.0,
                "date": date_str,
            })
            file_df.append(df_day)

    if not file_df:
        raise FileNotFoundError(f"No local GPM rasters matched dates for {typhoon_name} ({sid})")

    long_df = pd.concat(file_df, ignore_index=True)
    day_wide = long_df.pivot_table(index=["id", "iso3"], columns="date", values="mean", aggfunc="max")
    day_wide["rainfall_max_24h"] = day_wide.max(axis=1)
    day_wide = day_wide.reset_index()
    day_wide["sid"] = sid
    return day_wide[["id", "iso3", "sid", "rainfall_max_24h"]]


def _process_storm(args):
    iso3, sid, typhoon_name, metadata_country, grid_global = args
    df_meta = metadata_country[metadata_country.sid == sid]
    try:
        df_rainfall = create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name)
        return df_rainfall.fillna(0), None
    except Exception as e:
        logger.error(f"Failed to process {iso3}, {sid}: {e}")
        return None, pd.DataFrame([{"iso3": iso3, "sid": sid}])


def process_country_rainfall(iso3, metadata_global, grid_global, out_dir):
    out_file = out_dir / f"rainfall_data_{iso3}.csv"
    if out_file.exists():
        logger.info(f"Skipping {iso3}: file already exists")
        return

    metadata_country = metadata_global[metadata_global.iso3 == iso3]
    if metadata_country.empty:
        return

    df_rainfall_total = []
    not_working_cases = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        args_list = [
            (iso3, row.sid, row.typhoon, metadata_country, grid_global)
            for _, row in metadata_country.drop_duplicates("sid").iterrows()
        ]
        results = executor.map(_process_storm, args_list)

    for df_rainfall, not_working_case in results:
        if df_rainfall is not None:
            df_rainfall_total.append(df_rainfall)
        if not_working_case is not None:
            not_working_cases.append(not_working_case)

    if not_working_cases:
        nodata_path = out_dir / f"nodata_rainfall_{iso3}.csv"
        pd.concat(not_working_cases).to_csv(
            nodata_path,
            mode="a",
            header=not nodata_path.exists(),
            index=False,
        )

    if df_rainfall_total:
        pd.concat(df_rainfall_total).to_csv(out_file, index=False)
        logger.info(f"Rainfall data saved for {iso3}.")


def generate_all_rain_features(max_workers=4):
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading global grid and applying longitude adjustments...")
    grid_global = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid_global["iso3"] = grid_global.GID_0
    needs_wrap = grid_global.geometry.apply(
        lambda g: any(lon > 180 for lon, _ in g.exterior.coords) if g is not None else False
    )
    if needs_wrap.any():
        grid_global.loc[needs_wrap, "geometry"] = grid_global.loc[needs_wrap, "geometry"].apply(adjust_longitude)

    logger.info("Loading global metadata...")
    metadata_global = pd.read_csv(INPUT_DIR / "IBTRACS" / "merged" / "meta_data.csv")
    metadata_global = metadata_global.drop("DisNo.", axis=1).drop_duplicates()
    metadata_global["iso3"] = metadata_global.GID_0

    valid_iso3_list = [iso3 for iso3 in ISO3_LIST if iso3 in metadata_global["iso3"].unique()]

    logger.info(f"Starting rainfall processing for {len(valid_iso3_list)} countries...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_country_rainfall, iso3, metadata_global, grid_global, out_dir): iso3
            for iso3 in valid_iso3_list
        }
        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()
                logger.info(f"Successfully processed rainfall for {iso3}")
            except Exception as e:
                logger.error(f"Error processing {iso3}: {e}", exc_info=True)


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    generate_all_rain_features(max_workers=4)
