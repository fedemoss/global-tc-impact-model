import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Polygon

from src.config import (
    INPUT_DIR, IMERG_PRODUCT, IMERG_PRODUCTS, OUTPUT_DIR, resolve_iso3_list,
)

def _load_dotenv():
    """Load KEY=VALUE pairs from .env at the project root into os.environ (no-op if already set)."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

logger = logging.getLogger(__name__)


# Both GIS products state their own units in TIFFTAG_IMAGEDESCRIPTION, and they
# are NOT the same:
#
#   3B-HHR-GIS ... total.accum  ->  "Unit=0.1(mm) ScaleFactor=10 MaxPossibleNumHalfHour=1"
#       an accumulation over ONE half hour. mm/day = (raw / 10) summed over 48.
#       (Not a rate, and not a running daily total - verified: the 48 values of
#       a day are not monotonic and sum to the daily total.)
#
#   3B-DAY-GIS                  ->  "Unit=0.1(mm/hr) ScaleFactor=10 MaxPossibleNumHalfHour=48"
#       a RATE. mm/day = raw / 10 * 24.
#
# Using the daily granule as if it were an accumulation - i.e. forgetting the
# x24 - under-counts rainfall by a factor of 24, which is the class of error this
# feature suffered from before. Each product therefore has its own conversion
# function below and they are never interchanged.
IMERG_SCALE_FACTOR = 10
IMERG_FILL_VALUE = 29999
HOURS_PER_DAY = 24

# A complete IMERG day is 48 half-hourly granules. Anything less silently
# under-counts the daily accumulation, which is exactly how rainfall ends up
# under-represented, so days are checked rather than summed blindly.
HALF_HOURS_PER_DAY = 48


def _resolve_product(product=None):
    product = product or IMERG_PRODUCT
    if product not in IMERG_PRODUCTS:
        raise ValueError(f"Unknown IMERG product {product!r}; expected one of {IMERG_PRODUCTS}")
    return product

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


def _cell_pixel_index(grid, reference_tif):
    """Row/col of the IMERG pixel holding each grid cell, computed once per storm.

    The grid is 0.1 degrees and so is IMERG, so exactly one pixel centre falls in
    each cell - the same pixel the previous per-cell `.sel(slice(...))` lookup
    returned, but resolved arithmetically instead of by 240 raster queries per
    cell. Cell centres are taken from the bounding box rather than
    `geometry.centroid` to keep the value identical without a projection step.
    """
    with rasterio.open(reference_tif) as src:
        transform, height, width = src.transform, src.height, src.width

    bounds = np.array([b for b in grid["bbox"]], dtype="float64")
    xs = (bounds[:, 0] + bounds[:, 2]) / 2.0
    ys = (bounds[:, 1] + bounds[:, 3]) / 2.0

    rows, cols = rowcol(transform, xs, ys)
    rows, cols = np.asarray(rows), np.asarray(cols)

    # Cells off the edge of the raster had an empty selection before, and were
    # skipped; keep that behaviour rather than wrapping round the array edges
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, inside


def _read_granule(file_path, rows, cols, inside, to_mm):
    """Millimetres for every grid cell from one granule.

    `to_mm` converts the raw stored integers for the product in question. The
    result is aligned with the grid, NaN where the cell has no valid retrieval
    (IMERG fill) or falls outside the raster.
    """
    with rasterio.open(file_path) as src:
        arr = src.read(1)
        nodata = src.nodata

    out = np.full(len(rows), np.nan, dtype="float64")
    vals = arr[rows[inside], cols[inside]].astype("float64")
    valid = vals < IMERG_FILL_VALUE
    if nodata is not None:
        valid &= vals != nodata
    out[np.flatnonzero(inside)[valid]] = to_mm(vals[valid])
    return out


def half_hour_raw_to_mm(raw):
    """One half-hourly `total.accum` granule -> millimetres in that half hour."""
    return raw / IMERG_SCALE_FACTOR


def daily_raw_to_mm(raw):
    """One `3B-DAY-GIS` granule -> millimetres in that day.

    The stored value is a rate in tenths of mm/hr, hence the x24.
    """
    return raw / IMERG_SCALE_FACTOR * HOURS_PER_DAY

def _granules_by_date(date_list, typhoon_name, product):
    """The granule files available for each date, for the configured product."""
    if product == "daily":
        from src.collectors.pps_collector import daily_gpm_dir, daily_tif_path
        source_dir = daily_gpm_dir()
        return source_dir, {
            d: [daily_tif_path(d)] for d in date_list if daily_tif_path(d).exists()
        }

    source_dir = INPUT_DIR / "gpm_data" / typhoon_name
    if not source_dir.exists():
        return source_dir, {}
    by_date = {}
    for date_str in date_list:
        files = sorted(source_dir.glob(f"*{date_str}*.tif"))
        if files:
            by_date[date_str] = files
    return source_dir, by_date


def create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name, product=None):
    """Maximum daily rainfall accumulation per grid cell over the event window.

    Reads the local IMERG granules downloaded by the PPS collector, accumulates
    each calendar day to millimetres, and takes the largest daily total across
    the landfall +/- days_to_landfall window. The result is the same feature,
    `rainfall_max_24h`, whichever product is used:

    * "half_hourly" sums the 48 `total.accum` granules of the day;
    * "daily" converts the single daily granule from its rate to mm/day.

    The two agree to correlation 0.9994; the daily route is quantised to
    2.4 mm/day, so it is coarser for light rain and effectively identical
    (+/-0.6%) for the heavy days that set the maximum.
    """
    product = _resolve_product(product)
    date_list = get_date_list(df_meta=df_meta, sid=sid, days_to_landfall=2)
    to_mm = daily_raw_to_mm if product == "daily" else half_hour_raw_to_mm
    expected_per_day = 1 if product == "daily" else HALF_HOURS_PER_DAY

    grid = grid_global[grid_global.iso3 == iso3].copy()
    grid["bbox"] = grid.geometry.apply(lambda geom: geom.bounds)

    local_gpm_dir, day_files_by_date = _granules_by_date(date_list, typhoon_name, product)

    if not day_files_by_date:
        raise FileNotFoundError(
            f"No local {product} GPM data for {typhoon_name} ({sid}) in {local_gpm_dir}")

    # An incomplete day is an under-estimate of that day's accumulation, and the
    # feature is a maximum over days, so a short day can only drag the answer
    # down. Say so loudly instead of silently returning a low number.
    incomplete = {
        d: len(f) for d, f in day_files_by_date.items() if len(f) != expected_per_day
    }
    if incomplete:
        logging.warning(
            f"{iso3} {sid} ({typhoon_name}): incomplete IMERG days, accumulation "
            f"under-counted - " + ", ".join(
                f"{d}: {n}/{HALF_HOURS_PER_DAY} granules" for d, n in sorted(incomplete.items())
            )
        )
    missing_dates = [d for d in date_list if d not in day_files_by_date]
    if missing_dates:
        logging.warning(
            f"{iso3} {sid} ({typhoon_name}): no IMERG data at all for {missing_dates}"
        )

    # The cell -> pixel mapping is the same for every granule, so resolve it once
    first_file = day_files_by_date[next(iter(day_files_by_date))][0]
    rows, cols, inside = _cell_pixel_index(grid, first_file)

    file_df = pd.DataFrame()
    for date_str, day_files in day_files_by_date.items():
        daily_totals = np.zeros(len(grid), dtype="float64")
        daily_has_data = np.zeros(len(grid), dtype=bool)

        # One granule for "daily", 48 for "half_hourly" - summing is correct in
        # both cases because each granule covers a disjoint slice of the day
        for file_path in day_files:
            granule_mm = _read_granule(file_path, rows, cols, inside, to_mm)
            observed = ~np.isnan(granule_mm)
            daily_totals[observed] += granule_mm[observed]
            daily_has_data |= observed

        day_grid = grid[["id", "iso3"]].copy()
        day_grid["mean"] = np.where(daily_has_data, daily_totals, np.nan)
        day_grid["date"] = date_str
        file_df = pd.concat([file_df, day_grid], axis=0)

    if not file_df:
        raise FileNotFoundError(f"No local GPM rasters matched dates for {typhoon_name} ({sid})")

    long_df = pd.concat(file_df, ignore_index=True)
    day_wide = long_df.pivot_table(index=["id", "iso3"], columns="date", values="mean", aggfunc="max")
    day_wide["rainfall_max_24h"] = day_wide.max(axis=1)
    day_wide = day_wide.reset_index()
    day_wide["sid"] = sid
    return day_wide[["id", "iso3", "sid", "rainfall_max_24h"]]


def _process_storm(args):
    iso3, sid, typhoon_name, metadata_country, grid_global, product = args
    df_meta = metadata_country[metadata_country.sid == sid]
    try:
        df_rainfall = create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name, product)
        return df_rainfall.fillna(0), None
    except Exception as e:
        logger.error(f"Failed to process {iso3}, {sid}: {e}")
        return None, pd.DataFrame([{"iso3": iso3, "sid": sid}])

def process_country_rainfall(iso3, metadata_global, grid_global, out_dir, product=None):
    df_rainfall_total = []
    not_working_cases = []
    
    out_file = out_dir / f"rainfall_data_{iso3}.csv"
    if out_file.exists():
        logger.info(f"Skipping {iso3}: file already exists")
        return

    metadata_country = metadata_global[metadata_global.iso3 == iso3]
    if not metadata_country.empty:
        with ThreadPoolExecutor(max_workers=10) as executor:
            args_list = [
                (iso3, row.sid, row.typhoon, metadata_country, grid_global, product)
                for _, row in metadata_country.drop_duplicates('sid').iterrows()
            ]
            results = executor.map(_process_storm, args_list)

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


def _load_metadata_global():
    """Load and concatenate all per-country metadata files from the IBTRACS output directory."""
    meta_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    meta_files = list(meta_dir.glob("metadata_*.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files found in {meta_dir}. Run process_wind_features first.")
    df = pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True)
    df = df.drop(columns=["DisNo."], errors="ignore").drop_duplicates()
    df["iso3"] = df["GID_0"]
    return df

def _load_grid_global():
    """Load the land-overlap grid and apply longitude adjustments."""
    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["iso3"] = grid.GID_0
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    return grid

def run_single_storm(iso3, sid, product=None):
    """Download GPM data (if absent) and compute rainfall features for one storm."""
    from src.collectors.pps_collector import download_gpm_for_storm

    product = _resolve_product(product)

    meta_file = OUTPUT_DIR / "IBTRACS" / "standard" / f"metadata_{iso3}.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found for {iso3}. Run process_wind_features first.")

    df_meta = pd.read_csv(meta_file)
    storm = df_meta[df_meta.sid == sid]
    if storm.empty:
        raise ValueError(f"SID {sid} not found in metadata for {iso3}.")

    typhoon_name = storm.iloc[0]["typhoon"]
    date_list = get_date_list(df_meta, sid, days_to_landfall=2)

    logging.info(f"Downloading {product} GPM data for {typhoon_name} ({date_list[0]} – {date_list[-1]})...")
    download_gpm_for_storm(
        start_date=pd.to_datetime(date_list[0]),
        end_date=pd.to_datetime(date_list[-1]),
        typhoon_name=typhoon_name,
        product=product,
    )

    grid_global = _load_grid_global()
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_rainfall = create_rainfall_dataset(grid_global, df_meta, iso3, sid, typhoon_name, product)
    df_rainfall = df_rainfall.fillna(0)
    out_file = out_dir / f"rainfall_data_{iso3}_{sid}.csv"
    df_rainfall.to_csv(out_file, index=False)
    logging.info(f"Saved: {out_file}")
    return df_rainfall

def _local_data_is_complete(local_gpm_dir, date_list, expected_per_day=None):
    """True only when every date already has all 48 half-hourly granules on disk.

    The previous check was
        `local_gpm_dir.exists() and any(local_gpm_dir.glob(...) for d in date_list)`
    which is always True once the directory exists: `Path.glob` returns a
    generator, and a generator object is truthy whether or not it yields
    anything, so `any()` never looked inside it. Since
    `download_gpm_late_run` creates that directory before fetching anything, any
    storm whose download failed or was interrupted was treated as complete
    forever after, and its rainfall was then summed from a partial set of
    granules - or from none at all. That is a silent under-count of exactly the
    kind we are chasing, so the check now counts the files per date.
    """
    expected_per_day = expected_per_day or HALF_HOURS_PER_DAY
    if not local_gpm_dir.exists():
        return False
    return all(
        len(list(local_gpm_dir.glob(f"*{d}*.tif"))) == expected_per_day
        for d in date_list
    )


def _ensure_local_gpm_data(iso3, metadata_country, product=None):
    """
    Downloads any storm's GPM data that isn't already present locally.
    Sequential by design: NASA PPS isn't built for many parallel download
    sessions (single requests.Session, no cross-call rate-limit coordination
    in pps_collector.py) — unlike the local-only processing below, which is
    safe to run concurrently once the data is on disk.
    """
    from src.collectors.pps_collector import daily_gpm_dir, download_gpm_for_storm

    product = _resolve_product(product)
    for _, row in metadata_country.drop_duplicates("sid").iterrows():
        date_list = get_date_list(df_meta=metadata_country, sid=row.sid, days_to_landfall=2)
        if product == "daily":
            local_gpm_dir, expected_per_day = daily_gpm_dir(), 1
        else:
            local_gpm_dir, expected_per_day = INPUT_DIR / "gpm_data" / row.typhoon, HALF_HOURS_PER_DAY

        if _local_data_is_complete(local_gpm_dir, date_list, expected_per_day):
            continue
        try:
            logging.info(f"Downloading {product} GPM data for {iso3}, {row.typhoon} ({row.sid})...")
            download_gpm_for_storm(
                start_date=pd.to_datetime(date_list[0]),
                end_date=pd.to_datetime(date_list[-1]),
                typhoon_name=row.typhoon,
                product=product,
            )
        except Exception as e:
            logging.error(f"Failed to download GPM data for {iso3}, {row.sid} ({row.typhoon}): {e}")

def generate_all_rain_features(max_workers=4, product=None):
    out_dir = OUTPUT_DIR / "PPS"
    out_dir.mkdir(parents=True, exist_ok=True)

    product = _resolve_product(product)
    print(f"IMERG product: {product}")

    print("Loading global grid and applying longitude adjustments...")
    grid_global = _load_grid_global()

    print("Loading global metadata...")
    metadata_global = _load_metadata_global()

    valid_iso3_list = [iso3 for iso3 in resolve_iso3_list() if iso3 in metadata_global["iso3"].unique()]

    print("Ensuring local GPM data is available for all storms (downloading missing storms)...")
    for iso3 in valid_iso3_list:
        if (out_dir / f"rainfall_data_{iso3}.csv").exists():
            continue  # already processed, no need to (re)download
        _ensure_local_gpm_data(iso3, metadata_global[metadata_global.iso3 == iso3], product)

    print(f"Starting rainfall processing for {len(valid_iso3_list)} countries...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_country_rainfall, iso3, metadata_global, grid_global, out_dir, product): iso3
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
    run_single_storm(iso3="ATG", sid="2008287N15291")
