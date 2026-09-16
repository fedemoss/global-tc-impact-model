<<<<<<< HEAD
"""Downloading IMERG rainfall granules from NASA PPS.

Two products are supported, selected by `IMERG_PRODUCT` in src/config.py:

* "half_hourly" (default) - `download_gpm_late_run`, 48 `total.accum` granules
  per day under `INPUT_DIR/gpm_data/<typhoon>/`.
* "daily" - `download_gpm_daily_run`, one `3B-DAY-GIS` granule per date under
  `INPUT_DIR/gpm_data_daily/`, shared across storms.

`download_gpm_for_storm` dispatches between them, so callers do not need to know
which product is configured.

Note both are the IMERG **Final** run (no "-L" in the product name), which is
gauge-calibrated - appropriate for a historical record, unlike the Late run used
for near-real-time work.
"""
import io
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.config import INPUT_DIR, IMERG_PRODUCT, IMERG_PRODUCTS, NASA_PPS_BASE_URL
=======
import logging
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import INPUT_DIR
>>>>>>> 2aaf917cea7caa556c4f871607c174621f1bc43f

logger = logging.getLogger(__name__)

USERNAME = os.getenv("NASA_PPS_USERNAME")
PASSWORD = os.getenv("NASA_PPS_PASSWORD")

MAX_RETRIES = 3

# A complete IMERG day is 48 half-hourly granules. The download is the first
# place an incomplete day can appear, and a short day silently under-counts the
# accumulated rainfall downstream, so shortfalls are reported here rather than
# discovered as suspiciously low rain totals later.
HALF_HOURS_PER_DAY = 48

def gis_url(date):
    """The PPS GIS listing for one day."""
    return f"{NASA_PPS_BASE_URL}{date.year}/{date.month:02d}/{date.day:02d}/gis"


def list_files(url, session, suffix=".zip"):
    """Names in a PPS listing ending with `suffix`, as absolute URLs.

    The granule filenames carry a varying numeric field, so they are read from
    the listing rather than constructed.
    """
    response = session.get(url, auth=(USERNAME, PASSWORD))
    if not response.ok:
        logging.error(f"Listing {url} failed with HTTP {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        url + "/" + node.get("href").split("/")[-1]
        for node in soup.find_all("a")
        if node.get("href", "").endswith(suffix)
    ]


def daily_gpm_dir():
    """Daily granules are addressed by date, so they are shared across storms.

    Half-hourly granules stay under gpm_data/<typhoon>/ because that is how the
    existing downloads are laid out; the daily ones dedupe naturally, and storms
    in the same basin and season often share dates.
    """
    return INPUT_DIR / "gpm_data_daily"


def daily_tif_path(date_str):
    return daily_gpm_dir() / f"imerg_daily_{date_str}.tif"

def _download_one(zip_url, download_path, session):
    """Fetch one granule. Returns True only if the .tif is on disk afterwards."""
    zip_name = zip_url.split("/")[-1]
    file_name = zip_name.replace(".zip", ".total.accum.tif")
    file_path = download_path / file_name

    if file_path.exists():
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(zip_url, auth=(USERNAME, PASSWORD), timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                member = next((m for m in zf.namelist() if m.endswith("total.accum.tif")), None)
                if member is None:
                    logging.error(f"No total.accum.tif inside {zip_url}")
                    return False
                # Write to a temporary name first: a half-written tif left behind
                # by an interrupted run would otherwise look like a complete
                # granule and quietly under-count the day
                tmp_path = file_path.with_suffix(file_path.suffix + ".part")
                with zf.open(member) as src, open(tmp_path, "wb") as dst:
                    dst.write(src.read())
                tmp_path.rename(file_path)
            return True
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            if attempt == MAX_RETRIES:
                logging.error(f"Giving up on {zip_url} after {MAX_RETRIES} attempts: {e}")
                return False
            time.sleep(2 ** attempt)
    return False

def download_gpm_late_run(start_date, end_date, typhoon_name, max_workers=6):
    if not USERNAME or not PASSWORD:
        raise ValueError("NASA PPS credentials not set in environment variables.")

    date_list = pd.date_range(start_date, end_date)

    download_path = INPUT_DIR / "gpm_data" / typhoon_name
    download_path.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for date in date_list:
            day = f"{date.day:02d}"
            month = f"{date.month:02d}"
            year = date.year

<<<<<<< HEAD
            url = gis_url(date)
            zip_files = list_files(url=url, session=session, suffix=".zip")

            prefix = f'{url}/3B-HHR-GIS.MS.MRG.3IMERG.{year}{month}{day}'
            filtered_files = [f for f in zip_files if f.startswith(prefix)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    lambda zip_url: _download_one(zip_url, download_path, session),
                    filtered_files,
                ))

            on_disk = len(list(download_path.glob(f"*{year}{month}{day}*.tif")))
            if on_disk != HALF_HOURS_PER_DAY:
                logging.warning(
                    f"{typhoon_name} {year}{month}{day}: {on_disk}/{HALF_HOURS_PER_DAY} "
                    f"granules on disk ({len(filtered_files)} listed, "
                    f"{sum(results)} fetched) - daily accumulation will be under-counted"
                )
    print(f"Finished downloading rainfall data for {typhoon_name}")


def _download_daily_one(date, session):
    """Fetch the daily granule for one date. Returns True if it is on disk after.

    The daily product is served as a plain .tif (there is a .zip alongside it,
    but the .tif is the smaller download).
    """
    date_str = date.strftime("%Y%m%d")
    target = daily_tif_path(date_str)
    if target.exists() and target.stat().st_size > 0:
        return True

    tif_urls = [u for u in list_files(gis_url(date), session, suffix=".tif")
                if "3B-DAY-GIS" in u]
    if not tif_urls:
        logging.error(f"No 3B-DAY-GIS granule listed for {date_str}")
        return False
    if len(tif_urls) > 1:
        logging.warning(f"{date_str}: {len(tif_urls)} daily granules listed, taking the first")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(tif_urls[0], auth=(USERNAME, PASSWORD), timeout=300)
            response.raise_for_status()
            # Write under a temporary name so an interrupted run cannot leave a
            # truncated raster that the skip-if-exists check would trust
            tmp = target.with_suffix(".tif.part")
            tmp.write_bytes(response.content)
            tmp.rename(target)
            return True
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                logging.error(f"Giving up on {date_str} after {MAX_RETRIES} attempts: {e}")
                return False
            time.sleep(2 ** attempt)
    return False


def download_gpm_daily_run(start_date, end_date, typhoon_name=None, max_workers=4):
    """Download one IMERG daily granule per date in the window.

    `typhoon_name` is accepted and ignored, so this can stand in for
    `download_gpm_late_run`: daily granules are keyed by date and shared between
    storms, which is most of why this route is ~50x smaller.
    """
    if not USERNAME or not PASSWORD:
        raise ValueError("NASA PPS credentials not set in environment variables.")

    daily_gpm_dir().mkdir(parents=True, exist_ok=True)
    date_list = pd.date_range(start_date, end_date)

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda d: _download_daily_one(d, session), date_list))

    missing = [d.strftime("%Y%m%d") for d, ok in zip(date_list, results) if not ok]
    if missing:
        logging.warning(f"Daily IMERG granules missing for {missing}")
    print(f"Finished downloading daily rainfall data ({sum(results)}/{len(date_list)} dates)")


def download_gpm_for_storm(start_date, end_date, typhoon_name, product=None):
    """Download whichever IMERG product is configured, for one storm's window."""
    product = product or IMERG_PRODUCT
    if product not in IMERG_PRODUCTS:
        raise ValueError(f"Unknown IMERG product {product!r}; expected one of {IMERG_PRODUCTS}")
    if product == "daily":
        return download_gpm_daily_run(start_date, end_date, typhoon_name)
    return download_gpm_late_run(start_date, end_date, typhoon_name)
=======
        for tiff_file in filtered_files:
            file_name = tiff_file.split("/")[-1]
            file_path = download_path / file_name
            
            if not file_path.exists():
                r = requests.get(tiff_file, auth=(USERNAME, PASSWORD))
                time.sleep(0.2)
                with open(file_path, "wb") as f:
                    f.write(r.content)
    logger.info(f"Finished downloading rainfall data for {typhoon_name}")
>>>>>>> 2aaf917cea7caa556c4f871607c174621f1bc43f
