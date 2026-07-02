import io
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import pandas as pd
from src.config import INPUT_DIR, NASA_PPS_BASE_URL

USERNAME = os.getenv("NASA_PPS_USERNAME")
PASSWORD = os.getenv("NASA_PPS_PASSWORD")

MAX_RETRIES = 3

def list_files(url, session):
    page = session.get(url, auth=(USERNAME, PASSWORD)).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith("zip")
    ]

def _download_one(zip_url, download_path, session):
    zip_name = zip_url.split("/")[-1]
    file_name = zip_name.replace(".zip", ".total.accum.tif")
    file_path = download_path / file_name

    if file_path.exists():
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(zip_url, auth=(USERNAME, PASSWORD), timeout=120)
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                member = next((m for m in zf.namelist() if m.endswith("total.accum.tif")), None)
                if member:
                    with zf.open(member) as src, open(file_path, "wb") as dst:
                        dst.write(src.read())
            return
        except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
            if attempt == MAX_RETRIES:
                logging.error(f"Giving up on {zip_url} after {MAX_RETRIES} attempts: {e}")
            else:
                time.sleep(2 ** attempt)

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

            url = f"{NASA_PPS_BASE_URL}{year}/{month}/{day}/gis"
            zip_files = list_files(url=url, session=session)

            prefix = f'{url}/3B-HHR-GIS.MS.MRG.3IMERG.{year}{month}{day}'
            filtered_files = [f for f in zip_files if f.startswith(prefix)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(
                    lambda zip_url: _download_one(zip_url, download_path, session),
                    filtered_files,
                ))
    print(f"Finished downloading rainfall data for {typhoon_name}")
