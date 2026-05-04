import logging
import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import INPUT_DIR

logger = logging.getLogger(__name__)

USERNAME = os.getenv("NASA_PPS_USERNAME")
PASSWORD = os.getenv("NASA_PPS_PASSWORD")

def list_files(url):
    page = requests.get(url, auth=(USERNAME, PASSWORD)).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        url + "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith("tif")
    ]

def download_gpm_late_run(start_date, end_date, typhoon_name):
    if not USERNAME or not PASSWORD:
        raise ValueError("NASA PPS credentials not set in environment variables.")
        
    base_url = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/"
    date_list = pd.date_range(start_date, end_date)
    
    download_path = INPUT_DIR / "gpm_data" / typhoon_name
    download_path.mkdir(parents=True, exist_ok=True)

    for date in date_list:
        day = f"{date.day:02d}"
        month = f"{date.month:02d}"
        year = date.year
        
        url = f"{base_url}{year}/{month}"
        tiff_files = list_files(url=url)
        
        prefix = f'{url}/3B-HHR-L.MS.MRG.3IMERG.{year}{month}{day}'
        filtered_files = [f for f in tiff_files if f.startswith(prefix) and f.endswith('.30min.tif')]

        for tiff_file in filtered_files:
            file_name = tiff_file.split("/")[-1]
            file_path = download_path / file_name
            
            if not file_path.exists():
                r = requests.get(tiff_file, auth=(USERNAME, PASSWORD))
                time.sleep(0.2)
                with open(file_path, "wb") as f:
                    f.write(r.content)
    logger.info(f"Finished downloading rainfall data for {typhoon_name}")
