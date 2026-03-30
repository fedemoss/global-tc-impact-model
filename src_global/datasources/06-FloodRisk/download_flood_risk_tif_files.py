#!/usr/bin/env python3
import os
import re
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from bs4 import BeautifulSoup
from rasterio.io import MemoryFile
from rasterio.merge import merge
from shapely.geometry import box


def download_tif_files(
    url, download_folder="./tif_files_floods", adjusted=False
):
    """
    Downloads TIF files from the provided URL to the specified folder.

    Args:
        url (str): The URL of the directory containing the TIF files.
        download_folder (str): The folder where the files will be saved. Default is './tif_files_floods'.

    Returns:
        list: A list of file paths to the downloaded TIF files.
    """
    # Create the folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)

    # Fetch the page content
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links in the page
    links = soup.find_all("a")

    # Filter for .tif files ending with '_depth.tif'
    if adjusted:
        tif_files = [
            link.get("href")
            for link in links
            if link.get("href").endswith("_depth_reclass.tif")
        ]
    else:
        tif_files = [
            link.get("href")
            for link in links
            if link.get("href").endswith("_depth.tif")
        ]

    # List to hold paths of downloaded files
    tif_paths = []

    # Download each matching .tif file
    for tif_file in tif_files:
        file_url = url + tif_file
        local_file = os.path.join(download_folder, tif_file)

        # Download and save the file
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for HTTP errors
            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {tif_file}")

        # Add the file path to the list
        tif_paths.append(local_file)

    return tif_paths


if __name__ == "__main__":
    # URL of the directory listing
    url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/RP10/"
    # Download categorical data
    data_folder = "/data/big/fmoss/data/FloodRisk/tiles"
    os.makedirs(data_folder, exist_ok=True)
    tif_paths = download_tif_files(
        url, download_folder=data_folder, adjusted=True
    )
