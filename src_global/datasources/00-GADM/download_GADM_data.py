#!/usr/bin/env python3
import os
from urllib.parse import urljoin

import geopandas as gpd
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src_global.utils import blob, constant

PROJECT_PREFIX = "global_model"


def download_geodatabase(out_path):
    """
    Downloads a .gpkg.zip file linked as 'geodatabase' from the given URL and saves it to the specified output directory.

    Parameters:
    - url (str): The webpage URL to scrape for the geodatabase link.
    - out_dir (str): The directory where the downloaded file should be stored.

    Returns:
    - str: The path to the downloaded file if successful, otherwise None.
    """
    # GADM site
    url = "https://gadm.org/download_world.html"
    # Fetch the webpage content
    response = requests.get(url)

    if response.status_code != 200:
        print(
            f"Failed to retrieve the URL. Status code: {response.status_code}"
        )
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all anchor tags with href attributes
    links = soup.find_all("a", href=True)

    for link in links:
        print(link.text)
        # Check if the link text matches 'geodatabase' (case-insensitive, stripping whitespace)
        if link.text.strip().lower() == "geodatabase":
            file_url = link["href"]
            print(file_url)
            # Ensure the link is absolute
            if not file_url.startswith("http"):
                file_url = urljoin(url, file_url)

            print(f"Downloading: {file_url}")

            # If out_path is just a filename, use the current directory
            if os.path.dirname(out_path) == "":
                out_path = os.path.join(
                    os.getcwd(), out_path
                )  # Current directory

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Download the file without chunks
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(file_response.content)

                print(f"Download completed: {out_path}")
                return out_path  # Return the downloaded file path

            print(
                f"Failed to download file. Status code: {file_response.status_code}"
            )
            return None

    print("No valid 'geodatabase' link found on the page.")
    return None  # Return None if no file was downloaded


def save_to_blob(local_file_path):
    filename = "/SHP/global_shp.zip"
    blob_name = f"{PROJECT_PREFIX}{filename}"
    with open(local_file_path, "rb") as file:
        data = file.read()
        blob.upload_blob_data(blob_name=blob_name, data=data, prod_dev="dev")


if __name__ == "__main__":
    # Example usage: Its better to have a local copy of this
    # Download in current folder
    out_name = "global_shp.zip"
    download_geodatabase(out_name)
    # Push to blob
    save_to_blob(out_name)
