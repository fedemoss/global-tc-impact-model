#!/usr/bin/env python3
import os

import requests


def download_tif(url, out_path):
    """
    Downloads a TIFF file from a given URL and saves it to the specified path.

    Parameters:
    - url (str): The URL of the TIFF file.
    - out_path (str): The local path where the file should be saved.

    Returns:
    - bool: True if download was successful, False otherwise.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Send GET request
        response = requests.get(
            url, stream=True, timeout=30
        )  # Use streaming to handle large files
        response.raise_for_status()  # Raise error for HTTP issues (4xx, 5xx)

        # Write content to file in chunks
        with open(out_path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=8192
            ):  # 8KB per chunk
                file.write(chunk)

        print(f"Download successful: {out_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


if __name__ == "__main__":
    # Set input
    url = "https://datacatalogfiles.worldbank.org/ddh-published/0037584/DR0045419/LS_RF_Median_1980-2018_COG.tif?versionId=2023-01-18T20:42:41.4307260Z"
    out_path = "/data/big/fmoss/data/LandSlides/landslide_data.tif"
    # Download tif file
    download_tif(url, out_path)
