#!/usr/bin/env python3
import io
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Polygon
from tqdm import tqdm


def get_tile_names():
    """
    Generates a list of strings in the format 'srtm_xx_yy.zip'
    where xx ranges from 01 to 72 and yy ranges from 01 to 24.
    """
    return [
        f"srtm_{xx:02d}_{yy:02d}.zip"
        for xx in range(1, 73)  # xx ranges from 01 to 72
        for yy in range(1, 25)  # yy ranges from 01 to 24
    ]


def download_and_extract_single_tif(file, base_url, local_path=None):
    """
    Downloads and extracts .tif files from a single file URL.
    """
    try:
        req = requests.get(base_url + file, verify=True, stream=True)
        req.raise_for_status()  # Raise an error for HTTP issues

        with zipfile.ZipFile(io.BytesIO(req.content)) as zObj:
            fileNames = zObj.namelist()
            for fileName in fileNames:
                if fileName.endswith(".tif"):
                    output_file_path = os.path.join(local_path, fileName)

                    # Skip downloading if file already exists
                    if os.path.exists(output_file_path):
                        return None

                    # Ensure the directory exists
                    os.makedirs(
                        os.path.dirname(output_file_path), exist_ok=True
                    )

                    # Write the file to the custom path
                    with open(output_file_path, "wb") as output_file:
                        output_file.write(zObj.open(fileName).read())
    except requests.exceptions.RequestException as e:
        return None
    except zipfile.BadZipFile as e:
        return f"Failed to extract {file}: {e}"
    return None  # No errors


def download_and_extract_tifs_parallel(
    file_list, base_url, local_path=None, max_workers=4
):
    """
    Parallelizes the download and extraction of .tif files from a list of file URLs,
    with a progress bar. Skips files that have already been downloaded.

    Parameters:
        file_list (list): List of files to download.
        base_url (str): Base URL to prefix the file paths.
        local_path (str, optional): Local directory to save the files.
        max_workers (int): Number of threads to use for parallel processing.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = {
            executor.submit(
                download_and_extract_single_tif, file, base_url, local_path
            ): file
            for file in file_list
        }
        # Use tqdm for the progress bar
        with tqdm(
            total=len(file_list), desc="Processing Files", unit="file"
        ) as pbar:
            for future in as_completed(futures):
                result = future.result()  # Get the result or exception
                if result:  # Log only failures
                    print(result)
                pbar.update(1)  # Update the progress bar


if __name__ == "__main__":
    # Set input
    base_url = (
        "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"
    )
    local_path = "/data/big/fmoss/data/SRTM/tiles"
    # Download tif files
    global_tiles = get_tile_names()
    download_and_extract_tifs_parallel(
        global_tiles, base_url, local_path, max_workers=16
    )
