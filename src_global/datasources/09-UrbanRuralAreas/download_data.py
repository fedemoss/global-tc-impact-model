#!/usr/bin/env python3
import io
import os
import zipfile

import requests


def download_and_extract_smod(out_dir):
    """
    Downloads and extracts the GHS-SMOD dataset from JRC to the specified directory.

    Parameters:
    - out_dir (str): Path to the output directory where files will be saved.
    """
    os.makedirs(out_dir, exist_ok=True)  # Ensure the directory exists

    smod_link = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2022A/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000/V1-0/GHS_SMOD_P2025_GLOBE_R2022A_54009_1000_V1_0.zip"

    print(f"Downloading SMOD dataset from {smod_link}...")
    req = requests.get(smod_link, verify=False, stream=True)

    if req.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(req.content)) as zObj:
            fileNames = zObj.namelist()
            for fileName in fileNames:
                if fileName.endswith(".tif"):
                    print(f"Extracting {fileName}...")
                    content = zObj.open(fileName).read()
                    with open(os.path.join(out_dir, fileName), "wb") as f:
                        f.write(content)
        print(f"Download and extraction completed. Files saved to {out_dir}")
    else:
        print(
            f"Failed to download the dataset. HTTP Status Code: {req.status_code}"
        )


if __name__ == "__main__":
    out_dir = "/data/big/fmoss/data/JRC"
    # Download data
    download_and_extract_smod(out_dir)
