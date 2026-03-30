#!/usr/bin/env python3
import os

import requests


def download_pop_data(out_path):
    url = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

        # Write the file to the specified path
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"File downloaded successfully to {out_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download file: {e}")


if __name__ == "__main__":
    # Get data
    out_path = "/data/big/fmoss/data/Worldpop/ppp_2020_1km_Aggregated.tif"
    download_pop_data(out_path)
