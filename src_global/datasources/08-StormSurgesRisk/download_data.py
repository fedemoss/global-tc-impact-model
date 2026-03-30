#!/usr/bin/env python3
import os

import pandas as pd
import requests


# Download data
def download_file(url, out_dir):
    # Ensure the output directory exists, create it if necessary
    os.makedirs(out_dir, exist_ok=True)

    # Send the GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Define the full path to save the file
        file_path = os.path.join(out_dir, "storm_surges_data.nc")

        # Save the content to the file
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded successfully to {file_path}.")
    else:
        print(
            f"Failed to download the file. Status code: {response.status_code}"
        )


if __main__ == "__main__":
    # Download data
    url = "https://data.4tu.nl/file/4e291b8f-a37e-4378-8ca6-954a44fdc8fb/1263247c-4427-40eb-b497-a79f72caa267"
    out_dir = "/data/big/fmoss/data/StormSurges"
    download_file(url, out_dir)
