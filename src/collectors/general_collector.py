import os
import io
import zipfile
import requests
import concurrent.futures
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import (
    INPUT_DIR, GADM_BASE_URL, WORLDPOP_URL, LANDSLIDE_URL, 
    STORM_SURGE_URL, JRC_SMOD_URL, SRTM_BASE_URL, FLOOD_RISK_URL,
    SHDI_URL, GAUL_ADM2_URL
)

from src.utils.region_matching import create_basin_dataset

def download_file(url, out_path, stream=True, verify=True):
    """Generic download utility for large files and streaming."""
    if out_path.exists():
        return True
    
    os.makedirs(out_path.parent, exist_ok=True)
    try:
        response = requests.get(url, stream=stream, verify=verify, timeout=60)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        return False

def collect_gadm():
    """Scrapes gadm.org for the latest global geodatabase and downloads it."""
    print("Scraping GADM for geodatabase link...")
    out_path = INPUT_DIR / "SHP" / "gadm_410-gpkg.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(GADM_BASE_URL)
    if response.status_code != 200:
        return
    
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a", href=True)
    
    file_url = None
    for link in links:
        if link.text.strip().lower() == "geodatabase":
            file_url = link["href"]
            if not file_url.startswith("http"):
                file_url = urljoin(GADM_BASE_URL, file_url)
            break

    if file_url:
        print(f"Downloading GADM: {file_url}")
        file_response = requests.get(file_url)
        with open(out_path, "wb") as f:
            f.write(file_response.content)
    else:
        print("No geodatabase link found on GADM page.")

def collect_guil():
    """
    Downloads the GAUL 2015 ADM2 shapefile required for 
    geolocating EM-DAT administrative units.
    """
    print("Collecting GAUL/GUIL ADM2 administrative boundaries...")
    out_path = INPUT_DIR / "SHP" / "global_shapefile_GUIL_adm2.gpkg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Note: FAO GAUL usually requires a direct file request or 
    # download from a mirror like HDX if the API is restricted.
    # This logic assumes a direct download link is provided in config.
    try:
        response = requests.get(GAUL_ADM2_URL, stream=True)
        if response.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(response.content)
            print(f"GAUL data saved to {out_path}")
        else:
            print(f"Failed to download GAUL. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error collecting GAUL data: {e}")

def collect_worldpop():
    """Downloads the mosaicked 1km global population TIFF."""
    out_path = INPUT_DIR / "Worldpop" / "ppp_2020_1km_Aggregated.tif"
    download_file(WORLDPOP_URL, out_path)

def collect_landslide():
    """Downloads the rainfall-triggered landslide hazard map."""
    out_path = INPUT_DIR / "LandSlides" / "landslide_data.tif"
    download_file(LANDSLIDE_URL, out_path)

def collect_storm_surge():
    """Downloads COAST-RPv2 return period data."""
    out_path = INPUT_DIR / "StormSurges" / "storm_surges_data.nc"
    download_file(STORM_SURGE_URL, out_path)

def collect_flood_risk():
    """Scrapes and downloads GLOFAS RP10 flood depth reclassified tiles."""
    print("Collecting Flood Risk Data...")
    out_dir = INPUT_DIR / "FloodRisk" / "tiles"
    os.makedirs(out_dir, exist_ok=True)
    
    res = requests.get(FLOOD_RISK_URL)
    soup = BeautifulSoup(res.text, "html.parser")
    # Filter for reclassified depth files as used in the paper pipeline
    tif_files = [link.get("href") for link in soup.find_all("a") 
                 if link.get("href").endswith("_depth_reclass.tif")]
    
    for tif in tif_files:
        download_file(urljoin(FLOOD_RISK_URL, tif), out_dir / tif)

def collect_jrc():
    """Downloads and extracts the JRC SMOD urbanization dataset."""
    out_dir = INPUT_DIR / "JRC"
    os.makedirs(out_dir, exist_ok=True)
    
    response = requests.get(JRC_SMOD_URL, verify=False, stream=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file in z.namelist():
            if file.endswith(".tif"):
                z.extract(file, out_dir)

def collect_shdi():
    """Checks for SHDI data; provides instructions if missing."""
    print("Checking for Subnational HDI Data...")
    out_path = INPUT_DIR / "SHDI" / "SHDI_Complete_v10.csv" # Update version as needed
    if out_path.exists():
        print("SHDI data found.")
    else:
        print(f"SHDI data missing at {out_path}. Please download 'SHDI Complete' from {SHDI_URL}")

def collect_srtm():
    """Downloads and extracts SRTM 5x5 tiles in parallel."""
    local_path = INPUT_DIR / "SRTM" / "tiles"
    tile_names = [f"srtm_{x:02d}_{y:02d}.zip" for x in range(1, 73) for y in range(1, 25)]
    
    def download_and_extract(tile):
        try:
            url = SRTM_BASE_URL + tile
            res = requests.get(url, stream=True, timeout=30)
            res.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                for name in z.namelist():
                    if name.endswith(".tif") and not (local_path / name).exists():
                        z.extract(name, local_path)
        except:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(download_and_extract, tile_names), total=len(tile_names)))

def create_region_dataset():
    df = create_basin_dataset()
    out_dir = INPUT_DIR / "model_input_dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "un_regions.csv"
    df.to_csv(out_path, index=False)


def download_all_public_data():
    """Main orchestrator for collecting all public hazard and spatial data."""
    collect_gadm()
    collect_guil()
    collect_worldpop()
    collect_landslide()
    collect_storm_surge()
    collect_jrc()
    collect_srtm()
    collect_flood_risk()
    collect_shdi()
    # Create also a dataset with region information
    create_region_dataset()

if __name__ == "__main__":
    download_all_public_data()
    