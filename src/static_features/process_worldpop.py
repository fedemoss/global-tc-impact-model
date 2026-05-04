import os
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

import logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def adjust_longitude(polygon):
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)

def calculate_population(geometry, raster, transform):
    out_image, _ = mask(raster, [geometry], crop=True)
    out_image = out_image[0]
    return out_image[out_image > 0].sum()

def pop_to_grid(grid, raster):
    grid = grid.to_crs(raster.crs)
    grid["population"] = grid["geometry"].apply(calculate_population, args=(raster, raster.transform))
    return grid[["id", "iso3", "population"]]

def process_country(iso, grid, raster_path, out_folder):
    try:
        output_path = out_folder / f"population_grid_{iso}.csv"
        if output_path.exists():
            return f"File already exists for {iso}, skipping."

        grid_country = grid[grid.iso3 == iso].reset_index(drop=True)
        
        with rasterio.open(raster_path) as raster:
            pop_country = pop_to_grid(grid_country, raster)
            
        pop_country.to_csv(output_path, index=False)
        return f"Processed {iso} successfully."
    except Exception as e:
        logging.error(f"Error processing {iso}: {e}")
        return f"Error processing {iso}."
    
def process_all_worldpop():
    raster_path = INPUT_DIR / "Worldpop" / "ppp_2020_1km_Aggregated.tif"
    out_folder = OUTPUT_DIR / "Worldpop" / "grid_data"
    out_folder.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["GID_0"] = grid["iso3"] 
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)

    valid_iso3_list = [iso for iso in ISO3_LIST if iso in grid.iso3.unique()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_country, iso, grid, raster_path, out_folder) for iso in valid_iso3_list]
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    process_all_worldpop()