import logging
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon

# Configure logging
logging.basicConfig(
    filename="process_worldpop_data.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define a function to adjust the longitude of a single polygon
# (0, 360) --> (-180, 180)
def adjust_longitude(polygon):
    # Extract the coordinates of the polygon
    coords = list(polygon.exterior.coords)
    # Adjust longitudes from [-180, 180) to [-360, 0)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    # Create a new Polygon with adjusted coordinates
    return Polygon(coords)


# Function to calculate the sum of population for a given geometry
def calculate_population(geometry, raster, transform):
    # Mask the raster with the geometry
    out_image, _ = mask(raster, [geometry], crop=True)
    out_image = out_image[0]  # Get the single band

    # Calculate the sum of the population within the geometry
    population_sum = out_image[
        out_image > 0
    ].sum()  # Sum only positive values (valid population counts)
    return population_sum


# Load and create population dataset for each country at grid level
def pop_to_grid(grid, raster):
    # Reproject geometries to match the raster CRS
    grid = grid.to_crs(raster.crs)

    # Calculate population for each geometry in adm2
    grid["population"] = grid["geometry"].apply(
        calculate_population, args=(raster, raster.transform)
    )

    # Dataset
    pop_data_country = grid[["id", "iso3", "population"]]
    return pop_data_country


def iterate_pop_to_grid(grid, raster, out_folder, max_workers=4):
    # Ensure the directory exists
    os.makedirs(out_folder, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        filename="process_tif_files.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    def process_country(iso):
        try:
            output_path = f"{out_folder}/population_grid_{iso}.csv"
            if os.path.exists(output_path):
                return f"File already exists for {iso}, skipping."

            # Get data for the country
            grid_country = grid[grid.iso3 == iso].reset_index(drop=True)
            pop_country = pop_to_grid(grid_country, raster)

            # Save data
            pop_country.to_csv(output_path, index=False)
            return f"Processed {iso} successfully."
        except Exception as e:
            logging.error(f"Error processing {iso}: {e}")
            return f"Error processing {iso}, logged."

    # Unique country codes
    iso3_list = grid.iso3.unique()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_country, iso3_list))


if __name__ == "__main__":
    # Load Population raster
    raster_path = "/data/big/fmoss/data/Worldpop/ppp_2020_1km_Aggregated.tif"
    pop_raster = rasterio.open(raster_path)

    # Load grid cells
    grid = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )
    grid["iso3"] = grid["GID_0"]
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)

    # Output
    out_folder = "/data/big/fmoss/data/Worldpop/grid_data"

    # Population data to grid level
    iterate_pop_to_grid(grid, pop_raster, out_folder)
