import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from src.config import INPUT_DIR, ISO3_LIST

def generate_global_grid(res=0.1):
    """
    Generates a global 0.1 degree grid of bounding boxes.
    Matches the resolution specified in the paper (approx. 11km at equator).
    """
    print(f"Generating global grid at {res} degree resolution...")
    lon_bins = np.arange(-180, 180, res)
    lat_bins = np.arange(-90, 90, res)
    
    grid_cells = []
    for lon in lon_bins:
        for lat in lat_bins:
            grid_cells.append(box(lon, lat, lon + res, lat + res))
            
    return gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")

def filter_grid_by_land(grid_gdf, shp_path):
    """
    Clips the global grid to landmasses using GADM boundaries.
    Filters by the study's ISO3 list to optimize processing.
    """
    print("Loading GADM boundaries for land-overlap filtering...")
    # Load the global GeoPackage downloaded by general_collector.py
    world = gpd.read_file(shp_path, layer="ADM_0")
    
    # Filter for countries in the study consideration list
    world = world[world['GID_0'].isin(ISO3_LIST)]
    
    print("Performing spatial join (filtering grid to land overlap)...")
    grid_land = gpd.sjoin(grid_gdf, world[['GID_0', 'geometry']], how="inner", predicate="intersects")
    
    # Clean up and add unique IDs
    grid_land = grid_land.drop(columns=["index_right"])
    grid_land = grid_land.rename(columns={"GID_0": "iso3"})
    grid_land["id"] = range(len(grid_land))
    
    return grid_land

def create_centroids(grid_land):
    """
    Generates a point-based version of the grid for centroid-based hazard extraction.
    """
    centroids = grid_land.copy()
    centroids["geometry"] = centroids.geometry.centroid
    centroids["Latitude"] = centroids.geometry.y
    centroids["Longitude"] = centroids.geometry.x
    return centroids

def main_grid_generation():
    """Main execution to create the study's coordinate reference system."""
    shp_path = INPUT_DIR / "SHP" / "gadm_410-gpkg.zip"
    out_dir = INPUT_DIR / "GRID" / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not shp_path.exists():
        print(f"Error: GADM shapefile not found at {shp_path}. Run general_collector.py first.")
        return

    # 1. Generate Global Tessellation
    raw_grid = generate_global_grid(res=0.1)
    
    # 2. Filter to Land Overlap
    land_grid = filter_grid_by_land(raw_grid, shp_path)
    
    # 3. Save Polygon Grid (Used for Zonal Statistics)
    grid_file = out_dir / "global_grid_land_overlap.gpkg"
    land_grid.to_file(grid_file, driver="GPKG")
    print(f"Polygon grid saved to {grid_file}")
    
    # 4. Save Centroid Grid (Used for Wind/Rain/Distance hazards)
    centroid_grid = create_centroids(land_grid)
    centroid_file = out_dir / "global_grid_centroids.csv"
    centroid_grid.to_csv(centroid_file, index=False)
    print(f"Centroid data saved to {centroid_file}")

if __name__ == "__main__":
    main_grid_generation()