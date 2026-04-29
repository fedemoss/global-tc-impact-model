import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import country_converter as coco
from src.config import INPUT_DIR, OUTPUT_DIR



# -------------------------------------------------------------------
# Coastline Distance and Length
# -------------------------------------------------------------------
def calculate_distance_to_coast(grid_path, coastline_path):
    """
    Calculates the minimum distance from each grid cell centroid to the coastline.
    """
    print("Loading grid and coastline shapes...")
    grid = gpd.read_file(grid_path)
    coastline = gpd.read_file(coastline_path)
    
    # Ensure consistent CRS (Projected CRS recommended for distance calculations, e.g., EPSG:3857)
    # Using EPSG:4326 for initial load, but distance in meters requires projection
    grid_proj = grid.to_crs(epsg=3857)
    coastline_proj = coastline.to_crs(epsg=3857)
    
    # Calculate centroids
    grid_proj['centroid'] = grid_proj.geometry.centroid
    
    print("Calculating distances to coastline...")
    # Calculate distance from each centroid to the nearest coastline geometry
    # Note: For large datasets, spatial indexing (sindex) should be used.
    # geopandas .distance calculates shortest distance to the geometry.
    distances = grid_proj.set_geometry('centroid').geometry.apply(
        lambda x: coastline_proj.distance(x).min()
    )
    
    grid['distance_to_coast_meters'] = distances
    
    # Binary feature: True if distance is 0 (or very close, implying it touches the coast)
    # We use a small buffer (e.g., 100 meters) to account for projection inaccuracies
    grid['with_coast'] = grid['distance_to_coast_meters'] <= 100
    
    return grid[['id', 'iso3', 'distance_to_coast_meters', 'with_coast']]

def calculate_coastline_length(grid_path, coastline_path):
    """
    Calculates the length of the coastline contained within each grid cell.
    """
    print("Loading grid and coastline shapes for length calculation...")
    grid = gpd.read_file(grid_path)
    coastline = gpd.read_file(coastline_path)
    
    grid_proj = grid.to_crs(epsg=3857)
    coastline_proj = coastline.to_crs(epsg=3857)
    
    print("Intersecting grid with coastlines...")
    # Perform intersection
    intersection = gpd.overlay(coastline_proj, grid_proj, how='intersection')
    
    print("Calculating lengths...")
    # Calculate length of the intersected geometries
    intersection['coast_length_meters'] = intersection.geometry.length
    
    # Aggregate lengths per grid cell
    length_agg = intersection.groupby('id')['coast_length_meters'].sum().reset_index()
    
    # Merge back to the main grid id list, filling NaNs with 0
    final_lengths = grid[['id', 'iso3']].merge(length_agg, on='id', how='left')
    final_lengths['coast_length_meters'] = final_lengths['coast_length_meters'].fillna(0)
    
    return final_lengths

def process_all_coastal():
    """Entry point to execute coastal feature generation."""
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    coastline_path = INPUT_DIR / "SHP" / "ne_10m_coastline" / "ne_10m_coastline.shp"
    
    out_dir = OUTPUT_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dist_out = out_dir / "distance_to_coast.csv"
    if not dist_out.exists():
        dist_df = calculate_distance_to_coast(grid_path, coastline_path)
        dist_df.to_csv(dist_out, index=False)
        print(f"Saved distances to {dist_out}")
        
    len_out = out_dir / "coastline_length.csv"
    if not len_out.exists():
        len_df = calculate_coastline_length(grid_path, coastline_path)
        len_df.to_csv(len_out, index=False)
        print(f"Saved lengths to {len_out}")

if __name__ == "__main__":
    process_all_coastal()