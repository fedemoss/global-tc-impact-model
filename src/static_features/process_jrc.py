import os
import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from concurrent.futures import ProcessPoolExecutor
from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

def adjust_longitude(polygon):
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)

def adjust_data(grid, src):
    grid_transformed = grid.copy()
    grid_transformed["geometry"] = grid_transformed["geometry"].apply(adjust_longitude)
    grid_transformed = grid_transformed[["id", "iso3", "Latitude", "Longitude", "geometry"]]
    src_wgs84 = src.rio.reproject(grid_transformed.crs)
    return grid_transformed, src_wgs84

def calculate_urban_rural_water(grid, raster, out_dir, iso3, nodata_value=128):
    file_path = out_dir / f"degree_of_urbanization_{iso3}.csv"
    if file_path.exists():
        print(f"File already exists for {iso3}, skipping.")
        return

    smod_raster_wgs84_clip = raster.rio.clip_box(*grid.total_bounds)
    smod_raster_wgs84_clip = smod_raster_wgs84_clip.where(smod_raster_wgs84_clip != nodata_value)

    stats = zonal_stats(
        grid["geometry"], smod_raster_wgs84_clip.values[0],
        affine=smod_raster_wgs84_clip.rio.transform(),
        stats="count", categorical=True
    )

    smod_grid_vals = pd.DataFrame(stats).fillna(0)
    row_sums = smod_grid_vals.sum(axis=1)

    smod_grid_vals["urban"] = (smod_grid_vals.get(21, 0) + smod_grid_vals.get(22, 0) + smod_grid_vals.get(23, 0)) / row_sums
    smod_grid_vals["rural"] = (smod_grid_vals.get(11, 0) + smod_grid_vals.get(12, 0) + smod_grid_vals.get(13, 0)) / row_sums
    smod_grid_vals["water"] = smod_grid_vals.get(10, 0) / row_sums

    smod_grid_vals["id"] = grid["id"].values
    df_urban_rural = smod_grid_vals[["id", "urban", "rural", "water"]]
    df_urban_rural.to_csv(file_path, index=False)
    print(f"Processed {iso3} and saved.")

if __name__ == "__main__":
    file_name = "GHS_SMOD_P2025_GLOBE_R2022A_54009_1000_V1_0.tif"
    src = rxr.open_rasterio(INPUT_DIR / "JRC" / file_name)

    global_grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    global_grid["iso3"] = global_grid["GID_0"]

    grid_transformed, src_wgs84 = adjust_data(global_grid, src)

    out_dir = OUTPUT_DIR / "JRC" / "grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    valid_iso3_list = [iso for iso in ISO3_LIST if iso in grid_transformed.iso3.unique()]

    for iso3 in valid_iso3_list:
        calculate_urban_rural_water(grid_transformed[grid_transformed.iso3 == iso3], src_wgs84, out_dir, iso3)