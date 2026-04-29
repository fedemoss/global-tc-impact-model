import os
import logging
import subprocess
import tempfile
import concurrent.futures
from multiprocessing import Pool

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import Polygon
from sklearn.neighbors import NearestNeighbors

from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def adjust_longitude(polygon):
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)

def is_border_crossing(polygon):
    coords = list(polygon.exterior.coords)
    has_pos, has_neg = False, False
    for lon, lat in coords:
        if lon > 0: has_pos = True
        if lon < 0: has_neg = True
        if has_pos and has_neg:
            return True
    return False

def get_overlap_files(extent):
    tif_files = [f"srtm_{xx:02d}_{yy:02d}.tif" for xx in range(1, 73) for yy in range(1, 25)]
    minx, miny, maxx, maxy = extent
    lon_step, lat_step = 360 / 72, 120 / 24

    overlapping_files = []
    for file in tif_files:
        parts = file.replace("srtm_", "").replace(".tif", "").split("_")
        xx, yy = int(parts[0]) - 1, int(parts[1]) - 1

        cell_minx = -180 + xx * lon_step
        cell_maxx = cell_minx + lon_step
        cell_maxy = 60 - yy * lat_step
        cell_miny = cell_maxy - lat_step

        if not (cell_maxx <= minx or cell_minx >= maxx or cell_maxy <= miny or cell_miny >= maxy):
            overlapping_files.append(file)
            
    return overlapping_files

def get_country_tiles(grid):
    country_tiles = []
    for iso in grid.iso3.unique():
        if iso == "USA":
            tiles = get_overlap_files([172, 18.9, 180, 72.7]) + get_overlap_files([-180, 18.9, -67, 72.7])
        elif iso == "FJI":
            tiles = get_overlap_files([176.8, -21.1, 180, -12.4]) + get_overlap_files([-180, -21.1, -178, -12.4])
        elif iso == "NZL":
            tiles = get_overlap_files([165.8, -52.7, 180, -29.2]) + get_overlap_files([-180, -52.7, -176, -29.2])
        elif iso == "KIR":
            continue
        elif iso == "RUS":
            tiles = get_overlap_files([25, 41.18886566, 180, 81.856247]) + get_overlap_files([-180, 41.18886566, -170, 81.856247])
        else:
            extent = grid[grid.iso3 == iso].total_bounds
            tiles = get_overlap_files(extent)

        country_tiles.append({"iso3": iso, "tiles": tiles})
    return pd.DataFrame(country_tiles)

def get_mosaic_raster_paths(row, data_path):
    def process_tile(tile):
        tile_path = data_path / tile
        return tile_path if tile_path.exists() else None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_tile, row["tiles"])
    return [p for p in results if p is not None]

def calculate_zonal_stats(grid, raster_array, stats, nodata, affine):
    summary_stats = zonal_stats(grid, raster_array, stats=stats, nodata=nodata, all_touched=True, affine=affine)
    return pd.DataFrame(summary_stats)

def process_single_raster(grid, raster_path):
    grid = grid.copy()
    grid["border"] = grid["geometry"].apply(is_border_crossing)
    grid = grid[~grid.border]

    def process_altitude_task(r_path):
        with rasterio.open(r_path) as src:
            df = calculate_zonal_stats(grid, src.read(1), ["mean"], -32768, src.transform)
            df["id"] = grid["id"].values
            return df

    def process_gdaldem_task(r_path, mode, nodata):
        out_path = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        subprocess.run(
            ["gdaldem", mode, "-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=9", str(r_path), out_path, "-compute_edges"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        with rasterio.open(out_path) as src:
            df = calculate_zonal_stats(grid, src.read(1), ["mean", "std"], nodata, src.transform)
            df["id"] = grid["id"].values
            return df, out_path

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            alt_future = executor.submit(process_altitude_task, raster_path)
            slope_future = executor.submit(process_gdaldem_task, raster_path, "slope", -9999)
            rug_future = executor.submit(process_gdaldem_task, raster_path, "TRI", -9999)

            altitude_df = alt_future.result()
            slope_df, slope_path = slope_future.result()
            ruggedness_df, tri_path = rug_future.result()
    finally:
        for temp_file in [slope_path, tri_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return altitude_df, slope_df, ruggedness_df

def main_processing_pipeline(grid, mosaic_raster_paths, num_workers=4):
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_raster, [(grid, path) for path in mosaic_raster_paths])

    altitude_df = pd.concat([res[0].dropna() for res in results], ignore_index=True).merge(grid, how="right")
    slope_df = pd.concat([res[1].dropna() for res in results], ignore_index=True).merge(grid, how="right")
    ruggedness_df = pd.concat([res[2].dropna() for res in results], ignore_index=True).merge(grid, how="right")

    return altitude_df, slope_df, ruggedness_df

def spatial_interpolation(df, columns_to_interpolate):
    df = df.copy()
    df["centroid"] = df.geometry.centroid
    coordinates = np.array([[point.x, point.y] for point in df.centroid])

    for column in columns_to_interpolate:
        if df[column].isnull().any():
            missing_mask = df[column].isna().values
            non_missing_mask = ~missing_mask
            
            if not non_missing_mask.any(): continue
            
            known_coords, known_vals = coordinates[non_missing_mask], df.loc[non_missing_mask, column].values
            missing_coords = coordinates[missing_mask]

            nbrs = NearestNeighbors(n_neighbors=min(8, len(known_coords)), algorithm="auto").fit(known_coords)
            distances, indices = nbrs.kneighbors(missing_coords)

            weights = 1 / (distances + 1e-10)
            weights /= weights.sum(axis=1, keepdims=True)

            df.loc[missing_mask, column] = (weights * known_vals[indices]).sum(axis=1)

    return df.drop(columns=["centroid"])

def data_extraction(iso, grid, country_tiles, out_path, data_path):
    output_path = out_path / f"srtm_grid_data_{iso}.csv"
    if output_path.exists():
        return pd.read_csv(output_path)

    try:
        grid_country = grid[grid.iso3 == iso]
        mosaic_raster_paths = get_mosaic_raster_paths(country_tiles[country_tiles.iso3 == iso].iloc[0], data_path)

        if not mosaic_raster_paths:
            df_terrain = grid_country[["id"]].copy()
            df_terrain["mean_elev"] = df_terrain["mean_slope"] = df_terrain["mean_rug"] = np.nan
        else:
            alt_df, slope_df, rug_df = main_processing_pipeline(grid_country, mosaic_raster_paths, num_workers=4)
            df_terrain = slope_df.rename(columns={"mean": "mean_slope"})[["id", "mean_slope"]]
            df_terrain = df_terrain.merge(alt_df.rename(columns={"mean": "mean_elev"})[["id", "mean_elev"]], on="id")
            df_terrain = df_terrain.merge(rug_df.rename(columns={"mean": "mean_rug"})[["id", "mean_rug"]], on="id")
            
            df_terrain = grid_country[["id", "geometry"]].merge(df_terrain, on="id", how="left")
            df_terrain = spatial_interpolation(df_terrain, ["mean_elev", "mean_slope", "mean_rug"])

        df_terrain["iso3"] = iso
        
        # Drop geometry before saving
        final_df = df_terrain.drop(columns=["geometry"], errors="ignore")
        final_df.to_csv(output_path, index=False)
        print(f"Successfully processed SRTM for {iso}")
        return final_df

    except Exception as e:
        logging.error(f"Couldn't calculate terrain data for {iso}. Error: {e}")
        return None

def iterate_srtm_data_extracting(grid, out_path, data_path, valid_iso_list):
    country_tiles = get_country_tiles(grid)
    srtm_data = pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(data_extraction, iso, grid, country_tiles, out_path, data_path) for iso in valid_iso_list]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                srtm_data = pd.concat([srtm_data, res], ignore_index=True)

    return srtm_data

def process_all_srtm():
    out_path = OUTPUT_DIR / "SRTM" / "grid_data"
    data_path = INPUT_DIR / "SRTM" / "tiles"
    out_path.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid["geometry"] = grid["geometry"].apply(adjust_longitude)
    grid["iso3"] = grid["GID_0"]

    valid_iso_list = [iso for iso in ISO3_LIST if iso in grid.iso3.unique()]

    print(f"Processing SRTM data for {len(valid_iso_list)} countries...")
    srtm_data = iterate_srtm_data_extracting(grid, out_path, data_path, valid_iso_list)
    srtm_data.to_csv(out_path / "global_srtm_grid_data.csv", index=False)

if __name__ == "__main__":
    process_all_srtm()
