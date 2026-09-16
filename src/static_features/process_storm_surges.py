import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors

from src.config import INPUT_DIR, OUTPUT_DIR
from src.utils.geo_utils import adjust_longitude

logger = logging.getLogger(__name__)


def create_geodataframe_from_nc(dataset):
    lon = dataset["station_x_coordinate"].values
    lat = dataset["station_y_coordinate"].values
    storm_tide_10 = dataset["storm_tide_rp_0010"].values

    points = [Point(lon[i], lat[i]) for i in range(len(lon))]
    return gpd.GeoDataFrame({"storm_tide_rp_0010": storm_tide_10}, geometry=points, crs="EPSG:4326")

def spatial_interpolation(df, columns_to_interpolate):
    df = df.copy()
    df["centroid"] = df.geometry.centroid
    coordinates = np.array([[point.x, point.y] for point in df.centroid])

    for column in columns_to_interpolate:
        if df[column].isnull().any():
            missing_mask = df[column].isna().values
            non_missing_mask = ~missing_mask

            known_values = df.loc[non_missing_mask, column].values
            known_coords = coordinates[non_missing_mask]
            missing_coords = coordinates[missing_mask]

            nbrs = NearestNeighbors(n_neighbors=min(8, len(known_coords)), algorithm="auto").fit(known_coords)
            distances, indices = nbrs.kneighbors(missing_coords)

            weights = 1 / (distances + 1e-10)
            weights /= weights.sum(axis=1, keepdims=True)

            interpolated_values = (weights * known_values[indices]).sum(axis=1)
            df.loc[missing_mask, column] = interpolated_values

    df.drop(columns=["centroid"], inplace=True)
    return df

def process_storm_surge_risk(gdf, gdf_coastline, grid_cells, output_csv, R=0.5):
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(R)

    intersected_gdf = gpd.sjoin(gdf_buffered, gdf_coastline, how="right", predicate="intersects").drop(columns="index_left")
    intersected_gdf = intersected_gdf.groupby(["id", "GID_0", "geometry"]).agg({"storm_tide_rp_0010": "max"}).reset_index()

    intersected_gdf_copy = gpd.GeoDataFrame(intersected_gdf, geometry="geometry")
    intersected_gdf_copy["valid"] = intersected_gdf_copy["storm_tide_rp_0010"].notna()

    interpolated_gdf = []
    for country, group in intersected_gdf_copy.groupby("GID_0"):
        if group["valid"].all():
            interpolated_gdf.append(group)
        elif not group["valid"].any():
            continue
        else:
            group = spatial_interpolation(group, ["storm_tide_rp_0010"])
            interpolated_gdf.append(group)

    if interpolated_gdf:
        interpolated_gdf = pd.concat(interpolated_gdf).reset_index(drop=True)
        storm_surges_complete_df = interpolated_gdf[["id", "GID_0", "storm_tide_rp_0010"]].merge(grid_cells[["id", "GID_0"]], how="right")
        storm_surges_complete_df.fillna(0, inplace=True)
    else:
        storm_surges_complete_df = grid_cells[["id", "GID_0"]].copy()
        storm_surges_complete_df["storm_tide_rp_0010"] = 0

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    storm_surges_complete_df.to_csv(output_csv, index=False)
    logger.info(f"Storm surge risk data saved to: {output_csv}")



def process_all_surges():
    """
    Processes global storm surge risk by mapping surge data to coastal grid cells.
    Coastal identification is performed dynamically using the processed SRTM 
    terrain datasets for each country.
    """
    # 1. Load Storm Surge source data
    surge_nc_path = INPUT_DIR / "StormSurges" / "storm_surges_data.nc"
    dataset = xr.open_dataset(surge_nc_path)
    gdf_surges = create_geodataframe_from_nc(dataset)

    # 2. Load and prepare the global land grid
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    grid_cells = gpd.read_file(grid_path)
    needs_wrap = grid_cells.geometry.apply(
        lambda g: any(lon > 180 for lon, _ in g.exterior.coords) if g is not None else False
    )
    if needs_wrap.any():
        grid_cells.loc[needs_wrap, "geometry"] = grid_cells.loc[needs_wrap, "geometry"].apply(adjust_longitude)

    # Standardize country identifier
    grid_cells["GID_0"] = grid_cells["iso3"] if "iso3" in grid_cells.columns else grid_cells["GID_0"]

    # 3. Identify Coastal Grid IDs from SRTM feature sets
    srtm_dir = OUTPUT_DIR / "SRTM" / "grid_data"
    srtm_files = list(srtm_dir.glob("srtm_grid_data_*.csv"))

    if not srtm_files:
        raise FileNotFoundError(f"No processed SRTM data found in {srtm_dir}. Ensure process_srtm.py has run.")

    coastal_ids = []
    for srtm_file in srtm_files:
        df_srtm = pd.read_csv(srtm_file)
        
        # Identify grid cells with coastline presence
        if "with_coast" in df_srtm.columns:
            ids = df_srtm[df_srtm["with_coast"] == 1]["id"].tolist()
            coastal_ids.extend(ids)
        elif "coast_length_meters" in df_srtm.columns:
            ids = df_srtm[df_srtm["coast_length_meters"] > 0]["id"].tolist()
            coastal_ids.extend(ids)

    # 4. Filter the grid to coastal geometries for surge processing
    # We subset the main grid based on the IDs collected from the SRTM features
    gdf_coastline = grid_cells[grid_cells["id"].isin(coastal_ids)].copy()

    # 5. Execute spatial risk assessment and export results
    output_csv = OUTPUT_DIR / "StormSurges" / "grid_data" / "global_grid_storm_surges_risk.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    process_storm_surge_risk(gdf_surges, gdf_coastline, grid_cells, output_csv)

if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    process_all_surges()