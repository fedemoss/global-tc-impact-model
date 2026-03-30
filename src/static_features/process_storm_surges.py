import os
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon
from sklearn.neighbors import NearestNeighbors
from src.config import INPUT_DIR, OUTPUT_DIR

def create_geodataframe_from_nc(dataset):
    lon = dataset["station_x_coordinate"].values
    lat = dataset["station_y_coordinate"].values
    storm_tide_10 = dataset["storm_tide_rp_0010"].values

    points = [Point(lon[i], lat[i]) for i in range(len(lon))]
    return gpd.GeoDataFrame({"storm_tide_rp_0010": storm_tide_10}, geometry=points, crs="EPSG:4326")

def adjust_longitude(polygon):
    coords = list(polygon.exterior.coords)
    for i in range(len(coords)):
        lon, lat = coords[i]
        if lon > 180:
            coords[i] = (lon - 360, lat)
    return Polygon(coords)

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
    print(f"Storm surge risk data saved to: {output_csv}")

if __name__ == "__main__":
    dataset = xr.open_dataset(INPUT_DIR / "StormSurges" / "storm_surges_data.nc")
    gdf = create_geodataframe_from_nc(dataset)
    
    grid_cells = gpd.read_file(INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg")
    grid_cells["geometry"] = grid_cells["geometry"].apply(adjust_longitude)
    grid_cells["GID_0"] = grid_cells["iso3"] if "iso3" in grid_cells.columns else grid_cells["GID_0"]

    # Use the coastline feature created earlier in spatial_features.py
    df_coastline = pd.read_csv(OUTPUT_DIR / "features" / "coastline_length.csv")
    df_coastline = df_coastline[df_coastline.coast_length_meters > 0]
    
    gdf_coastline = gpd.GeoDataFrame(df_coastline.merge(grid_cells, how="left"), geometry="geometry")[["id", "GID_0", "geometry"]]
    gdf_coastline["geometry"] = gdf_coastline["geometry"].apply(adjust_longitude)

    output_csv = OUTPUT_DIR / "StormSurges" / "grid_data" / "global_grid_storm_surges_risk.csv"
    process_storm_surge_risk(gdf, gdf_coastline, grid_cells, output_csv)