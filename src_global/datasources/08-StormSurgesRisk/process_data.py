#!/usr/bin/env python3
import os

import geopandas as gpd
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Point


def create_geodataframe_from_nc(dataset):
    # Extract the coordinates and storm tide variable for the 100-year return period
    lon = dataset[
        "station_x_coordinate"
    ].values  # Longitude (station_x_coordinate)
    lat = dataset[
        "station_y_coordinate"
    ].values  # Latitude (station_y_coordinate)
    storm_tide_10 = dataset[
        "storm_tide_rp_0010"
    ].values  # Storm tide for 10-year return period

    # Create a list of Point geometries using longitude and latitude
    points = [Point(lon[i], lat[i]) for i in range(len(lon))]

    # Create a GeoDataFrame with the Point geometries and the storm tide values
    gdf = gpd.GeoDataFrame(
        {"storm_tide_rp_0010": storm_tide_10},
        geometry=points,
        crs="EPSG:4326",  # WGS84 coordinate system
    )

    return gdf


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


# For missing raster values
def spatial_interpolation(df, columns_to_interpolate):
    """
    Perform spatial interpolation for specified columns in a GeoDataFrame.

    Parameters:
        df (GeoDataFrame): Input GeoDataFrame with geometries and missing values.
        columns_to_interpolate (list): List of column names to interpolate.

    Returns:
        GeoDataFrame: Updated GeoDataFrame with interpolated values.
    """

    # Extract centroids of the geometries
    df = df.copy()
    df["centroid"] = df.geometry.centroid
    coordinates = np.array([[point.x, point.y] for point in df.centroid])

    # Loop through columns to interpolate
    for column in columns_to_interpolate:
        # Check for missing values
        if df[column].isnull().any():
            # Find missing and non-missing indices
            missing_mask = df[column].isna().values
            non_missing_mask = ~missing_mask

            # Extract known values and their coordinates
            known_values = df.loc[non_missing_mask, column].values
            known_coords = coordinates[non_missing_mask]

            # Extract missing coordinates
            missing_coords = coordinates[missing_mask]

            # Fit nearest neighbors model on known points
            nbrs = NearestNeighbors(
                n_neighbors=min(8, len(known_coords)), algorithm="auto"
            ).fit(known_coords)
            distances, indices = nbrs.kneighbors(missing_coords)

            # Perform IDW (Inverse Distance Weighting) interpolation
            weights = 1 / (
                distances + 1e-10
            )  # Add a small value to avoid division by zero
            weights /= weights.sum(axis=1, keepdims=True)

            # Calculate interpolated values for missing points
            interpolated_values = (weights * known_values[indices]).sum(axis=1)

            # Update DataFrame with interpolated values
            df.loc[missing_mask, column] = interpolated_values
        else:
            continue
    # Drop the temporary centroid column
    df.drop(columns=["centroid"], inplace=True)

    return df


# Main function to process storm surge risk data
def process_storm_surge_risk(
    gdf, gdf_coastline, grid_cells, output_csv, R=0.5
):
    """
    Processes storm surge risk data by buffering geometries, performing spatial intersection,
    interpolating missing values, and saving the results to a CSV file.

    Parameters:
        gdf (GeoDataFrame): The main GeoDataFrame containing storm surge data.
        gdf_coastline (GeoDataFrame): Adjusted coastline GeoDataFrame.
        grid_cells (DataFrame): Grid cell data containing 'id' and 'GID_0'.
        output_csv (str): Path to save the output CSV file.
        R (float): Buffer radius in degrees around geometries.

    Returns:
        str: Path to the saved CSV file.
    """

    # Create a buffer of R degrees around the geometries
    gdf_buffered = gdf.copy()
    gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(R)

    # Perform spatial intersection with the buffered geometries
    intersected_gdf = gpd.sjoin(
        gdf_buffered, gdf_coastline, how="right", predicate="intersects"
    ).drop(columns="index_left")

    # Group by ID and country and aggregate using the max storm tide value
    intersected_gdf = (
        intersected_gdf.groupby(["id", "GID_0", "geometry"])
        .agg({"storm_tide_rp_0010": "max"})
        .reset_index()
    )

    # Copy to ensure it's a GeoDataFrame
    intersected_gdf_copy = gpd.GeoDataFrame(
        intersected_gdf.copy(), geometry="geometry"
    )

    # Identify valid (non-NaN) storm tide values
    intersected_gdf_copy["valid"] = intersected_gdf_copy[
        "storm_tide_rp_0010"
    ].notna()

    # Group by country (GID_0)
    country_groups = intersected_gdf_copy.groupby("GID_0")

    interpolated_gdf = []

    for country, group in country_groups:
        if group["valid"].all():
            # If all values are valid, no interpolation needed
            interpolated_gdf.append(group)
        elif not group["valid"].any():
            # If all values are NaN, skip
            continue
        else:
            # If some values are valid, perform IDW interpolation for NaNs
            group = spatial_interpolation(group, ["storm_tide_rp_0010"])
            interpolated_gdf.append(group)

    # Combine the updated GeoDataFrames
    interpolated_gdf = pd.concat(interpolated_gdf)
    interpolated_gdf.reset_index(drop=True, inplace=True)

    # Merge with grid cells, filling missing values with 0
    storm_surges_complete_df = interpolated_gdf[
        ["id", "GID_0", "storm_tide_rp_0010"]
    ].merge(grid_cells[["id", "GID_0"]], how="right")
    storm_surges_complete_df.fillna(0, inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    storm_surges_complete_df.to_csv(output_csv, index=False)

    print(f"Storm surge risk data saved to: {output_csv}")
    return output_csv


if __main__ == "__name__":
    # Load data
    dataset = xr.open_dataset(
        "/data/big/fmoss/data/StormSurges/storm_surges_data.nc"
    )
    gdf = create_geodataframe_from_nc(dataset)
    # Load grid cells
    grid_cells = gpd.read_file(
        "/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg"
    )
    grid_cells["geometry"] = grid_cells["geometry"].apply(adjust_longitude)
    # Load coastline data
    df_coastline = pd.read_csv(
        "/data/big/fmoss/data/SRTM/grid_data_coastline/merged/global_grid_coastline.csv"
    )
    df_coastline = df_coastline[df_coastline.with_coast == 1]
    gdf_coastline = gpd.GeoDataFrame(
        df_coastline.merge(grid_cells, how="left"), geometry="geometry"
    )[["id", "GID_0", "geometry"]]
    gdf_coastline["geometry"] = gdf_coastline["geometry"].apply(
        adjust_longitude
    )

    # Process storm surge risk data
    output_csv = "/data/big/fmoss/data/StormSurges/grid_data/global_grid_storm_surges_risk.csv"
    process_storm_surge_risk(gdf, gdf_coastline, grid_cells, output_csv)
