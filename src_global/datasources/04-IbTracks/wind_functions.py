import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString


# From IbTracks tracks, create wind_speed and track_distance features and
# aggregate them to pre-defined grid cells
def windfield_to_grid(tc, tracks, grids):
    df_windfield = pd.DataFrame()

    for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
        # Get the windfield
        windfield = intensity_sparse.toarray().flatten()
        npoints = len(windfield)
        # Get the track distance
        tc_track = tracks.get_track(track_name=event_id)
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        tc_track_line = LineString(points)
        DEG_TO_KM = 111.1
        tc_track_distance = grids["geometry"].apply(
            lambda point: point.distance(tc_track_line) * DEG_TO_KM
        )
        # Add to DF
        df_to_add = pd.DataFrame(
            {
                "GID_0": grids["iso3"],
                "typhoon_name": [tc_track.name] * npoints,
                "typhoon_year": [int(tc_track.sid[:4])] * npoints,
                "sid": [event_id] * npoints,
                "grid_point_id": grids["id"],
                "wind_speed": windfield,
                "track_distance": tc_track_distance,
                "geometry": grids.geometry,
            }
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
    return df_windfield


# Function to calculate mean values for neighboring cells
def calculate_mean_for_neighbors(idx, gdf, buffer_size):
    row = gdf.iloc[idx]
    if row["wind_speed"] == 0:  # Check if wind_speed is 0
        buffered = row["geometry"].buffer(
            buffer_size
        )  # Adjust buffer size as needed

        # Find neighboring geometries that intersect with the buffer, excluding the current geometry
        neighbors = gdf[
            ~gdf.geometry.equals(row["geometry"])
            & gdf.geometry.intersects(buffered)
        ]

        if not neighbors.empty:
            # drop rows with 0 windspeed vals (we dont want to compute the mean while considering these cells)
            neighbors = neighbors[neighbors["wind_speed"] != 0]
            if len(neighbors) != 0:
                mean_val = neighbors["wind_speed"].mean()
            else:
                mean_val = 0
            return mean_val
    return row[
        "wind_speed"
    ]  # Return the original value if no neighbors or wind_speed != 0


# Function to add interpolation points
def add_interpolation_points(data, num_points_between):
    new_x_list = []
    for i in range(len(data) - 1):
        start_point, end_point = data[i], data[i + 1]
        interp_x = list(
            np.linspace(start_point, end_point, num_points_between + 2)
        )
        if i == 0:
            new_x_list.append(interp_x)
        elif i == (len(data) - 1):
            new_x_list.append(interp_x)
        else:
            new_x_list.append(interp_x[1:])

    new_x = np.concatenate(new_x_list)

    return new_x


# Function to create xarray
def adjust_tracks(forecast_df, name="", custom_sid="", custom_idno=""):
    track = xr.Dataset(
        data_vars={
            "max_sustained_wind": (
                "time",
                np.array(forecast_df.MeanWind.values, dtype="float32"),
            ),  # 0.514444 --> kn to m/s
            "environmental_pressure": (
                "time",
                forecast_df.Pressure_env.values,
            ),  # I assume its enviromental pressure
            "central_pressure": ("time", forecast_df.Pressure.values),
            "lat": ("time", forecast_df.Latitude.values),
            "lon": ("time", forecast_df.Longitude.values),
            "radius_max_wind": ("time", forecast_df.RadiusMaxWinds.values),
            "radius_oci": (
                "time",
                forecast_df.RadiusOCI.values,
            ),  # Works even if there is a bunch of nans. Doesnt change the windspeed values
            "time_step": ("time", forecast_df.time_step),
            "basin": ("time", np.array(forecast_df.basin, dtype="<U2")),
        },
        coords={
            "time": forecast_df.forecast_time.values,
        },
        attrs={
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb",
            "name": name,
            "sid": custom_sid,
            "orig_event_flag": True,
            "data_provider": "Custom",
            "id_no": custom_idno,
            "category": int(max(forecast_df.Category.iloc)),
        },
    )
    track = track.set_coords(["lat", "lon"])
    return track


# Closest point from track to shapefile
def get_closest_point_index(track_points, shp):
    """
    Find the index of the closest point in track_points to the shapefile shp.

    Parameters:
    track_points (GeoDataFrame): GeoPandas DataFrame representing track points.
    shp (GeoDataFrame): GeoPandas DataFrame representing a shapefile.

    Returns:
    int: Index of the closest point in track_points.
    """
    # Convert shapefile to a single polygon (if it contains multiple geometries)
    shp_polygon = shp.unary_union

    # Calculate distance to each point in track_points
    closest_distance = float("inf")  # Initialize with infinity
    closest_point_index = None

    for i, point in enumerate(track_points.geometry):
        # Calculate distance to the shapefile
        distance = point.distance(shp_polygon)

        # Update closest point if this point is closer
        if distance < closest_distance:
            closest_distance = distance
            closest_point_index = i

    return closest_point_index
