import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString

from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

logger = logging.getLogger(__name__)

def windfield_to_grid(tc, tracks, grids):
    """From IbTracks tracks, create wind_speed and track_distance features and aggregate to grid cells."""
    df_windfield = pd.DataFrame()

    for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
        windfield = intensity_sparse.toarray().flatten()
        npoints = len(windfield)
        
        tc_track = tracks.get_track(track_name=event_id)
        points = gpd.points_from_xy(tc_track.lon, tc_track.lat)
        tc_track_line = LineString(points)
        DEG_TO_KM = 111.1
        tc_track_distance = grids["geometry"].apply(
            lambda point: point.distance(tc_track_line) * DEG_TO_KM
        )
        
        df_to_add = pd.DataFrame({
            "GID_0": grids["iso3"],
            "typhoon_name": [tc_track.name] * npoints,
            "typhoon_year": [int(tc_track.sid[:4])] * npoints,
            "sid": [event_id] * npoints,
            "grid_point_id": grids["id"],
            "wind_speed": windfield,
            "track_distance": tc_track_distance,
            "geometry": grids.geometry,
        })
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
        
    return df_windfield

def add_interpolation_points(data, num_points_between):
    """Add interpolation points to smooth the track."""
    new_x_list = []
    for i in range(len(data) - 1):
        start_point, end_point = data[i], data[i + 1]
        interp_x = list(np.linspace(start_point, end_point, num_points_between + 2))
        if i == 0 or i == (len(data) - 1):
            new_x_list.append(interp_x)
        else:
            new_x_list.append(interp_x[1:])
    return np.concatenate(new_x_list)

def adjust_tracks(forecast_df, name="", custom_sid="", custom_idno=""):
    """Create xarray Dataset for the interpolated track."""
    track = xr.Dataset(
        data_vars={
            "max_sustained_wind": ("time", np.array(forecast_df.MeanWind.values, dtype="float32")),
            "environmental_pressure": ("time", forecast_df.Pressure_env.values),
            "central_pressure": ("time", forecast_df.Pressure.values),
            "lat": ("time", forecast_df.Latitude.values),
            "lon": ("time", forecast_df.Longitude.values),
            "radius_max_wind": ("time", forecast_df.RadiusMaxWinds.values),
            "radius_oci": ("time", forecast_df.RadiusOCI.values),
            "time_step": ("time", forecast_df.time_step),
            "basin": ("time", np.array(forecast_df.basin, dtype="<U2")),
        },
        coords={"time": forecast_df.forecast_time.values},
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
    return track.set_coords(["lat", "lon"])

def load_data():
    """Load global grid data and shapefile """
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_centroids.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Centroid grid not found at {grid_path}")
    
    gdf = pd.read_csv(grid_path)
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.Longitude, gdf.Latitude), crs="EPSG:4326")
    
    shp = gpd.read_file(INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg")
    return gdf, shp

def load_impact_data():
    """Load and merge impact data with windspeed flags."""
    impact_data = pd.read_csv(INPUT_DIR / "EMDAT" / "impact_data.csv")
    impact_metadata = pd.read_csv(INPUT_DIR / "IBTRACS" / "merged" / "tc_sustained_windspeed_information.csv")
    
    impact_data = impact_data.merge(
        impact_metadata[["DisNo.", "sid", "wind_sustained_flag"]], 
        on=["DisNo.", "sid"], 
        how="left"
    )
    all_events = impact_data[
        ["GID_0", "DisNo.", "Start Year", "Event Name", "sid", "wind_sustained_flag"]
    ].drop_duplicates()
    
    return all_events

def get_storm_tracks(all_events):
    conversion_to_10min = {'1-min': 0.88, '2-min': 0.88, '3-min': 0.88, '10-min': 1.0}
    sel_ibtracs = []
    problematic_sid = []
    
    for sid in all_events.sid:
        try:
            t = TCTracks.from_ibtracs_netcdf(storm_id=sid)
            t.get_track().sid  # Check if empty
            track = t.get_track()
            if len(track.time) == 0:
                problematic_sid.append(sid)
                continue

            ws_flag = all_events.loc[all_events['sid'] == sid, 'wind_sustained_flag'].values[0]
            factor = conversion_to_10min.get(ws_flag, 1.0)
            track['max_sustained_wind'] = track['max_sustained_wind'] * factor
            sel_ibtracs.append(t)
        except Exception as e:
            logger.warning(f"Skipping sid={sid}: {e}")
            problematic_sid.append(sid)

    tc_tracks = TCTracks()
    for track in sel_ibtracs:
        tc_track = track.get_track()
        tc_track.interp(
            time=pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30T")
        )
        tc_tracks.append(tc_track)
        
    return tc_tracks, problematic_sid

def process_storm_tracks(tc_tracks):
    tracks = TCTracks()
    for i in range(len(tc_tracks.get_track())):
        try:
            track_xarray = tc_tracks.get_track()[i]
        except (IndexError, TypeError):
            track_xarray = tc_tracks.get_track()
            
        w = np.array(track_xarray.max_sustained_wind)
        t = np.array(track_xarray.time)
        t_step = np.array(track_xarray.time_step)
        lat = np.array(track_xarray.lat)
        lon = np.array(track_xarray.lon)
        cp = np.array(track_xarray.central_pressure)
        ep = np.array(track_xarray.environmental_pressure)
        rmax = np.array(track_xarray.radius_max_wind)
        roci = np.array(track_xarray.radius_oci)

        num_points_between = 2

        new_w = add_interpolation_points(w, num_points_between)
        new_t_step = add_interpolation_points(t_step, num_points_between)
        new_lat = add_interpolation_points(lat, num_points_between)
        new_lon = add_interpolation_points(lon, num_points_between)
        new_cp = add_interpolation_points(cp, num_points_between)
        new_ep = add_interpolation_points(ep, num_points_between)
        new_rmax = add_interpolation_points(rmax, num_points_between)
        new_roci = add_interpolation_points(roci, num_points_between)

        timestamps = np.array([date.astype("datetime64[s]").astype("int64") for date in t])
        new_t = add_interpolation_points(timestamps, num_points_between)
        new_t = [np.datetime64(int(ts), "s") for ts in new_t]

        df_t = pd.DataFrame({
            "MeanWind": new_w, "Pressure_env": new_ep, "Pressure": new_cp,
            "Latitude": new_lat, "Longitude": new_lon, "RadiusMaxWinds": new_rmax,
            "RadiusOCI": new_roci, "time_step": new_t_step,
            "basin": np.array([np.array(track_xarray.basin)[0]] * len(new_t)),
            "forecast_time": new_t, "Category": track_xarray.category,
        })

        track = TCTracks()
        track.data = [adjust_tracks(df_t, name=track_xarray.name, custom_sid=track_xarray.sid, custom_idno=track_xarray.id_no)]
        tracks.append(track.get_track())
        
    return tracks

def process_single_country(iso3, out_dir, gdf_global, all_events_global):
    """Processes a single country's windfield and metadata data."""
    out_file = out_dir / f"windfield_data_{iso3}.csv"
    if out_file.exists(): return None

    # Filter global grid to current country
    gdf = gdf_global[gdf_global.iso3 == iso3].copy()
    cent = Centroids.from_geodataframe(gdf)
    all_events = all_events_global[all_events_global.GID_0 == iso3]

    tc_tracks, problematic_sid = get_storm_tracks(all_events=all_events)
    tracks = process_storm_tracks(tc_tracks=tc_tracks)

    # Calculate wind only for land centroids
    tc = TropCyclone.from_tracks(tracks, centroids=cent, store_windfields=True, intensity_thres=0)
    df_wind = windfield_to_grid(tc=tc, tracks=tracks, grids=gdf)

    df_wind.to_csv(out_file, index=False)
    return df_wind

def generate_all_wind_features(max_workers=5):
    """Entry point to execute wind processing."""
    logger.info("Loading global grid and shapefile data...")
    gdf_global, shp_global = load_data()
    shp_global["iso3"] = shp_global.GID_0

    logger.info("Loading global impact metadata...")
    all_events_global = load_impact_data()

    # Filter ISO3 list based on events present in impact dataset
    valid_iso3_list = [iso3 for iso3 in ISO3_LIST if iso3 in all_events_global.GID_0.unique()]

    out_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting windfield processing for {len(valid_iso3_list)} countries...")
    # CLIMADA's TC computations are CPU-bound NumPy work — processes give
    # actual parallelism whereas threads were blocked by the GIL.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_country, iso3, out_dir, gdf_global, all_events_global
            ): iso3
            for iso3 in valid_iso3_list
        }

        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing wind data for {iso3}: {e}", exc_info=True)


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    generate_all_wind_features(max_workers=5)