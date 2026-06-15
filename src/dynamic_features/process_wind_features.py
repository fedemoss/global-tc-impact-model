import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString

from src.config import INPUT_DIR, OUTPUT_DIR, resolve_iso3_list

def windfield_to_grid(tc, tracks, grids, cent_indices=None):
    """From IbTracks tracks, create wind_speed and track_distance features and aggregate to grid cells."""
    df_windfield = pd.DataFrame()

    for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
        windfield_cent = intensity_sparse.toarray().flatten()
        # Map centroid values to grid cells (handles CLIMADA's internal deduplication)
        windfield = windfield_cent[cent_indices] if cent_indices is not None else windfield_cent
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
    """Load global grid centroids."""
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_centroids.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Centroid grid not found at {grid_path}")
    gdf = pd.read_csv(grid_path)
    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf.Longitude, gdf.Latitude), crs="EPSG:4326")
    return gdf

def load_shapefile():
    """Load GADM ADM2 shapefile for landfall detection."""
    shp_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    return gpd.read_file(shp_path)[["GID_0", "geometry"]]

def load_impact_data():
    """Load unique TC events from the processed EMDAT output."""
    impact_data = pd.read_csv(OUTPUT_DIR / "EMDAT" / "impact_data_adm1_level.csv")
    return impact_data[["GID_0", "DisNo.", "Start Year", "Event Name", "sid"]].drop_duplicates()

def _get_closest_point_index(track_points, shp):
    """Return index of the track point closest to the country boundary (fallback for offshore storms)."""
    country_union = shp.union_all()
    distances = track_points.geometry.apply(lambda p: p.distance(country_union))
    return int(distances.idxmin())

def create_metadata(tracks, all_events, shp, iso3):
    """Build a per-storm metadata DataFrame with track dates and landfall info."""
    country_shp = shp[shp.GID_0 == iso3]
    rows = []
    for track_ds in tracks.data:
        sid   = track_ds.sid
        name  = track_ds.name
        year  = sid[:4]

        startdate = np.datetime64(np.array(track_ds.time[0]),  "D")
        enddate   = np.datetime64(np.array(track_ds.time[-1]), "D")

        track_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(track_ds.lon, track_ds.lat),
            crs=country_shp.crs,
        )
        try:
            min_index = int(country_shp.sjoin(track_points)["index_right"].min())
        except Exception:
            min_index = _get_closest_point_index(track_points, country_shp)

        landfall_dt   = np.datetime64(np.array(track_ds.time[min_index]), "s")
        landfalldate  = np.datetime64(landfall_dt, "D")
        landfall_time = str(landfall_dt).split("T")[1]

        rows.append({
            "GID_0": iso3, "sid": sid, "typhoon": name + year,
            "startdate": startdate, "enddate": enddate,
            "landfalldate": landfalldate, "landfall_time": landfall_time,
        })

    df = pd.DataFrame(rows)
    return df.merge(
        all_events[["GID_0", "sid", "DisNo.", "Start Year", "Event Name"]],
        on=["GID_0", "sid"], how="left",
    )

def get_storm_tracks(all_events):
    """Fetch IBTrACS tracks for all events. Wind convention is handled by CLIMADA."""
    sel_ibtracs = []
    problematic_sid = []

    for sid in all_events.sid:
        try:
            t = TCTracks.from_ibtracs_netcdf(storm_id=sid)
            t.get_track().sid  # raises if empty
            track = t.get_track()
            if len(track.time) == 0:
                problematic_sid.append(sid)
                continue
            sel_ibtracs.append(t)
        except:
            problematic_sid.append(sid)

    tc_tracks = TCTracks()
    for track in sel_ibtracs:
        tc_track = track.get_track()
        tc_track.interp(
            time=pd.date_range(tc_track.time.values[0], tc_track.time.values[-1], freq="30min")
        )
        tc_tracks.append(tc_track)

    return tc_tracks, problematic_sid

def process_storm_tracks(tc_tracks):
    tracks = TCTracks()
    for i in range(len(tc_tracks.get_track())):
        try:
            track_xarray = tc_tracks.get_track()[i]
        except:
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

def process_single_country(iso3, out_dir, gdf_global, all_events_global, shp_global):
    """Processes a single country's windfield and metadata."""
    wind_file = out_dir / f"windfield_data_{iso3}.csv"
    meta_file = out_dir / f"metadata_{iso3}.csv"
    if wind_file.exists() and meta_file.exists():
        return None

    gdf = gdf_global[gdf_global.iso3 == iso3].copy().reset_index(drop=True)
    cent = Centroids.from_geodataframe(gdf)
    all_events = all_events_global[all_events_global.GID_0 == iso3]

    tc_tracks, _ = get_storm_tracks(all_events=all_events)
    tracks = process_storm_tracks(tc_tracks=tc_tracks)

    # store_windfields=True causes a segfault on small centroid sets; intensity is
    # always stored regardless and is all we need for wind_speed output.
    tc = TropCyclone.from_tracks(tracks, centroids=cent, intensity_thres=0)

    # CLIMADA deduplicates identical lat/lon centroids internally, so tc.intensity
    # may have fewer columns than len(gdf). Build a mapping from each grid cell to
    # its CLIMADA centroid index so windfield_to_grid can expand values back to gdf.
    cent_lats = np.round(tc.centroids.lat, 6)
    cent_lons = np.round(tc.centroids.lon, 6)
    cent_lookup = {(lat, lon): idx for idx, (lat, lon) in enumerate(zip(cent_lats, cent_lons))}
    gdf_lats = np.round(gdf.Latitude.values, 6)
    gdf_lons = np.round(gdf.Longitude.values, 6)
    cent_indices = np.array([cent_lookup.get((lat, lon), 0) for lat, lon in zip(gdf_lats, gdf_lons)])

    if not wind_file.exists():
        df_wind = windfield_to_grid(tc=tc, tracks=tracks, grids=gdf, cent_indices=cent_indices)
        df_wind.to_csv(wind_file, index=False)

    if not meta_file.exists():
        df_meta = create_metadata(tracks=tracks, all_events=all_events, shp=shp_global, iso3=iso3)
        df_meta.to_csv(meta_file, index=False)

def generate_all_wind_features(max_workers=5, iso3_filter=None):
    """Entry point to execute wind processing."""
    print("Loading global grid centroids...")
    gdf_global = load_data()

    print("Loading global impact metadata...")
    all_events_global = load_impact_data()

    print("Loading country shapefile for landfall detection...")
    shp_global = load_shapefile()

    valid_iso3_list = [iso3 for iso3 in resolve_iso3_list() if iso3 in all_events_global.GID_0.unique()]
    if iso3_filter:
        valid_iso3_list = [iso3 for iso3 in valid_iso3_list if iso3 == iso3_filter]

    out_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting windfield processing for {len(valid_iso3_list)} countries...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_country, iso3, out_dir, gdf_global, all_events_global, shp_global
            ): iso3
            for iso3 in valid_iso3_list
        }

        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()
                print(f"Done: {iso3}")
            except Exception as e:
                print(f"Error processing wind data for {iso3}: {e}")

if __name__ == "__main__":
    generate_all_wind_features(max_workers=1, iso3_filter="ATG")