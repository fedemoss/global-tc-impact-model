import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path

import xarray as xr
from climada.hazard import Centroids, TCTracks, TropCyclone
from shapely.geometry import LineString

from src.config import INPUT_DIR, OUTPUT_DIR, resolve_iso3_list

logger = logging.getLogger(__name__)


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

#: CLIMADA's wording when a storm id genuinely is not in the archive.
_ABSENT_FROM_IBTRACS = "not in IBTrACS"


def ensure_ibtracs_available(sample_sid, attempts=3, backoff=10):
    """Fetch the IBTrACS archive once, before any worker pool is created.

    CLIMADA downloads IBTrACS.ALL.v04r01.nc on the first track request. Issuing
    that request in the parent ensures the archive is complete on disk before
    concurrent workers read it.
    """
    for n in range(attempts):
        try:
            TCTracks.from_ibtracs_netcdf(storm_id=sample_sid)
            return True
        except ValueError as e:
            if _ABSENT_FROM_IBTRACS in str(e):
                return True  # archive readable; this particular id is unknown
            logger.warning(f"IBTrACS prefetch attempt {n + 1} failed: {e}")
        except Exception as e:
            logger.warning(f"IBTrACS prefetch attempt {n + 1} failed: {e}")
        if n + 1 < attempts:
            time.sleep(backoff * (n + 1))
    logger.error(
        "Could not fetch the IBTrACS archive; every wind field will be empty."
    )
    return False


def _fetch_track(sid, attempts=3, backoff=5):
    """Return (TCTracks, None) or (None, reason).

    A storm absent from IBTrACS is a property of the data and is final. An
    unreadable archive is a transient condition and is retried, so that the two
    are never conflated in the reported reason.
    """
    last = None
    for n in range(attempts):
        try:
            return TCTracks.from_ibtracs_netcdf(storm_id=sid), None
        except ValueError as e:
            if _ABSENT_FROM_IBTRACS in str(e):
                return None, f"absent from IBTrACS ({e})"
            last = e
        except (OSError, RuntimeError) as e:
            last = e  # missing or partially written netCDF, HDF read error
        except Exception as e:
            last = e
        if n + 1 < attempts:
            time.sleep(backoff * (n + 1))
    return None, f"unreadable after {attempts} attempts ({last})"


def get_storm_tracks(all_events):
    """Fetch IBTrACS tracks for all events. Wind convention is handled by CLIMADA."""
    sel_ibtracs = []
    problematic_sid = []

    for sid in all_events.sid:
        t, reason = _fetch_track(sid)
        if t is None:
            logger.warning(f"Skipping sid={sid}: {reason}")
            problematic_sid.append((sid, reason))
            continue
        try:
            track = t.get_track()
            if not isinstance(track, list) and len(track.time) == 0:
                problematic_sid.append((sid, "track has no timesteps"))
                continue
            sel_ibtracs.append(t)
        except Exception as e:
            logger.warning(f"Skipping sid={sid}: unusable track ({e})")
            problematic_sid.append((sid, f"unusable track ({e})"))

    # Tracks are passed through at their native resolution; the temporal
    # refinement is applied in process_storm_tracks, which interpolates every
    # track variable together.
    tc_tracks = TCTracks()
    for track in sel_ibtracs:
        tc_tracks.append(track.get_track())

    return tc_tracks, problematic_sid

def process_storm_tracks(tc_tracks):
    tracks = TCTracks()

    # CLIMADA's get_track() returns a list of Datasets for several tracks but a
    # bare Dataset for exactly one. Normalise to a list: indexing or taking the
    # length of a Dataset addresses its data variables, not its tracks.
    raw_tracks = tc_tracks.get_track()
    if not isinstance(raw_tracks, list):
        raw_tracks = [raw_tracks]
    if not raw_tracks:
        logger.warning("No usable storm tracks; returning empty TCTracks.")
        return tracks

    for track_xarray in raw_tracks:
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

def covers_storms(path, expected_sids, column="sid"):
    """True when `path` exists and already holds every expected storm id.

    Existing output is reused only when it covers the storms currently
    expected, so that a change upstream -- a corrected identifier, a new event
    -- is picked up on the next run instead of being masked by the file left
    by the previous one.
    """
    if not path.exists():
        return False
    try:
        have = set(pd.read_csv(path, usecols=[column])[column].astype(str))
    except (ValueError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return set(map(str, expected_sids)) <= have


def basin_lookup(refresh=False):
    """Map each storm id to the basin of its first record.

    Returned as a frame of (sid, cyclone_basin) and cached beside the wind
    fields. Tracks that cross between basins are represented by the basin they
    formed in, which is the attribution the basin-constrained evaluation uses.
    """
    path = OUTPUT_DIR / "IBTRACS" / "standard" / "cyclone_basin.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    from climada.util.constants import SYSTEM_DIR

    archives = sorted(Path(SYSTEM_DIR).glob("IBTrACS.ALL.*.nc"))
    if not archives:
        logger.error(f"No IBTrACS archive under {SYSTEM_DIR}; cannot map basins")
        return pd.DataFrame(columns=["sid", "cyclone_basin"])

    with xr.open_dataset(archives[-1]) as ds:
        decode = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        sids = [decode(s) for s in ds.sid.values]
        basins = [decode(b[0]).strip() for b in ds.basin.values]

    df = pd.DataFrame({"sid": sids, "cyclone_basin": basins})
    df = df[df["cyclone_basin"] != ""].drop_duplicates(subset="sid")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Basin lookup: {len(df):,} storms -> {path}")
    return df


def process_single_country(iso3, out_dir, gdf_global, all_events_global, shp_global):
    """Processes a single country's windfield and metadata."""
    wind_file = out_dir / f"windfield_data_{iso3}.csv"
    meta_file = out_dir / f"metadata_{iso3}.csv"
    all_events = all_events_global[all_events_global.GID_0 == iso3]
    expected = set(all_events["sid"].astype(str))
    if covers_storms(wind_file, expected) and covers_storms(meta_file, expected):
        return None
    if wind_file.exists() or meta_file.exists():
        logger.info(f"{iso3}: existing wind output does not cover the current "
                    f"storm set ({len(expected)} storms); recomputing")

    gdf = gdf_global[gdf_global.iso3 == iso3].copy().reset_index(drop=True)
    cent = Centroids.from_geodataframe(gdf)

    tc_tracks, problematic = get_storm_tracks(all_events=all_events)
    if problematic:
        # Written beside the wind field so that storms without a track are
        # recorded in the output tree as well as in the run log.
        pd.DataFrame(problematic, columns=["sid", "reason"]).to_csv(
            out_dir / f"skipped_tracks_{iso3}.csv", index=False)
        transient = [s for s, r in problematic if "absent from IBTrACS" not in r]
        logger.error(
            f"{iso3}: {len(problematic)} of {len(all_events)} storms produced no "
            f"track ({len(transient)} for reasons other than being absent from "
            f"IBTrACS) -> {out_dir / f'skipped_tracks_{iso3}.csv'}"
        )
    tracks = process_storm_tracks(tc_tracks=tc_tracks)

    # store_windfields=True causes a segfault on small centroid sets; intensity is
    # always stored regardless and is all we need for wind_speed output.
    #
    # ignore_distance_to_coast=True disables CLIMADA's inland culling, which
    # skips centroids beyond max_dist_inland_km (default 1000 km) from the
    # coast. The filter is a speed optimisation and does not affect the wind
    # field; such centroids take negligible wind. It also requires a NASA
    # distance-to-coast raster whose published URL no longer serves the file.
    tc = TropCyclone.from_tracks(
        tracks,
        centroids=cent,
        intensity_thres=0,
        ignore_distance_to_coast=True,
    )

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

# Set once in the parent before the pool is created; with the "fork" start
# method the workers inherit these copy-on-write instead of having the global
# grid and the GADM shapefile pickled and shipped for every country.
_WIND_CTX = {}


def _wind_worker(iso3):
    """ProcessPool entry point: reads the big frames from inherited globals."""
    return process_single_country(
        iso3,
        _WIND_CTX["out_dir"],
        _WIND_CTX["gdf_global"],
        _WIND_CTX["all_events_global"],
        _WIND_CTX["shp_global"],
    )


def generate_all_wind_features(max_workers=5, iso3_filter=None):
    """Entry point to execute wind processing."""
    logger.info("Loading global grid centroids...")
    gdf_global = load_data()

    logger.info("Loading global impact metadata...")
    all_events_global = load_impact_data()

    logger.info("Loading country shapefile for landfall detection...")
    shp_global = load_shapefile()

    valid_iso3_list = [iso3 for iso3 in resolve_iso3_list() if iso3 in all_events_global.GID_0.unique()]
    if iso3_filter:
        valid_iso3_list = [iso3 for iso3 in valid_iso3_list if iso3 == iso3_filter]

    out_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the archive before the pool is created; see
    # ensure_ibtracs_available.
    if len(all_events_global):
        logger.info("Prefetching the IBTrACS archive...")
        ensure_ibtracs_available(all_events_global.sid.iloc[0])

    basin_lookup()

    logger.info(f"Starting windfield processing for {len(valid_iso3_list)} countries...")

    # TropCyclone.from_tracks is CPU-bound numpy work, so threads serialise on
    # the GIL. Processes give real parallelism -- but only via fork-inherited
    # globals: passing gdf_global/shp_global as task arguments would pickle the
    # global grid and the GADM shapefile once per country and cost more than the
    # GIL ever did.
    _WIND_CTX.update(
        out_dir=out_dir, gdf_global=gdf_global,
        all_events_global=all_events_global, shp_global=shp_global,
    )

    if max_workers == 1:
        # Keep the single-worker path in-process: easier to debug, and it avoids
        # paying for a fork to run one country at a time.
        for iso3 in valid_iso3_list:
            try:
                _wind_worker(iso3)
                logger.info(f"Done: {iso3}")
            except Exception as e:
                logger.error(f"Error processing wind data for {iso3}: {e}", exc_info=True)
        return

    if mp.get_start_method(allow_none=True) != "fork":
        logger.warning(
            "the 'fork' start method is unavailable; wind workers will pickle the "
            "global grid per country. Reduce max_workers if memory is tight."
        )

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=mp.get_context("fork")) as executor:
        futures = {executor.submit(_wind_worker, iso3): iso3 for iso3 in valid_iso3_list}
        for future in as_completed(futures):
            iso3 = futures[future]
            try:
                future.result()
                logger.info(f"Done: {iso3}")
            except Exception as e:
                logger.error(f"Error processing wind data for {iso3}: {e}", exc_info=True)


if __name__ == "__main__":
    generate_all_wind_features(max_workers=1, iso3_filter="ATG")
