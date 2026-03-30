import pandas as pd
import os
import gzip
import xml.etree.ElementTree as ET
import re
from rapidfuzz import fuzz, process
import geopandas as gpd
import sys

from shapely.geometry import LineString
from climada.hazard import Centroids, Hazard, TCTracks, TropCyclone
from climada_petals.hazard import TCForecast
import xarray as xr
import numpy as np

# EMDAT
emdat = pd.read_csv("/data/big/fmoss/data/EMDAT/impact_data.csv")
current_model = pd.read_parquet("/data/big/fmoss/data/model_output/merged/adm1_grouped/xgb_model.parquet").dropna()
events = current_model["DisNo."].unique()

impact_data = emdat[emdat["DisNo."].isin(events)][["Event Name", "GID_0" ,"DisNo.", "sid"]].drop_duplicates()

# Original file
original_emdat = pd.read_csv(
        "/data/big/fmoss/data/EMDAT/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
    )[["DisNo.", 'Start Year',
       'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']]

impact_data = impact_data.merge(original_emdat)

# Tested subsets of demonstrated matching events 

subset_events = [
    '2016-0256-TWN', '2016-0350-TWN', '2017-0432-PHL', '2012-0282-CHN',
    '2020-0308-CHN', '2016-0256-PHL', '2018-0347-CHN', '2016-0256-CHN',
    '2019-0472-KOR', '2016-0350-CHN', '2020-0425-VNM', '2019-0424-PRK',
    '2016-0490-PHL', '2019-0549-PHL', '2018-0285-CHN', '2020-0403-KOR',
    '2012-0294-CHN', '2016-0363-PHL', '2020-0470-VNM', '2016-0503-PHL',
    '2017-0432-JPN', '2018-0399-PHL', '2020-0470-PHL', '2012-0294-VNM',
    '2019-0573-PHL', '2016-0342-CHN'
       ]

impact_data = impact_data[impact_data["DisNo."].isin(subset_events)]

# Dirs
dirs = sorted(os.listdir("/data/big/fmoss/data/past_forecasts/events"))

# Grid
global_shp = gpd.read_file("/data/big/fmoss/data/SHP/GADM_adm1.gpkg")
global_grid_cells = gpd.read_file("/data/big/fmoss/data/GRID/merged/global_grid_land_overlap.gpkg")

def parse_cyclone_forecast(filepath, cyclone_name_of_interest=None, verbose=False):
    """
    Parse a TIGGE-style XML forecast file and extract fixes for each ensemble member.
    Optionally filter only the disturbance whose <cycloneName> matches the given name.
    """
    rows = []
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for data_tag in root.findall(".//data"):
            ftype   = data_tag.attrib.get("type")
            member  = data_tag.attrib.get("member")
            perturb = data_tag.attrib.get("perturb")
            origin  = data_tag.attrib.get("origin")

            # normalize member
            if member is not None:
                member = member.strip()
                try:
                    member = int(member)
                except ValueError:
                    pass

            # iterate disturbances inside this ensemble member
            for disturbance in data_tag.findall("disturbance"):
                cname = disturbance.findtext("cycloneName")
                if cname:
                    cname_clean = re.sub(r"[^a-z0-9]", "", cname.lower())
                else:
                    cname_clean = ""

                # If cyclone_name_of_interest is given, skip others
                if cyclone_name_of_interest:
                    target_clean = re.sub(r"[^a-z0-9]", "", cyclone_name_of_interest.lower())
                    if cname_clean != target_clean:
                        continue

                cnum   = disturbance.findtext("cycloneNumber")
                basin  = disturbance.findtext("basin")
                distid = disturbance.attrib.get("ID")

                # iterate through fixes (hours)
                for fix in disturbance.findall("fix"):
                    hour  = fix.attrib.get("hour", "").strip()
                    src   = fix.attrib.get("source", "").strip()
                    vtime = fix.findtext("validTime")
                    lat   = fix.findtext("latitude")
                    lon   = fix.findtext("longitude")

                    pressure = fix.find("cycloneData/minimumPressure/pressure")
                    wind     = fix.find("cycloneData/maximumWind/speed")

                    rows.append({
                        "origin": origin,
                        "forecast_type": ftype,
                        "ensemble_member": member,
                        "perturb": perturb,
                        "disturbance_id": distid,
                        "cyclone_name": cname,
                        "cyclone_number": cnum,
                        "basin": basin,
                        "hour": int(hour) if hour else None,
                        "source": src,
                        "valid_time": vtime,
                        "latitude": float(lat) if lat else None,
                        "longitude": float(lon) if lon else None,
                        "pressure_hpa": float(pressure.text) if pressure is not None else None,
                        "wind_speed_ms": float(wind.text) if wind is not None else None,
                    })

    except ET.ParseError:
        if verbose:
            print(f"⚠️ XML parse error in {filepath}")
        return pd.DataFrame()

    except Exception as e:
        print(f"⚠️ Unexpected error parsing {filepath}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(rows)

def collect_all_event_forecasts(
    impact_data: pd.DataFrame,
    forecast_base: str = "/data/big/fmoss/data/past_forecasts/events",
    parse_cyclone_forecast=None,
    match_threshold: int = 70,
    verbose: bool = False,
) -> dict[str, list[pd.DataFrame]]:
    """
    Parse all forecast XML files under forecast_base and return those that
    (a) fuzzily match the cyclone name(s) in impact_data, and
    (b) have initial time_0 <= event landfall date.

    Each returned dataframe is trimmed to the first continuous track and
    carries extra columns:
        'event_id', 'DisNo.', 'sid', 'GID_0', 'time_0', 'ensemble_member', 'perturb'
    """

    def get_first_track(df):
        if df.empty:
            return df
        hour_values = df['hour'].values
        for i in range(1, len(hour_values)):
            if hour_values[i] <= hour_values[i - 1]:
                return df.iloc[:i]
        return df

    def extract_aliases(event_name):
        quoted = re.findall(r"'([^']+)'", event_name)
        paren  = re.findall(r"\(([^)]+)\)", event_name)
        fallback = [event_name.split()[-1]]
        aliases = quoted + paren + fallback
        aliases = {re.sub(r"[^a-z0-9]", "", a.lower()) for a in aliases}
        return list(aliases)

    def is_match(forecast_name, aliases, thresh=70):
        forecast_clean = re.sub(r"[^a-z0-9]", "", forecast_name.lower())
        return max(fuzz.ratio(forecast_clean, a) for a in aliases) >= thresh

    all_events: dict[str, list[pd.DataFrame]] = {}

    for event_id in sorted(os.listdir(forecast_base)):
        row = impact_data[impact_data["DisNo."] == event_id]
        if row.empty:
            if verbose:
                print(f"⚠️  {event_id} not in impact_data")
            continue

        # ---------- event meta ----------
        aliases    = extract_aliases(row["Event Name"].iloc[0])
        sid_val    = row["sid"].iloc[0]  if "sid"  in row else None
        gid0_val   = row["GID_0"].iloc[0] if "GID_0" in row else None
        landfall_dt = pd.Timestamp(
            int(row["Start Year"].iloc[0]),
            int(row["Start Month"].iloc[0]),
            int(row["Start Day"].iloc[0]),
            tz="UTC"
        )

        path = os.path.join(forecast_base, event_id)
        xml_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml")]

        tracks = []
        for xml in xml_files:
            df = parse_cyclone_forecast(xml)
            if df.empty:
                continue
            if not is_match(df["cyclone_name"].iloc[0], aliases, match_threshold):
                continue

            # --- iterate over each ensemble member + perturbation ---
            group_cols = ["ensemble_member", "perturb"]
            for (member_val, perturb_val), member_df in df.groupby(group_cols, dropna=False):
                member_df = member_df.sort_values("hour")
                member_df = get_first_track(member_df)

                time0 = pd.to_datetime(member_df["valid_time"].iloc[0], utc=True, errors="coerce")
                if pd.isna(time0) or time0 > landfall_dt:
                    continue

                member_df = member_df.copy()
                member_df["event_id"]        = event_id
                member_df["DisNo."]          = event_id
                member_df["sid"]             = sid_val
                member_df["GID_0"]           = gid0_val
                member_df["time_0"]          = time0.isoformat()
                member_df["ensemble_member"] = member_val
                member_df["perturb"]         = perturb_val

                tracks.append(member_df)

        if tracks:
            all_events[event_id] = tracks
        elif verbose:
            print(f"⏭️  No pre-land-fall forecasts for {event_id} ({', '.join(aliases)})")

    return all_events

def adjust_tracks(forecast_df, name="", custom_sid="", custom_idno="", ensemble_member=None, perturb=None):
    track = xr.Dataset(
        data_vars={
            "max_sustained_wind": ("time", np.array(forecast_df.wind_speed_ms.values, dtype="float32")),
            "environmental_pressure": ("time", forecast_df.Pressure_env.values),
            "central_pressure": ("time", forecast_df.pressure_hpa.values),
            "lat": ("time", forecast_df.latitude.values),
            "lon": ("time", forecast_df.longitude.values),
            "radius_max_wind": ("time", forecast_df.RadiusMaxWinds.values),
            "radius_oci": ("time", forecast_df.RadiusOCI.values),
            "time_step": ("time", forecast_df.time_step),
            "basin": ("time", np.array(forecast_df.basin, dtype="<U2")),
        },
        coords={"time": forecast_df.valid_time.values},
        attrs={
            "max_sustained_wind_unit": "m/s",
            "central_pressure_unit": "mb",
            "name": name,
            "sid": custom_sid,
            "orig_event_flag": True,
            "data_provider": "Custom",
            "id_no": custom_idno,
            "category": int(max(forecast_df.Category.iloc)),
            "ensemble_member": ensemble_member,   # <-- added
            "perturb": perturb,                   # <-- added
        },
    )
    track = track.set_coords(["lat", "lon"])
    return track

def process_storm_dfs(storm_dfs):
    name = storm_dfs[0].iloc[0]["DisNo."]
    sid = storm_dfs[0].iloc[0].sid
    tracks = TCTracks()

    for df in storm_dfs:
        df = df.copy()
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.sort_values("valid_time")
        df["time_step"] = df["valid_time"].diff().dt.total_seconds().fillna(0) / 3600
        df["RadiusOCI"] = np.nan
        df["RadiusMaxWinds"] = np.nan
        df["Pressure_env"] = 1013.25
        df["Category"] = 0

        ensemble_member = df["ensemble_member"].iloc[0] if "ensemble_member" in df else None
        perturb         = df["perturb"].iloc[0] if "perturb" in df else None

        track = TCTracks()
        track.data = [
            adjust_tracks(
                df,
                name=name,
                custom_sid=f"{sid}-{df['valid_time'].iloc[0]}",
                custom_idno=df["valid_time"].iloc[0],
                ensemble_member=ensemble_member,
                perturb=perturb,
            )
        ]
        tracks.append(track.get_track())

    return tracks

# ---------------------------------------------------------------------
def windfield_to_grid(
    tc,
    tracks,
    grids,
    impact_data: pd.DataFrame,
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Convert sparse wind‑field arrays stored in *tc* to a grid‑level dataframe
    and compute the lead time (in hours) between each forecast cycle and the
    observed land‑fall date from *impact_data*.

    Parameters
    ----------
    tc : TropCyclone
        Object returned by TropCyclone.from_tracks(...), containing windfields.
    tracks : trackslib.Tracks
        Tracks container; must have .get_track().
    grids : geopandas.GeoDataFrame
        Grid polygons with columns ['id', 'GID_0', geometry].
    impact_data : pandas.DataFrame
        Must contain 'DisNo.', 'Start Year', 'Start Month', 'Start Day'.
    tz : str, default "UTC"
        Time zone used for the land‑fall timestamp.

    Returns
    -------
    pandas.DataFrame
        Grid‑level wind field with extra columns:
            - delta_time : hours until land‑fall (positive before land‑fall)
    """
    # ---- pre‑compute land‑fall timestamps per event -------------------
    lf_map = (
        impact_data.assign(
            landfall_dt=lambda d: pd.to_datetime(
                dict(year=d["Start Year"], month=d["Start Month"], day=d["Start Day"]),
                utc=True,
            )
        )[["DisNo.", "landfall_dt"]]
        .set_index("DisNo.")["landfall_dt"]
        .to_dict()
    )

    DEG_TO_KM = 111.1
    df_windfield = pd.DataFrame()

    # TropCyclone.intensity and event_name are aligned
    for intensity_sparse, event_name in zip(tc.intensity, tc.event_name):
        # ----------------------------------------------------------------
        # 1. Parse forecast initial time from `event_name`
        #    Format looks like:  {sid}-{YYYY-MM-DD HH:MM:SS+00:00}
        sid_part, date_part = event_name.split("-", 1)
        forecast_dt = pd.to_datetime(date_part, utc=True)

        # 2. Land‑fall timestamp for this storm (key = track.name / DisNo.)
        disno = tracks.get_track(track_name=event_name).name
        member = tracks.get_track(track_name=event_name).ensemble_member 
        landfall_dt = lf_map.get(disno)

        # Skip if no land‑fall information
        if landfall_dt is None:
            continue

        delta_hours = (landfall_dt - forecast_dt).total_seconds() / 3600.0

        # ----------------------------------------------------------------
        # 3. Wind field + distance
        windfield = intensity_sparse.toarray().flatten()
        tc_track = tracks.get_track(track_name=event_name)
        tc_track_line = LineString(gpd.points_from_xy(tc_track.lon, tc_track.lat))
        track_distance = (
            grids.geometry.to_crs(grids.crs)  # ensure same CRS
            .distance(tc_track_line)
            * DEG_TO_KM
        )

        # ----------------------------------------------------------------
        # 4. Assemble
        npoints = len(windfield)
        df_to_add = pd.DataFrame(
            {
                "GID_0": grids["GID_0"],
                "iso3": grids["GID_0"],
                "DisNo.": [disno] * npoints,
                "ensemble_member": [member] * npoints,
                "typhoon_year": [int(sid_part[:4])] * npoints,
                "sid": [event_name] * npoints,
                "id": grids["id"],
                "wind_speed": windfield,
                "track_distance": track_distance,
                "delta_time": [delta_hours] * npoints,        # NEW
                "geometry": grids.geometry,
            }
        )
        df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)

    return df_windfield

def create_windfield_db(results, impact_data):
    df_windfield_all_storms = pd.DataFrame()

    for storm, storm_tracks in results.items():
        country = storm_tracks[0].iloc[0].GID_0

        # Define centroids once per country
        grids = global_grid_cells[global_grid_cells.GID_0 == country].copy()
        grids.geometry = grids.geometry.to_crs(grids.crs).centroid
        cent = Centroids.from_geodataframe(grids)

        # Each element in storm_tracks is one member × time_0
        for track_df in storm_tracks:
            member = track_df.ensemble_member.iloc[0]
            if member is None:
                continue

            # Convert single forecast dataframe to TCTracks
            tracks = process_storm_dfs([track_df])

            # TropCyclone class
            tc = TropCyclone.from_tracks(
                tracks, centroids=cent, store_windfields=True, intensity_thres=0
            )

            # At grid level
            df_windfield = windfield_to_grid(tc, tracks, grids, impact_data)
            df_windfield_all_storms = pd.concat([df_windfield_all_storms, df_windfield], ignore_index=True)

    return df_windfield_all_storms

# Parallel processing
# from joblib import Parallel, delayed
# import pandas as pd
# def create_windfield_db(results, impact_data, n_jobs=2):
#     def process_one_track(storm, track_df, grids, cent):
#         """Process a single track (member × time_0)"""
#         try:
#             member = track_df.ensemble_member.iloc[0]
#             if member is None:
#                 return pd.DataFrame()

#             # Convert single forecast dataframe to TCTracks
#             tracks = process_storm_dfs([track_df])

#             # TropCyclone class
#             tc = TropCyclone.from_tracks(
#                 tracks, centroids=cent, store_windfields=True, intensity_thres=0
#             )

#             # At grid level
#             df_windfield = windfield_to_grid(tc, tracks, grids, impact_data)
#             return df_windfield

#         except Exception as e:
#             print(f"⚠️ Error in {storm} member={member}: {e}")
#             return pd.DataFrame()

#     all_jobs = []
#     for storm, storm_tracks in results.items():
#         country = storm_tracks[0].iloc[0].GID_0

#         # Define centroids once per country
#         grids = global_grid_cells[global_grid_cells.GID_0 == country].copy()
#         grids.geometry = grids.geometry.to_crs(grids.crs).centroid
#         cent = Centroids.from_geodataframe(grids)

#         # Schedule each member × time_0 as a job
#         for track_df in storm_tracks:
#             all_jobs.append(delayed(process_one_track)(storm, track_df, grids, cent))

#     # Run in parallel with 5 workers
#     results_parallel = Parallel(n_jobs=n_jobs, backend="loky")(all_jobs)

#     # Concatenate all resulting dataframes
#     df_windfield_all_storms = pd.concat(
#         [df for df in results_parallel if not df.empty], ignore_index=True
#     )

#     return df_windfield_all_storms


if __name__ == "__main__":

    results = collect_all_event_forecasts(
        impact_data,
        parse_cyclone_forecast=parse_cyclone_forecast,
        verbose=True
    )

    df_windfield = create_windfield_db(results, impact_data)
    df_windfield.drop(columns=["geometry"]).to_parquet(
        "/data/big/fmoss/data/past_forecasts/windspeed_dataset_kms_members.parquet",
        index=False
    )

    model_input = pd.read_parquet("/data/big/fmoss/data/model_input_dataset/training_dataset_weighted_N_events.parquet")
    events_to_consider = df_windfield["DisNo."].unique()
    stationary_data = model_input[model_input["DisNo."].isin(events_to_consider)].drop(columns=["wind_speed", "track_distance"])

    testing_data = df_windfield.drop("geometry", axis=1).merge(stationary_data, on=["GID_0", "DisNo.", "id"])

    testing_data.to_parquet("/data/big/fmoss/data/past_forecasts/testing_dataset_kms_members.parquet", index=False)

