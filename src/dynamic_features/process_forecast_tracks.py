"""Ensemble forecast tracks, from the archived cyclone-track files.

Each file is one forecast cycle: every ensemble member's projected track of
every disturbance the centre was following at that hour. Turning those into
the model's wind feature takes three steps -- read the members out of the
file, decide which disturbance is the storm under study, and run each member
through the same wind model the historical features use.

The files are not well-formed XML. The writer closes `<data>` after every
`<disturbance>` rather than once per section, so a section's later
disturbances sit outside any element and a strict parser stops at the first
stray close. The disturbances themselves are intact, so they are read
individually and the enclosing section's attributes -- which carry the
ensemble member -- are tracked alongside. Discarding these files instead
would lose most of the archive.
"""
import gzip
import logging
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from src.collectors.forecast_collector import DEFAULT_PROVIDER, forecast_root

logger = logging.getLogger(__name__)

#: A `<data>` opening tag, which introduces one ensemble member's section.
_DATA_OPEN = re.compile(r"<data\b([^>]*)>")
#: One disturbance, which is well-formed even where its section is not.
_DISTURBANCE = re.compile(r"<disturbance\b.*?</disturbance>", re.S)
_ATTR = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def _read_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _sections(text):
    """Each `<data>` section's attributes, with the raw text that follows it.

    The section runs to the next `<data>` opening rather than to a closing
    tag, since the closings are unreliable.
    """
    opens = list(_DATA_OPEN.finditer(text))
    for i, match in enumerate(opens):
        end = opens[i + 1].start() if i + 1 < len(opens) else len(text)
        yield dict(_ATTR.findall(match.group(1))), text[match.end():end]


def read_cycle(path):
    """Every fix in one forecast cycle, one row per member, storm and hour."""
    rows = []
    for attrs, body in _sections(_read_text(path)):
        # The archive pads these fields, and a padded identifier would group
        # as a different member from the same one written without padding.
        member = (attrs.get("member") or "").strip() or None
        for block in _DISTURBANCE.findall(body):
            try:
                disturbance = ET.fromstring(block)
            except ET.ParseError:
                continue
            name = disturbance.findtext("cycloneName") or ""
            basin = disturbance.findtext("basin") or ""
            for fix in disturbance.findall("fix"):
                valid = fix.findtext("validTime")
                lat, lon = fix.findtext("latitude"), fix.findtext("longitude")
                if not (valid and lat and lon):
                    continue
                rows.append({
                    "origin": attrs.get("origin"),
                    "forecast_type": attrs.get("type"),
                    "ensemble_member": member,
                    "perturb": (attrs.get("perturb") or "").strip() or None,
                    "cyclone_name": name.strip(),
                    "basin": basin.strip(),
                    "hour": pd.to_numeric(fix.attrib.get("hour", "").strip(),
                                          errors="coerce"),
                    "valid_time": pd.to_datetime(valid, utc=True, errors="coerce"),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "pressure_hpa": pd.to_numeric(
                        fix.findtext("cycloneData/minimumPressure/pressure"),
                        errors="coerce"),
                    "wind_speed_ms": pd.to_numeric(
                        fix.findtext("cycloneData/maximumWind/speed"),
                        errors="coerce"),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["cycle"] = _cycle_time(path)
    return out


def _cycle_time(path):
    stamp = re.search(r"_(\d{14})[_.]", Path(path).name)
    return pd.to_datetime(stamp.group(1), format="%Y%m%d%H%M%S",
                          utc=True) if stamp else pd.NaT


def read_event(disno, provider=DEFAULT_PROVIDER):
    """Every cycle collected for one event, as one frame."""
    folder = forecast_root(provider) / disno
    files = sorted(list(folder.glob("*.xml")) + list(folder.glob("*.xml.gz")))
    if not files:
        return pd.DataFrame()

    frames, unreadable = [], 0
    for path in files:
        try:
            frame = read_cycle(path)
        except Exception as e:                                   # noqa: BLE001
            unreadable += 1
            logger.debug(f"{disno}: {path.name} unreadable ({e})")
            continue
        if not frame.empty:
            frames.append(frame)
    if unreadable:
        logger.warning(f"{disno}: {unreadable} of {len(files)} cycles unreadable")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------
# Deciding which disturbance is the storm
# --------------------------------------------------------------------------
def storm_aliases(event_name):
    """The names an event might be forecast under.

    EM-DAT writes a storm several ways -- a local name in quotes, an
    international name in brackets, or neither -- and a centre uses one of
    them. All are kept, reduced to letters and digits so that punctuation and
    case do not decide a match.
    """
    event_name = str(event_name or "")
    candidates = (re.findall(r"'([^']+)'", event_name)
                  + re.findall(r"\(([^)]+)\)", event_name)
                  + event_name.replace("/", " ").split())
    aliases = {re.sub(r"[^a-z0-9]", "", c.lower()) for c in candidates}
    return {a for a in aliases if len(a) > 2}


def name_matches(forecast_name, aliases, threshold=0.8):
    """Whether a forecast disturbance carries one of the event's names."""
    cleaned = re.sub(r"[^a-z0-9]", "", str(forecast_name).lower())
    if not cleaned or not aliases:
        return False
    if cleaned in aliases:
        return True
    return max(SequenceMatcher(None, cleaned, a).ratio()
               for a in aliases) >= threshold


def select_storm(cycles, event_name, threshold=0.8):
    """The rows of one storm, chosen by name across every cycle.

    A cycle follows several disturbances at once, so the storm has to be
    picked out by name rather than by being the only one present.
    """
    if cycles.empty:
        return cycles
    aliases = storm_aliases(event_name)
    names = cycles["cyclone_name"].dropna().unique()
    keep = {n for n in names if name_matches(n, aliases, threshold)}
    if not keep:
        logger.info(f"no disturbance matches {sorted(aliases)}; "
                    f"the cycles name {sorted(names)[:8]}")
        return cycles.iloc[0:0]
    return cycles[cycles["cyclone_name"].isin(keep)]


def member_tracks(storm_rows, lead_hours=None):
    """One track per (cycle, ensemble member), ordered in time.

    A member's track from a given cycle is a forecast in its own right, so
    they are kept apart rather than pooled: the spread between them is the
    uncertainty the study is about.
    """
    if storm_rows.empty:
        return []
    rows = storm_rows.dropna(subset=["valid_time", "latitude", "longitude"])
    tracks = []
    for (cycle, member), group in rows.groupby(["cycle", "ensemble_member"],
                                               dropna=False):
        group = group.sort_values("valid_time").drop_duplicates("valid_time")
        if len(group) < 2:
            continue
        if lead_hours is not None:
            group = group[group["valid_time"] <= cycle +
                          pd.Timedelta(hours=lead_hours)]
            if len(group) < 2:
                continue
        tracks.append(group.assign(cycle=cycle, ensemble_member=member))
    return tracks


# --------------------------------------------------------------------------
# Landfall, which is what lead time is measured from
# --------------------------------------------------------------------------
#: Interval the best track is resampled to before the crossing is located.
#: IBTrACS reports three-hourly, which puts landfall within an hour and a half
#: at best; the track between two reports is close enough to straight at this
#: scale for linear interpolation to sharpen that.
LANDFALL_STEP = "15min"


def country_outline(iso3, path=None):
    """One polygon for a country, read without loading the whole world."""
    import geopandas as gpd

    from src.config import INPUT_DIR

    path = path or INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    units = gpd.read_file(path, columns=["GID_0", "geometry"],
                          where=f"GID_0 = '{iso3}'")
    if units.empty:
        raise ValueError(f"{iso3} is not in {Path(path).name}")
    return units.union_all()


def best_track_points(sid):
    """The observed track of a storm, as time, longitude and latitude."""
    from src.dynamic_features.process_wind_features import _fetch_track

    tracks, reason = _fetch_track(sid)
    if tracks is None:
        raise ValueError(f"{sid}: {reason}")
    track = tracks.get_track()
    return pd.DataFrame({
        "time": pd.to_datetime(track.time.values, utc=True),
        "lon": np.asarray(track.lon.values, dtype=float),
        "lat": np.asarray(track.lat.values, dtype=float),
    })


def landfall_time(sid, iso3, step=LANDFALL_STEP, points=None, outline=None):
    """When the storm reached the country, from the observed track.

    Lead time is how long before the storm arrived a forecast was issued, so
    the arrival has to be one instant that every forecast of that storm is
    measured against. Taking it from the observed track rather than from each
    forecast keeps it that way: a member that carries the storm ashore late
    would otherwise be credited with more lead time for the same cycle.

    A storm that stays offshore still has an arrival -- its closest approach --
    and is reported as such, since impact does not require a crossing.
    """
    from shapely.geometry import Point

    points = best_track_points(sid) if points is None else points
    outline = country_outline(iso3) if outline is None else outline
    if len(points) < 2:
        raise ValueError(f"{sid}: too few track points to locate a crossing")

    # Resample the track so the crossing is not pinned to a reporting hour.
    dense = (points.set_index("time").sort_index()
             .resample(step).interpolate("linear").reset_index())
    inside = dense.apply(
        lambda r: outline.contains(Point(r["lon"], r["lat"])), axis=1)

    if inside.any():
        crossing = dense.loc[inside.idxmax()]
        return crossing["time"], "crossing"

    distance = dense.apply(
        lambda r: outline.distance(Point(r["lon"], r["lat"])), axis=1)
    closest = dense.loc[distance.idxmin()]
    logger.info(f"{sid} does not cross {iso3}; using closest approach "
                f"({distance.min():.2f} deg)")
    return closest["time"], "closest approach"


def lead_times(cycles, landfall):
    """Hours between each forecast cycle and the storm's arrival.

    Positive before arrival, so a larger number is a longer warning.
    """
    cycle = pd.to_datetime(pd.Series(cycles), utc=True)
    return (pd.Timestamp(landfall) - cycle).dt.total_seconds() / 3600.0
