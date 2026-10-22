"""Historical ensemble cyclone-track forecasts, from the TIGGE archive.

TIGGE collects the operational ensemble forecasts of the major weather
centres. The cyclone-track product carries, for each forecast cycle, every
ensemble member's projected track of every disturbance the centre was
following: position, central pressure and maximum wind at six-hourly steps.
That is what a forecaster had before landfall, so it is what a forecast-time
impact prediction has to be built from.

Two properties of the archive shape this module. Filenames are not uniform --
centres differ in the case of their identifier, the model they name and the
product suffix -- so the names are read from the archive's own index rather
than constructed. And centres miss cycles: a storm's file may simply not exist
for a given day, so a storm is collected as far as the archive covers it
rather than discarded for being incomplete.
"""
import logging
import re
import urllib.error
import urllib.request
from datetime import timedelta

import pandas as pd

from src.config import INPUT_DIR

logger = logging.getLogger(__name__)

#: The dataset's index, and the host its files are served from.
GDEX_DATASET = "d330003"
GDEX_API = f"https://gdex.ucar.edu/api/datasets/{GDEX_DATASET}/filelist"
GDEX_DATA = "https://data.gdex.ucar.edu"

#: The centres TIGGE carries, by their WMO identifier. Which of them covers a
#: given storm varies; the archive is the authority on what exists.
PROVIDERS = {
    "rksl": "Korea Meteorological Administration",
    "ecmf": "ECMWF",
    "kwbc": "NCEP",
    "egrr": "Met Office",
    "rjtd": "Japan Meteorological Agency",
    "lfpw": "Meteo-France",
    "ammc": "Bureau of Meteorology",
    "babj": "China Meteorological Administration",
}
DEFAULT_PROVIDER = "rksl"

#: Cycles to keep. Most centres issue this product at 00 and 12 UTC.
DEFAULT_CYCLES = ("00", "12")

#: How far before the reported start date to begin collecting. A forecast is
#: only useful here if it was issued before landfall.
DEFAULT_LEAD_DAYS = 3

#: The 14-digit cycle stamp every centre puts in the filename, whatever else
#: it names the file.
_STAMP = re.compile(r"_(\d{14})[_.]")

_group_cache = {}


def forecast_root(provider=DEFAULT_PROVIDER):
    return INPUT_DIR / "FORECAST" / provider


def _api(gindex=None):
    url = GDEX_API if gindex is None else f"{GDEX_API}/{gindex}"
    with urllib.request.urlopen(url, timeout=120) as response:
        import json

        return json.loads(response.read())


def _rows(gindex=None):
    """Every row of a listing, across all the sections it is split into.

    A year's files arrive in several groups rather than one, so reading only
    the first returns a fraction of what the archive holds.
    """
    payload = _api(gindex)
    rows = []
    for group in payload.get("data", {}).get("groups") or []:
        rows.extend(group.get("rows", []))
    return rows


def _provider_group(provider):
    """The index of a provider's section of the archive."""
    if "providers" not in _group_cache:
        _group_cache["providers"] = {
            row[0]["value"]: row[0]["url"].rsplit("/", 1)[-1] for row in _rows()
        }
    return _group_cache["providers"].get(provider)


def _year_group(provider, year):
    """The index of one provider-year, which is where filenames live."""
    key = (provider, year)
    if key not in _group_cache:
        gindex = _provider_group(provider)
        if gindex is None:
            _group_cache[key] = None
        else:
            found = None
            for row in _rows(gindex):
                if row[0]["value"].endswith(f"-{year}"):
                    found = row[0]["url"].rsplit("/", 1)[-1]
                    break
            _group_cache[key] = found
    return _group_cache[key]


def available_files(provider, year):
    """Every file the archive holds for a provider in a year.

    Returns the filename, the path it is served at and the cycle it belongs
    to, read from the index rather than assumed from a naming convention.
    """
    gindex = _year_group(provider, year)
    if gindex is None:
        return pd.DataFrame(columns=["name", "path", "cycle"])

    records = []
    for row in _rows(gindex):
        cell = row[0]
        if not cell.get("is_file"):
            continue
        stamp = _STAMP.search(cell["value"])
        if not stamp:
            continue
        records.append({
            "name": cell["value"],
            "path": cell.get("data_path") or "",
            "cycle": pd.to_datetime(stamp.group(1), format="%Y%m%d%H%M%S",
                                    utc=True),
        })
    return pd.DataFrame(records)


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as src:
            partial.write_bytes(src.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError) as e:
        partial.unlink(missing_ok=True)
        return False, str(e)
    # Renamed only once whole, so an interrupted run leaves no truncated XML
    # for the parser to meet later.
    partial.replace(destination)
    return True, None


def collect_event_forecasts(disno, start, end, provider=DEFAULT_PROVIDER,
                            lead_days=DEFAULT_LEAD_DAYS,
                            cycles=DEFAULT_CYCLES):
    """Every cycle the archive holds for one event's window.

    The window opens `lead_days` before the reported start, since a forecast
    is only of interest here if it was issued before the storm arrived, and
    closes at the reported end. Returns how many files are on disk afterwards,
    counting those already there: collection is resumable.
    """
    out_dir = forecast_root(provider) / disno
    first = pd.Timestamp(start, tz="UTC").normalize() - timedelta(days=lead_days)
    last = pd.Timestamp(end, tz="UTC").normalize() + timedelta(days=1)

    wanted = []
    for year in range(first.year, last.year + 1):
        files = available_files(provider, year)
        if files.empty:
            continue
        in_window = files[(files["cycle"] >= first) & (files["cycle"] <= last)]
        wanted.append(in_window[in_window["cycle"].dt.strftime("%H").isin(cycles)])

    if not wanted:
        return 0
    wanted = pd.concat(wanted, ignore_index=True)

    present, fetched, absent = 0, 0, 0
    for _, row in wanted.iterrows():
        target = out_dir / row["name"]
        if target.exists():
            present += 1
            continue
        ok, error = _download(f"{GDEX_DATA}{row['path']}", target)
        if ok:
            fetched += 1
            present += 1
        else:
            absent += 1
            logger.debug(f"{disno}: {row['name']} not retrieved ({error})")

    logger.info(f"{disno}: {present} cycles available"
                + (f", {fetched} newly fetched" if fetched else "")
                + (f", {absent} failed" if absent else ""))
    return present


def collect_all_forecasts(events, provider=DEFAULT_PROVIDER,
                          lead_days=DEFAULT_LEAD_DAYS, cycles=DEFAULT_CYCLES):
    """Collect forecasts for every event in `events`.

    `events` carries one row per event, with `DisNo.` and the reported start
    and end dates. Events are independent: one the archive does not cover
    leaves the rest untouched.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"TIGGE carries {sorted(PROVIDERS)}")
    logger.info(f"Collecting {len(events)} events from "
                f"{PROVIDERS[provider]} ({provider}) into {forecast_root(provider)}")

    counts = {}
    for _, row in events.iterrows():
        disno = row["DisNo."]
        try:
            counts[disno] = collect_event_forecasts(
                disno, row["start"], row["end"], provider, lead_days, cycles)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"{disno}: collection failed ({e})")
            counts[disno] = 0

    covered = sum(1 for n in counts.values() if n)
    logger.info(f"{covered} of {len(counts)} events have at least one cycle; "
                f"{sum(counts.values())} files in total")
    return pd.Series(counts, name="cycles").rename_axis("DisNo.")


def event_windows(impact_path=None, disnos=None):
    """The reported start and end date of each event, which bound its window."""
    impact_path = impact_path or INPUT_DIR / "EMDAT" / "emdat.csv"
    emdat = pd.read_csv(impact_path)
    cols = ["DisNo.", "Event Name", "Start Year", "Start Month", "Start Day",
            "End Year", "End Month", "End Day"]
    events = emdat[[c for c in cols if c in emdat.columns]].drop_duplicates("DisNo.")
    if disnos is not None:
        events = events[events["DisNo."].isin(disnos)]

    def to_date(year, month, day, default_day):
        return pd.to_datetime(dict(
            year=year, month=month.fillna(1),
            day=day.fillna(default_day).astype(int)), errors="coerce")

    events = events.assign(
        start=to_date(events["Start Year"], events["Start Month"],
                      events["Start Day"], 1),
        end=to_date(events["End Year"], events["End Month"],
                    events["End Day"], 28),
    )
    # An event whose window cannot be dated cannot be matched to a cycle.
    return events.dropna(subset=["start", "end"])
