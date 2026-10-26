import logging
import os

import numpy as np
import pandas as pd

from src.config import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

#: Column whose positive values mark the area an event reported impact over.
IMPACT_COLUMN = 'Total Affected'

#: What a past event is counted against.
LEVELS = ('grid', 'country')
DEFAULT_LEVEL = 'grid'


def _counts_by_grid(df, grid_cells, unique_events, impact_column):
    """Per cell: the past events whose reported area covered that cell.

    A past event counts for a cell only where it reported impact there. The
    sub-national expansion gives every cell of an affected country a row,
    zero-filled outside the reported units, so appearing in an event's records
    is not the same as having been hit by it -- counting rows would return the
    national total to every cell and leave the feature spatially flat.
    """
    if impact_column not in df.columns:
        raise KeyError(
            f"{impact_column!r} is needed to tell where an event was reported; "
            f"got {sorted(df.columns)}"
        )

    reported = (
        df.loc[pd.to_numeric(df[impact_column], errors='coerce') > 0,
               ['DisNo.', 'iso3', 'id']]
        .drop_duplicates()
        .merge(unique_events[['DisNo.', 'landfalldate']], on='DisNo.', how='inner')
    )

    results = []
    for iso3, events in unique_events.groupby('iso3'):
        cells = grid_cells.loc[grid_cells['iso3'] == iso3, ['id', 'iso3']]
        if cells.empty:
            continue
        country_hits = reported[reported['iso3'] == iso3]

        for _, event in events.iterrows():
            window_start = event['landfalldate'] - pd.DateOffset(years=5)
            past = country_hits[
                (country_hits['landfalldate'] >= window_start)
                & (country_hits['landfalldate'] < event['landfalldate'])
                & (country_hits['DisNo.'] != event['DisNo.'])
            ]
            counts = past.groupby('id')['DisNo.'].nunique()

            out = cells.copy()
            out['DisNo.'] = event['DisNo.']
            out['N_events_5_years'] = (
                out['id'].map(counts).fillna(0).astype('int32'))
            results.append(out)

    if not results:
        return pd.DataFrame(
            columns=['id', 'iso3', 'DisNo.', 'N_events_5_years'])
    return pd.concat(results, ignore_index=True)


def _counts_by_country(grid_cells, unique_events):
    """Per country-event: the past events in that country, given to every cell.

    The feature is then constant within a country-event, carrying how often the
    country is struck rather than where.
    """
    counted = []
    for _, group in unique_events.groupby('iso3'):
        group = group.sort_values('landfalldate')
        dates = group['landfalldate'].values
        window_starts = (group['landfalldate'] - pd.DateOffset(years=5)).values
        n_before = np.searchsorted(dates, dates, side='left')
        n_before_window = np.searchsorted(dates, window_starts, side='left')
        counted.append(group.assign(N_events_5_years=n_before - n_before_window))
    event_counts = pd.concat(counted, ignore_index=True)

    return grid_cells.merge(
        event_counts[['DisNo.', 'iso3', 'N_events_5_years']],
        on='iso3',
        how='inner',
    )


def create_past_events_feature(df, grid_data_path, level=DEFAULT_LEVEL,
                               impact_column=IMPACT_COLUMN):
    """
    Calculates the N_events_5_years feature: how many previously reported
    cyclones struck in the five years preceding the event.

    `level` decides what "struck" means, and the two are different features:

    * ``grid`` (default) counts, for each cell, the past events whose reported
      impacting area covered that cell, so the feature varies within a country.
    * ``country`` counts the past events anywhere in the country and gives that
      one number to every one of its cells, so the feature is flat within a
      country-event and separates country-events from one another only.

    Args:
        df (pd.DataFrame): Grid-level impact data merged with storm metadata
                           (must contain: id, iso3, DisNo., landfalldate, and
                           `impact_column` when `level` is ``grid``).
        grid_data_path (Path): Path to the global grid centroids CSV.
        level (str): ``grid`` or ``country``.
        impact_column (str): Column read to decide where an event was reported.

    Returns:
        pd.DataFrame: One row per (grid cell, event) for the impacted country,
                      with columns id, iso3, DisNo., N_events_5_years.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    grid_data = pd.read_csv(grid_data_path)
    grid_cells = grid_data[['id', 'iso3']].drop_duplicates()

    df = df.copy()
    df['landfalldate'] = pd.to_datetime(df['landfalldate'])

    # One row per (event, country)
    unique_events = (
        df[['DisNo.', 'iso3', 'landfalldate']]
        .drop_duplicates(subset=['DisNo.', 'iso3'])
        .dropna(subset=['landfalldate'])
    )

    if level == 'country':
        return _counts_by_country(grid_cells, unique_events)
    return _counts_by_grid(df, grid_cells, unique_events, impact_column)

def _build_impact_with_dates(iso3_filter=None):
    """Merge grid-level impact data with storm metadata to attach date columns."""
    impact_df = pd.read_csv(OUTPUT_DIR / "EMDAT" / "impact_data.csv")

    meta_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    meta_files = list(meta_dir.glob("metadata_*.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata files found in {meta_dir}. Run process_wind_features first.")
    metadata_df = pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True)

    if iso3_filter:
        impact_df   = impact_df[impact_df["iso3"] == iso3_filter]
        metadata_df = metadata_df[metadata_df["GID_0"] == iso3_filter]

    df = impact_df.merge(
        metadata_df[["DisNo.", "sid", "startdate", "enddate", "landfalldate"]],
        on=["DisNo.", "sid"],
        how="left",
    )
    return df

def generate_all_historical_features(iso3_filter=None, level=DEFAULT_LEVEL):
    """Entry point to execute historical feature generation.

    `level` picks which feature is built; see `create_past_events_feature`. The
    level that produced the file is recorded beside it, so a run at one level
    never reuses a file built at the other.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_centroids.csv"

    out_dir = OUTPUT_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix  = f"_{iso3_filter}" if iso3_filter else ""
    out_file = out_dir / f"historical_events_feature{suffix}.csv"
    level_file = out_file.with_suffix(".level")
    # dataset_builder reads the unsuffixed file only, so a filtered run writes a
    # file nothing consumes and N_events_5_years silently stays absent.
    if suffix:
        logger.warning(
            f"iso3_filter={iso3_filter!r} writes {out_file.name}, but dataset_builder "
            f"reads historical_events_feature.csv -- this output will not be picked up"
        )

    if out_file.exists():
        # A file with no level beside it predates the toggle, when the count
        # was taken per country.
        built_at = (level_file.read_text().strip()
                    if level_file.exists() else 'country')
        if built_at == level:
            print(f"Historical feature dataset already exists: {out_file}")
            return
        logger.info(f"{out_file.name} holds the {built_at}-level feature and "
                    f"{level} was asked for; rebuilding")

    print("Building merged impact + metadata dataset...")
    df = _build_impact_with_dates(iso3_filter=iso3_filter)

    print(f"Generating N_events_5_years feature ({level} level)...")
    historical_df = create_past_events_feature(df, grid_path, level=level)
    historical_df.to_csv(out_file, index=False)
    level_file.write_text(f"{level}\n")
    logger.info(f"Saved {level}-level historical features to {out_file}")


if __name__ == "__main__":
    generate_all_historical_features(iso3_filter="ATG")
