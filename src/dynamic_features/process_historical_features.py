import logging
import os

import pandas as pd

from src.config import INPUT_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

#: Column whose positive values mark the area an event reported impact over.
IMPACT_COLUMN = 'Total Affected'


def create_past_events_feature(df, grid_data_path, impact_column=IMPACT_COLUMN):
    """
    Calculates the N_events_5_years feature: for each grid cell, the number of
    previously reported cyclones whose reported impacting area intersected that
    cell in the five years preceding the event.

    The count is per cell, not per country. A past event counts for a cell only
    where it reported impact there: the sub-national expansion gives every cell
    of an affected country a row, zero-filled outside the reported units, so
    appearing in an event's records is not the same as having been hit by it.
    Counting rows rather than reported impact would return the national total to
    every cell of the country and leave the feature spatially flat.

    Args:
        df (pd.DataFrame): Grid-level impact data merged with storm metadata
                           (must contain: id, iso3, DisNo., landfalldate and
                           `impact_column`).
        grid_data_path (Path): Path to the global grid centroids CSV.
        impact_column (str): Column read to decide where an event was reported.

    Returns:
        pd.DataFrame: One row per (grid cell, event) for the impacted country,
                      with columns id, iso3, DisNo., N_events_5_years.
    """
    if impact_column not in df.columns:
        raise KeyError(
            f"{impact_column!r} is needed to tell where an event was reported; "
            f"got {sorted(df.columns)}"
        )

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

    # The cells each event actually reported impact over.
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

def generate_all_historical_features(iso3_filter=None):
    """Entry point to execute historical feature generation."""
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_centroids.csv"

    out_dir = OUTPUT_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix  = f"_{iso3_filter}" if iso3_filter else ""
    out_file = out_dir / f"historical_events_feature{suffix}.csv"
    # dataset_builder reads the unsuffixed file only, so a filtered run writes a
    # file nothing consumes and N_events_5_years silently stays absent.
    if suffix:
        logger.warning(
            f"iso3_filter={iso3_filter!r} writes {out_file.name}, but dataset_builder "
            f"reads historical_events_feature.csv -- this output will not be picked up"
        )

    if out_file.exists():
        print(f"Historical feature dataset already exists: {out_file}")
        return

    print("Building merged impact + metadata dataset...")
    df = _build_impact_with_dates(iso3_filter=iso3_filter)

    print("Generating N_events_5_years feature...")
    historical_df = create_past_events_feature(df, grid_path)
    historical_df.to_csv(out_file, index=False)
    logger.info(f"Saved historical features to {out_file}")


if __name__ == "__main__":
    generate_all_historical_features(iso3_filter="ATG")
