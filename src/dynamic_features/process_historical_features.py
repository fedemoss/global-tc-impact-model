import os
import numpy as np
import pandas as pd
from src.config import INPUT_DIR, OUTPUT_DIR

def create_past_events_feature(df, grid_data_path):
    """
    Calculates the N_events_5_years feature: for each event, the number of
    previous events that hit the same country in the preceding 5 years.
    The country-level count is assigned to every grid cell of that country.

    Args:
        df (pd.DataFrame): Grid-level impact data merged with storm metadata
                           (must contain: iso3, DisNo., landfalldate).
        grid_data_path (Path): Path to the global grid centroids CSV.

    Returns:
        pd.DataFrame: One row per (grid cell, event) for the impacted country,
                      with columns id, iso3, DisNo., N_events_5_years.
    """
    grid_data = pd.read_csv(grid_data_path)
    grid_cells = grid_data[['id', 'iso3']].drop_duplicates()

    df['landfalldate'] = pd.to_datetime(df['landfalldate'])

    # One row per (event, country)
    unique_events = (
        df[['DisNo.', 'iso3', 'landfalldate']]
        .drop_duplicates(subset=['DisNo.', 'iso3'])
        .dropna(subset=['landfalldate'])
    )

    # Count prior events per country within the 5-year window
    counted = []
    for _, group in unique_events.groupby('iso3'):
        group = group.sort_values('landfalldate')
        dates = group['landfalldate'].values
        window_starts = (group['landfalldate'] - pd.DateOffset(years=5)).values
        n_before = np.searchsorted(dates, dates, side='left')
        n_before_window = np.searchsorted(dates, window_starts, side='left')
        group = group.assign(N_events_5_years=n_before - n_before_window)
        counted.append(group)
    event_counts = pd.concat(counted, ignore_index=True)

    # Broadcast each country-level count to that country's grid cells only
    final_historical_df = grid_cells.merge(
        event_counts[['DisNo.', 'iso3', 'N_events_5_years']],
        on='iso3',
        how='inner',
    )

    return final_historical_df

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

    if out_file.exists():
        print(f"Historical feature dataset already exists: {out_file}")
        return

    print("Building merged impact + metadata dataset...")
    df = _build_impact_with_dates(iso3_filter=iso3_filter)

    print("Generating N_events_5_years feature...")
    historical_df = create_past_events_feature(df, grid_path)
    historical_df.to_csv(out_file, index=False)
    print(f"Saved historical features to {out_file}")

if __name__ == "__main__":
    generate_all_historical_features(iso3_filter="ATG")