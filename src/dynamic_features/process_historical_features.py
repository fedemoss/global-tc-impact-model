import os
import pandas as pd
from src.config import INPUT_DIR, OUTPUT_DIR

def create_past_events_feature(df, grid_data_path):
    """
    Calculates the N_events_5_years feature (number of historical tropical
    cyclones that impacted a given grid cell in the preceding 5 years).

    Args:
        df (pd.DataFrame): Grid-level impact data merged with storm metadata
                           (must contain: id, iso3, DisNo., sid, landfalldate).
        grid_data_path (Path): Path to the global grid centroids CSV.

    Returns:
        pd.DataFrame: Grid dataset merged with historical event counts.
    """
    # Load grid data to get full universe of grid cells
    grid_data = pd.read_csv(grid_data_path)
    grid_cells = grid_data[['id', 'iso3']].drop_duplicates()
    
    # Convert dates
    df['startdate'] = pd.to_datetime(df['startdate'])
    df['enddate'] = pd.to_datetime(df['enddate'])
    df['landfalldate'] = pd.to_datetime(df['landfalldate'])
    
    # Sort events chronologically
    df_sorted = df.sort_values('landfalldate').reset_index(drop=True)
    
    # Extract unique events
    unique_events = df_sorted[['DisNo.', 'sid', 'landfalldate']].drop_duplicates()
    
    results = []
    
    # Iterate through each unique event to look back 5 years
    for _, event_row in unique_events.iterrows():
        current_event_id = event_row['DisNo.']
        current_date = event_row['landfalldate']
        
        # Define the 5-year lookback window
        window_start = current_date - pd.DateOffset(years=5)
        
        # Filter past events within the 5-year window, EXCLUDING the current event
        past_events = df_sorted[
            (df_sorted['landfalldate'] >= window_start) & 
            (df_sorted['landfalldate'] < current_date) & 
            (df_sorted['DisNo.'] != current_event_id)
        ]
        
        # Base dataframe with all grid cells initialized to 0 past events
        event_result = grid_cells.copy()
        event_result['DisNo.'] = current_event_id
        event_result['N_events_5_years'] = 0
        
        if not past_events.empty:
            # Count the number of past events that hit each grid cell (id)
            # A grid cell is considered "hit" if it appears in the past_events dataframe
            past_hits = past_events.groupby('id')['DisNo.'].nunique().reset_index()
            past_hits.rename(columns={'DisNo.': 'past_hit_count'}, inplace=True)
            
            # Merge the counts back into the event_result dataframe
            event_result = event_result.merge(past_hits, on='id', how='left')
            event_result['N_events_5_years'] = event_result['past_hit_count'].fillna(0)
            event_result.drop(columns=['past_hit_count'], inplace=True)
            
        results.append(event_result)
        
    # Concatenate all results
    final_historical_df = pd.concat(results, ignore_index=True)
    
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