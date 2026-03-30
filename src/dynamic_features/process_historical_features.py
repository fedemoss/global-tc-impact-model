import os
import pandas as pd
from src.config import INPUT_DIR, OUTPUT_DIR

def create_past_events_feature(impact_data_path, grid_data_path):
    """
    Calculates the N_events_5_years feature (number of historical tropical 
    cyclones that impacted a given grid cell in the preceding 5 years).
    
    Args:
        impact_data_path (Path): Path to the grid-level impact dataset.
        grid_data_path (Path): Path to the global grid information.
        
    Returns:
        pd.DataFrame: Grid dataset merged with historical event counts.
    """
    # Load impact data
    df = pd.read_csv(impact_data_path)
    
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

def generate_historical_features():
    """Entry point to execute historical feature generation."""
    impact_path = INPUT_DIR / "EMDAT" / "global_grid_impact_data.csv"
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_municipality_info.csv"
    
    out_dir = OUTPUT_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "historical_events_feature.csv"
    
    if out_file.exists():
        print("Historical feature dataset already exists.")
        return
        
    print("Generating N_events_5_years feature...")
    historical_df = create_past_events_feature(impact_path, grid_path)
    historical_df.to_csv(out_file, index=False)
    print(f"Saved historical features to {out_file}")

if __name__ == "__main__":
    generate_historical_features()