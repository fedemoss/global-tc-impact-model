import logging

import pandas as pd

from src.config import INPUT_DIR, OUTPUT_DIR, ISO3_LIST

logger = logging.getLogger(__name__)

def load_static_features(iso3):
    """
    Loads and merges all static spatial features for a specific country.

    """
    df_static = pd.DataFrame()

    # 1. Gridded Static Layers
    # The SRTM dataset contains elevation, slope, ruggedness, and coastal features.
    layers = {
        "SRTM": f"srtm_grid_data_{iso3}.csv",
        "Worldpop": f"population_grid_{iso3}.csv",
        "JRC": f"degree_of_urbanization_{iso3}.csv",
        "FloodRisk": f"flood_risk_{iso3}.csv",
        "SHDI": f"shdi_grid_{iso3}.csv"
    }

    for folder, file in layers.items():
        path = OUTPUT_DIR / folder / "grid_data" / file
        if path.exists():
            layer_df = pd.read_csv(path)
            
            if df_static.empty:
                df_static = layer_df
            else:
                # Merge on the unique grid identifier and country ISO code
                df_static = df_static.merge(layer_df, on=["id", "iso3"], how="left")

    if df_static.empty:
        return pd.DataFrame()

    # 2. Global Static Layers
    # Landslide Risk
    landslide_path = OUTPUT_DIR / "LandSlides" / "global_grid_landslide_risk.csv"
    if landslide_path.exists():
        df_land = pd.read_csv(landslide_path)
        # Filter global data to the specific country during merge
        df_static = df_static.merge(
            df_land[df_land.iso3 == iso3], 
            on=["id", "iso3"], 
            how="left"
        )

    # Storm Surge Risk
    surge_path = OUTPUT_DIR / "StormSurges" / "grid_data" / "global_grid_storm_surges_risk.csv"
    if surge_path.exists():
        df_surge = pd.read_csv(surge_path).rename(columns={"GID_0": "iso3"})
        df_static = df_static.merge(
            df_surge[df_surge.iso3 == iso3], 
            on=["id", "iso3"], 
            how="left"
        )

    return df_static


def build_country_dataset(iso3, df_meta, df_impact_master):
    """
    Core Logic: Maps grid-level hazard intensity to sub-national impact targets.
    """
    # 1. Load Wind Exposure (Grid level: 'id')
    wind_path = OUTPUT_DIR / "IBTRACS" / "standard" / f"windfield_data_{iso3}.csv"
    if not wind_path.exists(): return None
    df_wind = pd.read_csv(wind_path).rename(columns={"grid_point_id": "id"})
    
    # 2. Load Grid-to-GID Mapping (Crucial for sub-national alignment)
    # This file should be created by your grid_cells.py script
    grid_map_path = INPUT_DIR / "GRID" / "merged" / "global_grid_centroids.csv"
    if not grid_map_path.exists(): return None
    df_grid_map = pd.read_csv(grid_map_path)[["id", "iso3", "GID_1", "GID_2"]]
    df_grid_map = df_grid_map[df_grid_map.iso3 == iso3]

    # 3. Merge Wind with Grid Mapping
    df_dyn = df_wind.merge(df_grid_map, on=["id", "iso3"], how="inner")

    # 4. Merge with Impact Targets (Inner Join on sid AND GID)
    # This ensures a grid cell only gets a 'Total Affected' value if its parent GID was affected
    df_impact_country = df_impact_master[df_impact_master["iso3"] == iso3]
    
    # We join on GID_1 or GID_2 based on the impact data 'level'
    # For simplicity, we merge on all GID levels present in the impact registry
    df_dyn = df_dyn.merge(
        df_impact_country[["sid", "GID_1", "GID_2", "Total Affected", "Total Deaths"]], 
        on=["sid", "GID_1", "GID_2"], 
        how="inner"
    )
    
    if df_dyn.empty: return None

    # 5. Merge Rainfall & Metadata
    rain_path = OUTPUT_DIR / "PPS" / f"rainfall_data_{iso3}.csv"
    if rain_path.exists():
        df_dyn = df_dyn.merge(pd.read_csv(rain_path), on=["id", "iso3", "sid"], how="left")
        df_dyn["rainfall_max_24h"] = df_dyn["rainfall_max_24h"].fillna(0)
        
    df_dyn = df_dyn.merge(df_meta[["sid", "DisNo."]], on="sid", how="left")
        
    # 6. Merge Historical Frequency
    hist_path = OUTPUT_DIR / "dynamic_features" / "historical_events_feature.csv"
    if hist_path.exists():
        df_hist = pd.read_csv(hist_path)
        df_dyn = df_dyn.merge(df_hist[df_hist.iso3 == iso3], on=["id", "iso3", "DisNo."], how="left")
        df_dyn["N_events_5_years"] = df_dyn["N_events_5_years"].fillna(0)

    # 7. Merge Static Features
    df_static = load_static_features(iso3)
    if not df_static.empty:
        df_final = df_dyn.merge(df_static, on=["id", "iso3"], how="left")
    else:
        df_final = df_dyn
        
    return df_final

def compile_global_dataset():
    """Compiles the master global training dataset."""
    meta_path = INPUT_DIR / "IBTRACS" / "merged" / "meta_data.csv"
    if not meta_path.exists(): return
    df_meta = pd.read_csv(meta_path)[["sid", "DisNo."]].drop_duplicates()
        
    impact_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    if not impact_path.exists(): return
    df_impact_master = pd.read_csv(impact_path)

    master_dataset = []
    for iso3 in ISO3_LIST:
        df_country = build_country_dataset(iso3, df_meta, df_impact_master)
        if df_country is not None and not df_country.empty:
            master_dataset.append(df_country)
            
    if not master_dataset: return
    df_master = pd.concat(master_dataset, ignore_index=True)
    
    # Fill structural zeros for hazard/exposure columns
    fill_0 = ["mean_elev", "mean_slope", "mean_rug", "urban", "rural", "water",
              "storm_tide_rp_0010", "landslide_risk_sum", "flood_risk", "population", "coast_length_meters"]
    for col in fill_0:
        if col in df_master.columns:
            df_master[col] = df_master[col].fillna(0)
    
    # Median fill for SHDI
    if "shdi" in df_master.columns:
        df_master["shdi"] = df_master["shdi"].fillna(df_master["shdi"].median())
    
    out_path = INPUT_DIR / "model_input_dataset"
    out_path.mkdir(parents=True, exist_ok=True)
    df_master.to_parquet(out_path / "training_dataset.parquet", index=False)
    logger.info(f"Master training dataset saved. Total rows: {len(df_master)}")


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    compile_global_dataset()