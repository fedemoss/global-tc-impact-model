import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from src.config import INPUT_DIR, OUTPUT_DIR, resolve_iso3_list

def load_grid_gid_mapping(iso3):
    """
    Spatially joins grid cell centroids to their GID_1/GID_2 administrative unit.
    Mirrors the enrichment done in process_shdi.py::process_all_shdi, since the
    grid produced by grid_cells.py only carries GID_0 (country-level).
    """
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    adm2_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    if not grid_path.exists() or not adm2_path.exists():
        return pd.DataFrame(columns=["id", "iso3", "GID_1", "GID_2"])

    grid = gpd.read_file(grid_path)
    grid = grid[grid.iso3 == iso3].copy()
    if grid.empty:
        return pd.DataFrame(columns=["id", "iso3", "GID_1", "GID_2"])

    def national_fallback():
        # Small countries (e.g. ATG) where the 0.1 deg grid is too coarse relative
        # to admin-1 unit size have no usable GID_1 breakdown — evaluate the whole
        # country as a single GID_0 unit instead.
        df_grid_map = grid[["id", "iso3"]].copy()
        df_grid_map["GID_1"] = iso3
        df_grid_map["GID_2"] = None
        return df_grid_map

    adm_country = gpd.read_file(adm2_path)
    adm_country = adm_country[adm_country.GID_0 == iso3]
    if adm_country.empty:
        return national_fallback()

    grid["centroid"] = grid.geometry.centroid
    enriched = gpd.sjoin(
        grid.set_geometry("centroid"),
        adm_country[["GID_1", "GID_2", "geometry"]],
        how="left",
        predicate="within"
    )
    df_grid_map = enriched[["id", "iso3", "GID_1", "GID_2"]].drop_duplicates(subset="id")

    # If most grid cells fail to fall within any admin-1 polygon, the country's
    # GID_1 breakdown isn't practically usable — fall back to national level
    # rather than keeping a mostly-unmatched, patchy subnational mapping.
    match_rate = df_grid_map["GID_1"].notna().mean()
    if match_rate < 0.5:
        return national_fallback()

    df_grid_map[["GID_1", "GID_2"]] = df_grid_map[["GID_1", "GID_2"]].astype(object)
    return df_grid_map

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
                # Merge on the unique grid identifier (and iso3 when the layer has it —
                # e.g. JRC's output omits iso3, but "id" alone is already globally unique)
                merge_keys = ["id", "iso3"] if "iso3" in layer_df.columns else ["id"]
                # Drop columns already present (e.g. SHDI's own GID_1/GID_2 helper
                # columns) to avoid pandas _x/_y suffixing on the merge below.
                overlap = [c for c in layer_df.columns if c in df_static.columns and c not in merge_keys]
                df_static = df_static.merge(layer_df.drop(columns=overlap), on=merge_keys, how="left")

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

    # GID_1/GID_2 are internal helper columns from SHDI's own admin mapping — the
    # authoritative ones already live on df_dyn via load_grid_gid_mapping.
    return df_static.drop(columns=["GID_1", "GID_2"], errors="ignore")


def build_country_dataset(iso3, df_meta, df_impact_master):
    """
    Core Logic: Maps grid-level hazard intensity to sub-national impact targets.
    """
    # 1. Load Wind Exposure (Grid level: 'id')
    wind_path = OUTPUT_DIR / "IBTRACS" / "standard" / f"windfield_data_{iso3}.csv"
    if not wind_path.exists(): return None
    df_wind = pd.read_csv(wind_path).rename(columns={"grid_point_id": "id", "GID_0": "iso3"})
    
    # 2. Load Grid-to-GID Mapping (Crucial for sub-national alignment)
    df_grid_map = load_grid_gid_mapping(iso3)
    if df_grid_map.empty: return None

    # 3. Merge Wind with Grid Mapping
    df_dyn = df_wind.merge(df_grid_map, on=["id", "iso3"], how="inner")

    # 4. Merge with Impact Targets (Inner Join on sid AND GID)
    # This ensures a grid cell only gets a 'Total Affected' value if its parent GID was affected
    df_impact_country = df_impact_master[df_impact_master["iso3"] == iso3].copy()
    df_impact_country[["GID_1", "GID_2"]] = df_impact_country[["GID_1", "GID_2"]].astype(object)

    is_national_level = (df_grid_map["GID_1"] == iso3).all()
    if is_national_level:
        # GID_1 doesn't resolve subnationally for this country (see load_grid_gid_mapping) —
        # every grid cell shares the same national impact figures for a given event.
        impact_by_event = df_impact_country.groupby("sid").agg({
            "GID_0": "first", "level": "first",
            "Total Affected": "max", "perc_affected_pop_grid_region": "max"
        }).reset_index()
        df_dyn = df_dyn.merge(impact_by_event, on="sid", how="inner")
    else:
        # We join on GID_1 or GID_2 based on the impact data 'level'
        # For simplicity, we merge on all GID levels present in the impact registry
        df_dyn = df_dyn.merge(
            df_impact_country[["sid", "GID_1", "GID_2", "GID_0", "level", "Total Affected", "perc_affected_pop_grid_region"]],
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
    hist_path = OUTPUT_DIR / "features" / "historical_events_feature.csv"
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
    meta_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    meta_files = list(meta_dir.glob("metadata_*.csv"))
    if not meta_files: return
    df_meta = pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True)[["sid", "DisNo."]].drop_duplicates()
        
    impact_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    if not impact_path.exists(): return
    df_impact_master = pd.read_csv(impact_path)

    master_dataset = []
    for iso3 in resolve_iso3_list():
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
    print(f"Master training dataset saved. Total rows: {len(df_master)}")

if __name__ == "__main__":
    compile_global_dataset()