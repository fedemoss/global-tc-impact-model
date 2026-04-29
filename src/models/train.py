import os
import pandas as pd
from src.config import INPUT_DIR, FEATURES
from src.models.two_stage_xgb import TwoStageXGBoost
from src.models.baselines import HistoricalModel, WindspeedExposedModel, WindspeedHistoricalModel
from src.evaluation.cv_strategies import run_loocv_pipeline

def prepare_data(aggregate_to_adm1=False):
    """Loads dataset, applies subnational filters, and optionally aggregates to ADM1 non-grid level."""
    
    # Load core data
    df = pd.read_parquet(INPUT_DIR / "model_input_dataset" / "training_dataset.parquet")
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Merge region info
    regions = pd.read_csv(INPUT_DIR / "model_input_dataset" / "un_regions.csv")
    df = df.merge(regions, on="iso3", how="left")
    
    # Apply filtering constraints
    events_to_consider_c1 = df[df["Total Affected"] > 100]["DisNo."].unique()
    events_to_consider_c2 = df[df["level"] != "ADM0"]["DisNo."].unique()
    
    proportions = (
        df[df["level"] != "ADM0"]
        .assign(is_affected=lambda x: x["Total Affected"] > 0)
        .groupby(["DisNo.", "GID_0", "GID_1"])["is_affected"].max()
        .groupby(["DisNo.", "GID_0"]).mean()
        .reset_index(name="proportion_affected_gid1")
    )
    events_to_consider_c3 = proportions[proportions.proportion_affected_gid1 < 1]["DisNo."].unique()
    
    valid_events = list(set(events_to_consider_c1) & set(events_to_consider_c2) & set(events_to_consider_c3))
    df = df[df["DisNo."].isin(valid_events)].drop_duplicates()

    # Conditionally Aggregate to ADM1
    if aggregate_to_adm1:
        # Features should be averaged or taken as max/sum depending on their nature
        agg_dict = {f: "mean" for f in FEATURES if f not in ["wind_speed", "rainfall_max_24h", "population", "N_events_5_years"]}
        agg_dict.update({
            "wind_speed": "max",
            "rainfall_max_24h": "max",
            "population": "sum",
            "perc_affected_pop_grid_region": "max", # Keep the target as reported
            "Total Affected": "max",
            "N_events_5_years": "max"
        })
        
        df = df.groupby(["DisNo.", "sid", "level", "GID_0", "GID_1", "iso3", "cyclone_basin", "date"]).agg(agg_dict).reset_index()

    # Add Date Info
    emdat = pd.read_csv(INPUT_DIR / "EMDAT" / "emdat-tropicalcyclone-2000-2022-processed-sids.csv")
    emdat_red = emdat[["DisNo.", 'Start Year', 'Start Month', 'Start Day']].drop_duplicates()
    emdat_red["Start Day"] = emdat_red["Start Day"].fillna(1)
    emdat_red["date"] = pd.to_datetime(dict(year=emdat_red["Start Year"], month=emdat_red["Start Month"], day=emdat_red["Start Day"].astype(int)), errors="coerce")
    
    df = df.merge(emdat_red[["DisNo.", "date"]], on="DisNo.", how="left").drop_duplicates()
    
    return df


def execute_training_run(model_name, strategy, aggregate_to_adm1=False):
    """Orchestrates the data prep and LOOCV execution for a specific model and strategy."""
    
    print(f"\n--- Starting Run ---")
    print(f"Model: {model_name}")
    print(f"Strategy: {strategy}")
    print(f"Level: {'ADM1' if aggregate_to_adm1 else 'Grid'}")
    
    df = prepare_data(aggregate_to_adm1=aggregate_to_adm1)
    events = df["DisNo."].unique()
    
    # Model Router strictly using Paper Conventions
    if model_name == "historical":
        model = HistoricalModel()
    elif model_name == "windspeed-exposed":
        model = WindspeedExposedModel()
    elif model_name == "windspeed-historical":
        model = WindspeedHistoricalModel()
    elif model_name == "2-stage-XGBoost":
        model = TwoStageXGBoost(features=FEATURES)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Allowed: 'historical', 'windspeed-exposed', 'windspeed-historical', '2-stage-XGBoost'.")
        
    # Output path routing
    level_str = "adm1" if aggregate_to_adm1 else "grid"
    output_folder = f"model_output/{model_name}_{strategy}_{level_str}"
    
    run_loocv_pipeline(
        df=df, 
        events=events, 
        model=model, 
        strategy=strategy, 
        output_folder=output_folder
    )


if __name__ == "__main__":
    # 1. Evaluate the proposed 2-Stage XGBoost
    execute_training_run("2-stage-XGBoost", "global", aggregate_to_adm1=True)
    # execute_training_run("2-stage-XGBoost", "walk_forward", aggregate_to_adm1=False)
    # execute_training_run("2-stage-XGBoost", "geo_constrained", aggregate_to_adm1=False)
    
    # 2. Compare against baselines
    # execute_training_run("historical", "global", aggregate_to_adm1=False)
    # execute_training_run("windspeed-exposed", "global", aggregate_to_adm1=False)
    # execute_training_run("windspeed-historical", "global", aggregate_to_adm1=False)