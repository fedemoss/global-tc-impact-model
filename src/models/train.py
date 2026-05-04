import logging

import pandas as pd

from src.config import INPUT_DIR, FEATURES
from src.evaluation.cv_strategies import run_loocv_pipeline
from src.models.baselines import (
    HistoricalModel,
    WindspeedExposedModel,
    WindspeedHistoricalModel,
)
from src.models.two_stage_xgb import TwoStageXGBoost

logger = logging.getLogger(__name__)


def _attach_event_date(df):
    """Attach a single per-event ``date`` column derived from EM-DAT."""
    emdat = pd.read_csv(INPUT_DIR / "EMDAT" / "emdat.csv")
    emdat_red = emdat[["DisNo.", "Start Year", "Start Month", "Start Day"]].drop_duplicates()
    emdat_red["Start Day"] = emdat_red["Start Day"].fillna(1)
    emdat_red["date"] = pd.to_datetime(
        dict(
            year=emdat_red["Start Year"],
            month=emdat_red["Start Month"],
            day=emdat_red["Start Day"].astype(int),
        ),
        errors="coerce",
    )
    # Drop any pre-existing date column to avoid _x/_y collisions during merge.
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df.merge(emdat_red[["DisNo.", "date"]], on="DisNo.", how="left")


def prepare_data(aggregate_to_adm1=False):
    """Loads dataset, applies subnational filters, and optionally aggregates to ADM1 non-grid level."""
    df = pd.read_parquet(INPUT_DIR / "model_input_dataset" / "training_dataset.parquet")
    df = df.drop_duplicates().reset_index(drop=True)

    regions = pd.read_csv(INPUT_DIR / "model_input_dataset" / "un_regions.csv")
    df = df.merge(regions, on="iso3", how="left")

    # Filter events
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

    # Attach event-level date BEFORE aggregation so it can serve as a groupby key.
    df = _attach_event_date(df)

    if aggregate_to_adm1:
        agg_dict = {f: "mean" for f in FEATURES if f not in ["wind_speed", "rainfall_max_24h", "population", "N_events_5_years"]}
        agg_dict.update({
            "wind_speed": "max",
            "rainfall_max_24h": "max",
            "population": "sum",
            "perc_affected_pop_grid_region": "max",
            "Total Affected": "max",
            "N_events_5_years": "max",
        })
        groupby_keys = ["DisNo.", "sid", "level", "GID_0", "GID_1", "iso3", "cyclone_basin", "date"]
        df = df.groupby(groupby_keys, dropna=False).agg(agg_dict).reset_index()

    return df.drop_duplicates()


def execute_training_run(model_name, strategy, aggregate_to_adm1=False):
    """Orchestrates the data prep and LOOCV execution for a specific model and strategy."""
    logger.info("--- Starting Run ---")
    logger.info(f"Model: {model_name}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Level: {'ADM1' if aggregate_to_adm1 else 'Grid'}")

    df = prepare_data(aggregate_to_adm1=aggregate_to_adm1)
    events = df["DisNo."].unique()

    if model_name == "historical":
        model = HistoricalModel()
    elif model_name == "windspeed-exposed":
        model = WindspeedExposedModel()
    elif model_name == "windspeed-historical":
        model = WindspeedHistoricalModel()
    elif model_name == "2-stage-XGBoost":
        model = TwoStageXGBoost(features=FEATURES)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. Allowed: 'historical', "
            "'windspeed-exposed', 'windspeed-historical', '2-stage-XGBoost'."
        )

    level_str = "adm1" if aggregate_to_adm1 else "grid"
    output_folder = f"model_output/{model_name}_{strategy}_{level_str}"

    run_loocv_pipeline(
        df=df,
        events=events,
        model=model,
        strategy=strategy,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    execute_training_run("2-stage-XGBoost", "global", aggregate_to_adm1=True)