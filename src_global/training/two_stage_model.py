from xgboost.sklearn import XGBRegressor
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
import os
import logging
import gc

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "model.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),  # Append to log file
        logging.StreamHandler()  # Also print to console
    ]
)

features = [
    "wind_speed",
    # "track_distance",
    "rainfall_max_24h",
    # "N_events_5_years_weighted",
    # "N_events_5_years",
    "population",
    "coast_length",
    "with_coast",
    "mean_elev",
    "mean_slope",
    "mean_rug",
    "urban",
    "rural",
    "water",
    "storm_tide_rp_0010",
    "landslide_risk_sum",
    # "flood_risk",
    # "shdi",
    # "perc_affected_pop_grid_region"
    ]

xgb_params_classifier = {
        # the same
        "booster": "gbtree",
        "colsample_bytree": 0.8,
        "gamma": 0.5,
        "learning_rate": 0.01,
        "max_depth": 4,
        "min_child_weight": 1,
        "n_estimators": 100,
        "subsample": 0.8,
        "verbosity": 0,
        "random_state": 0,
        # binary classification
        "objective": "binary:logistic",  # outputs probability between 0 and 1
        "eval_metric": "logloss"         # binary classification loss
    }

xgb_params_regressor = {
        "booster": "gbtree",
        "colsample_bytree": 0.8,
        "gamma": 0.5,
        "learning_rate": 0.01,
        "max_depth": 4,
        "min_child_weight": 1,
        "n_estimators": 100,
        # "early_stopping_rounds": 10,
        "objective": "reg:squarederror",  # squared error
        "eval_metric": "rmse",            # RMSE metric
        "subsample": 0.8,
        "verbosity": 0,
        "random_state": 0,
}


# -------------------------
# Helper: oversampling
# -------------------------
def oversample(df, target, u=1):
    """
    Oversample majority class using multiplier u.
    Returns a shuffled dataframe.
    """
    minority_size = df[target].sum()
    majority_size = int(u * minority_size)
    
    df_balanced = (
        df.groupby(target, group_keys=False)
        .apply(lambda x: x.sample(n=min(minority_size if x.name == 1 else majority_size, len(x)), random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    return df_balanced

# -------------------------
# Stage 1 - Classification
# -------------------------
def stage1_classifier(df_train, df_test, features, clf_params):
    clf = XGBClassifier(**clf_params)
    clf.fit(df_train[features], df_train["reported_bin"])
    
    y_proba = clf.predict_proba(df_test[features])[:, 1]
    df_result = df_test.copy()
    df_result["predicted_proba"] = y_proba
    df_result["predicted_bin"] = (y_proba >= 0.5).astype(int)
    return df_result

# -------------------------
# Stage 2 - Regression
# -------------------------
def stage2_regressor(df_train, df_test, features, reg_params, target_name="perc_affected_pop_grid_region"):
    df_stage2 = df_test[df_test["predicted_bin"] == 1].copy()
    
    if not df_stage2.empty:
        reg = XGBRegressor(**reg_params)
        reg.fit(df_train[features], df_train[target_name])
        df_stage2["prediction_perc"] = reg.predict(df_stage2[features])
    
    df_stage2_zero = df_test[df_test["predicted_bin"] == 0].copy()
    df_stage2_zero["prediction_perc"] = 0
    
    df_test_full = pd.concat([df_stage2, df_stage2_zero], axis=0).sort_index()
    return df_test_full

# -------------------------
# LOOCV predictions
# -------------------------

def save_event_predictions(df_output, output_dir, event_id):
    """
    Save per-event predictions to CSV in output_dir.
    Skip if file already exists. Create directory if missing.
    
    Parameters
    ----------
    df_output : pd.DataFrame
        Predictions for a single event.
    output_dir : str
        Path to folder where outputs will be saved.
    event_id : str or int
        Identifier for the event; used as filename.
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # File path
    output_file = os.path.join(output_dir, f"predictions_event_{event_id}.csv")

    # Skip if already exists
    if os.path.exists(output_file):
        print(f"Skipping event {event_id}: file already exists.")
        return

    # Save
    df_output.to_csv(output_file, index=False)
    print(f"Saved predictions for event {event_id} -> {output_file}")


def loocv_predictions(df, events, features, clf_params, reg_params,
                      u1=3, u2=3, target_name="perc_affected_pop_grid_region",
                      output_dir=None, walk_forward=False, geo_constrained=False):
    """
    Run LOOCV: leave one event out for testing, train on all other events.
    Per-event oversampling using u1, u2.
    Saves each event prediction separately if output_dir is provided.
    Skips event if file already exists.
    Logs completion for each event.
    Memory is freed per iteration using gc.collect().
    """
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for ev in events:
        # Skip if output file exists
        if output_dir:
            out_file = os.path.join(output_dir, f"predictions_event_{ev}.csv")
            if os.path.exists(out_file):
                logging.info(f"Skipping event {ev}: file already exists.")
                continue
        if walk_forward:
            # year_event = df[df["DisNo."] == ev].year.iloc[0]
            # df_train = df[(df["DisNo."] != ev) & (df["year"] <= year_event)].copy()
            # df_test  = df[df["DisNo."] == ev].copy()
            event_date = df.loc[df["DisNo."] == ev, "date"].iloc[0]
            df_train = df[(df["DisNo."] != ev) & (df["date"] < event_date)].copy()
            df_test = df[df["DisNo."] == ev].copy()
            if df_train.empty:
                logging.info(f"Skipping {ev} — no past data to train on.")
                continue
        if geo_constrained:
            event_basin = df.loc[df["DisNo."] == ev, "cyclone_basin"].iloc[0]
            df_train = df[(df["DisNo."] != ev) & (df["cyclone_basin"] == event_basin)].copy()
            df_test = df[df["DisNo."] == ev].copy()

        else:
            df_train = df[df["DisNo."] != ev].copy()
            df_test  = df[df["DisNo."] == ev].copy()

        # Stage 1: classification
        df_train["reported_bin"] = (df_train[target_name] > 0).astype(int)
        df_test["reported_bin"]  = (df_test[target_name] > 0).astype(int)
        df_train_balanced_1stage = oversample(df_train, target="reported_bin", u=u1)
        df_stage1 = stage1_classifier(df_train_balanced_1stage, df_test, features, clf_params)

        # Stage 2: regression
        df_train_high = df_train[df_train[target_name] > 0].copy()
        df_train_high["impact_high"] = (df_train_high[target_name] >= 15).astype(int)
        df_train_balanced_2stage = oversample(df_train_high, target="impact_high", u=u2)
        df_stage2 = stage2_regressor(df_train_balanced_2stage, df_stage1, features, reg_params, target_name)

        df_stage2["method"] = "LOOCV"
        df_stage2["event"] = ev

        # Save per event if directory provided
        if output_dir:
            df_stage2.to_csv(out_file, index=False)
            logging.info(f"Saved predictions for event {ev} -> {out_file}")

        # Log completion
        logging.info(f"Completed processing LOOCV for event {ev}")

        # Free memory per iteration
        del df_train, df_test, df_train_balanced_1stage, df_stage1
        del df_train_high, df_train_balanced_2stage, df_stage2
        gc.collect()

    logging.info("LOOCV processing completed for all events.")





if __name__ == "__main__":

    # Training set
    df = pd.read_parquet("/data/big/fmoss/data/model_input_dataset/training_dataset_weighted_N_events_standard_windspeed.parquet")
    df = df.drop_duplicates().reset_index(drop=True)

    regions = pd.read_csv("/data/big/fmoss/data/model_input_dataset/un_regions.csv")
    # Add region information
    df = df.merge(regions, on="iso3")
    # People affected > 100 and subnational levels
    events_to_consider_c1 = df[df["Total Affected"] > 100]["DisNo."].unique()
    events_to_consider_c2 = df[df["level"] != "ADM0"]["DisNo."].unique()
    # More constraints regarding subnational reported events
    proportions = (
        df[df["level"] != "ADM0"]
        .assign(is_affected=lambda x: x["Total Affected"] > 0)
        .groupby(["DisNo.", "GID_0", "GID_1"])["is_affected"].max()
        .groupby(["DisNo.", "GID_0"]).mean()
        .reset_index(name="proportion_affected_gid1")
    )
    events_to_consider_c3 = proportions[proportions.proportion_affected_gid1 < 1]["DisNo."].unique()
    events_to_consider = list(set(events_to_consider_c1) & set(events_to_consider_c2) & set(events_to_consider_c3))
    df = df[df["DisNo."].isin(events_to_consider)].drop_duplicates()

    # Add year info
    df["year"] = df["DisNo."].apply(lambda x: int(x[:4]))

    # Add date info
    emdat = pd.read_csv("/data/big/fmoss/data/EMDAT/emdat-tropicalcyclone-2000-2022-processed-sids.csv")
    emdat_red = emdat[["DisNo.", 'Start Year', 'Start Month', 'Start Day']].drop_duplicates()
    # Replace missing Start Day with 1
    emdat_red["Start Day"] = emdat_red["Start Day"].fillna(1)
    # Create date column
    emdat_red["date"] = pd.to_datetime(
        dict(
            year=emdat_red["Start Year"],
            month=emdat_red["Start Month"],
            day=emdat_red["Start Day"].astype(int)
        ),
        errors="coerce"
    )
    dates = emdat_red[["DisNo.", "date"]].copy()
    df = df.merge(dates, how="left")

    # Duplicates
    df = df.drop_duplicates()
    # Testing set
    events_testing = df["DisNo."].unique()

    # LOOCV
    loocv_predictions(df=df, events=events_testing,
                           features=features, clf_params=xgb_params_classifier, reg_params=xgb_params_regressor,
                           output_dir="/data/big/fmoss/data/model_output/loocv_2_stg_model_geo/",
                           walk_forward=False,
                           geo_constrained=True
                           )
    # loocv_predictions(df=df, events=events_testing,
    #                   features=features, clf_params=xgb_params_classifier, reg_params=xgb_params_regressor,
    #                   output_dir="/data/big/fmoss/data/model_output/walk_forward_hr_2_stg_model/",
    #                   walk_forward=True
    #                   )
