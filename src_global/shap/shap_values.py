#!/usr/bin/env python3
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import shap
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split

# Load model
df = pd.read_parquet("/data/big/fmoss/data/model_input_dataset/training_dataset_weighted_N_events.parquet")

features = [
    "wind_speed",
    # "track_distance",
    "rainfall_max_24h",
    "N_events_5_years",
    # "N_events_5_years_weighted",
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
    # "shdi"
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
        "early_stopping_rounds": 10,
        "objective": "reg:squarederror",  # squared error
        "eval_metric": "rmse",            # RMSE metric
        "subsample": 0.8,
        "verbosity": 0,
        "random_state": 0,
    }

def first_stage_classifier_proba(df_test, df_train, features, xgb_params, target_name, compute_shap=False):
    """
    First-stage classification: predict probability of positive class.
    Optionally compute SHAP values.
    """
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(df_train[features], df_train[target_name])

    y_proba = model.predict_proba(df_test[features])[:, 1]  # prob of class 1
    y_pred_default = (y_proba >= 0.5).astype(int)

    df_result = df_test.copy()
    df_result["predicted_proba"] = y_proba
    df_result["predicted_bin"] = y_pred_default

    shap_values = None
    if compute_shap:
        explainer = shap.Explainer(model, df_train[features])
        shap_values = explainer(df_test[features])

        # Plot feature importance summary
        shap.summary_plot(shap_values, df_test[features])

    return df_result, shap_values

def second_stage_regressor(df_train, df_test, features, target_name, xgb_params, compute_shap=False):
    """
    Second-stage regression after first-stage classification.
    Predicts target for rows with predicted_bin == 1, sets 0 for predicted_bin == 0.
    Optionally compute SHAP values.
    """
    try:
        # Rows predicted as 1 by first stage
        df_stage2 = df_test[df_test["predicted_bin"] == 1].copy()
        X_train, y_train = df_train[features], df_train[target_name]
        X_test_stage2 = df_stage2[features]

        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

        df_stage2["predicted"] = model.predict(X_test_stage2)

        # Rows predicted as 0 by first stage
        df_stage2_zero = df_test[df_test["predicted_bin"] == 0].copy()
        df_stage2_zero["predicted"] = 0

        # Concatenate and restore original order
        df_test_full = pd.concat([df_stage2, df_stage2_zero], axis=0).sort_index()

        shap_values = None
        if compute_shap and not df_stage2.empty:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test_stage2)

            # Plot feature importance summary
            shap.summary_plot(shap_values, X_test_stage2)

        return df_test_full, shap_values

    except Exception as e:
        print(f"Second stage training failed: {e}")
        return None, None

def compute_and_save_shap_first_stage(df, features, xgb_params, outdir, u1=3, random_state=42):
    """
    Compute SHAP values for the first-stage classifier using all data.
    Applies oversampling/undersampling before training.
    Saves SHAP values and trained model to `outdir`.
    """
    os.makedirs(outdir, exist_ok=True)

    df = df.copy()
    df["reported_bin"] = (df["perc_affected_pop_grid_region"] > 0).astype(int)

    # Balance dataset
    minority_size = df['reported_bin'].sum()
    majority_size = int(u1 * minority_size)

    df_balanced = (
        df.groupby('reported_bin', group_keys=False)
          .apply(lambda x: x.sample(
              minority_size if x.name == 1 else majority_size,
              replace=(x.name == 1 and minority_size > len(x)),  # oversample if needed
              random_state=random_state
          ))
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )

    # Train classifier
    X_train, y_train = df_balanced[features], df_balanced["reported_bin"]
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    # Compute SHAP values on full original dataset
    explainer = shap.Explainer(model, df[features])
    shap_values = explainer(df[features])

    # Save SHAP values and model
    shap_outfile = os.path.join(outdir, "shap_values_stage1.pkl")
    model_outfile = os.path.join(outdir, "xgb_classifier_stage1.pkl")

    joblib.dump(shap_values, shap_outfile)
    joblib.dump(model, model_outfile)

    print(f"✅ Saved SHAP values to {shap_outfile}")
    print(f"✅ Saved model to {model_outfile}")

    return shap_outfile, model_outfile

def compute_and_save_shap_second_stage(df, features, xgb_params, outdir, u2=3, random_state=42):
    """
    Compute SHAP values for second-stage regressor and save them to `outdir`.
    """
    os.makedirs(outdir, exist_ok=True)

    df = df.copy()
    df_high = df[df["perc_affected_pop_grid_region"] != 0].copy()
    df_high["impact_high"] = (df_high["perc_affected_pop_grid_region"] >= 15).astype(int)

    # Balance dataset
    minority_size = df_high['impact_high'].sum()
    majority_size = int(u2 * minority_size)

    df_balanced = (
        df_high.groupby('impact_high', group_keys=False)
        .apply(lambda x: x.sample(
            minority_size if x.name == 1 else majority_size,
            replace=(x.name == 1 and minority_size > len(x)),
            random_state=random_state
        ))
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    # Train regressor
    X_train, y_train = df_balanced[features], df_balanced["perc_affected_pop_grid_region"]
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    # Compute SHAP values on full dataset
    explainer = shap.Explainer(model, df[features])
    shap_values = explainer(df[features])

    # Save SHAP values and model
    shap_outfile = os.path.join(outdir, "shap_values_stage2.pkl")
    model_outfile = os.path.join(outdir, "xgb_regressor_stage2.pkl")

    joblib.dump(shap_values, shap_outfile)
    joblib.dump(model, model_outfile)

    print(f"✅ Saved SHAP values to {shap_outfile}")
    print(f"✅ Saved model to {model_outfile}")

    return shap_outfile, model_outfile

if __name__ == "__main__":

    shap_file, model_file = compute_and_save_shap_first_stage(
        df=df,
        features=features,
        xgb_params=xgb_params_classifier,
        outdir="/data/big/fmoss/data/model_output/shap/",
        u1=3
    )
    shap_file, model_file = compute_and_save_shap_second_stage(
        df=df,
        features=features,
        xgb_params=xgb_params_regressor,
        outdir="/data/big/fmoss/data/model_output/shap/",
        u2=3
    )
