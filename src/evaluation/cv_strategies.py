import gc
import logging

import joblib
import pandas as pd

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def _save_final_model(model, out_dir):
    """Persist trained components so SHAP can load them later."""
    if hasattr(model, "classifier"):
        joblib.dump(model.classifier, out_dir / "classifier.joblib")
    if hasattr(model, "regressor"):
        joblib.dump(model.regressor, out_dir / "regressor.joblib")
    if hasattr(model, "model"):
        joblib.dump(model.model, out_dir / "model.joblib")


def run_loocv_pipeline(df, events, model, strategy="global", output_folder="loocv_results"):
    """
    Executes LOOCV strategies based exactly on the original paper parameters.
    
    Strategies:
    - 'global': Standard LOOCV (train on all except test event).
    - 'walk_forward': Train only on events occurring before the test event.
    - 'geo_constrained': Train only on events in the same cyclone_basin as the test event.
    """
    out_dir = OUTPUT_DIR / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    for ev in events:
        out_file = out_dir / f"predictions_event_{ev}.csv"
        
        # Skip if already processed
        if out_file.exists():
            logger.info(f"Skipping event {ev}: file already exists.")
            continue

        # -------------------------------------------------------------
        # Exact Exclusion Logic per Strategy
        # -------------------------------------------------------------
        if strategy == "walk_forward":
            event_date = df.loc[df["DisNo."] == ev, "date"].iloc[0]
            # Train only on events strictly before the test event's date
            df_train = df[(df["DisNo."] != ev) & (df["date"] < event_date)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no past data to train on for walk-forward.")
                continue
                
        elif strategy == "geo_constrained":
            event_basin = df.loc[df["DisNo."] == ev, "cyclone_basin"].iloc[0]
            # Train only on events in the exact same basin
            df_train = df[(df["DisNo."] != ev) & (df["cyclone_basin"] == event_basin)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no other events in basin {event_basin} to train on.")
                continue
                
        elif strategy == "global":
            # Train on all events except the target
            df_train = df[df["DisNo."] != ev].copy()
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Test set is always the isolated event
        df_test = df[df["DisNo."] == ev].copy()

        # -------------------------------------------------------------
        # Execute Model Pipeline
        # -------------------------------------------------------------
        # The model's train_and_predict method must handle the pipeline (e.g., 2-stage oversampling)
        df_preds = model.train_and_predict(df_train, df_test)
        
        # Append metadata
        df_preds["method"] = f"LOOCV_{strategy}"
        df_preds["event"] = ev

        # Save to disk
        df_preds.to_csv(out_file, index=False)
        logger.info(f"Saved predictions for event {ev} using {strategy} strategy -> {out_file}")

        # Memory management per iteration
        del df_train, df_test, df_preds
        gc.collect()

    # Persist a final fit on the full dataset so SHAP/interpretability
    # has a model to load. The LOOCV folds themselves stay event-isolated.
    try:
        model.train_and_predict(df.copy(), df.head(1).copy())
        _save_final_model(model, out_dir)
        logger.info(f"Final model artifacts written to {out_dir}")
    except Exception as e:
        logger.warning(f"Could not persist final model artifacts: {e}")

    # Concatenate per-event predictions into a single compiled file used by SHAP.
    pred_files = sorted(out_dir.glob("predictions_event_*.csv"))
    if pred_files:
        compiled = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
        compiled.to_csv(out_dir / "all_predictions_compiled.csv", index=False)

    logger.info(f"LOOCV ({strategy}) processing completed for all events.")