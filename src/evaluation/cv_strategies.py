import os
import gc
import logging
import pandas as pd
from src.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
            logging.info(f"Skipping event {ev}: file already exists.")
            continue

        # -------------------------------------------------------------
        # Exact Exclusion Logic per Strategy
        # -------------------------------------------------------------
        if strategy == "walk_forward":
            event_date = df.loc[df["DisNo."] == ev, "date"].iloc[0]
            # Train only on events strictly before the test event's date
            df_train = df[(df["DisNo."] != ev) & (df["date"] < event_date)].copy()
            if df_train.empty:
                logging.info(f"Skipping {ev} — no past data to train on for walk-forward.")
                continue
                
        elif strategy == "geo_constrained":
            event_basin = df.loc[df["DisNo."] == ev, "cyclone_basin"].iloc[0]
            # Train only on events in the exact same basin
            df_train = df[(df["DisNo."] != ev) & (df["cyclone_basin"] == event_basin)].copy()
            if df_train.empty:
                logging.info(f"Skipping {ev} — no other events in basin {event_basin} to train on.")
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
        logging.info(f"Saved predictions for event {ev} using {strategy} strategy -> {out_file}")

        # Memory management per iteration
        del df_train, df_test, df_preds
        gc.collect()

    logging.info(f"LOOCV ({strategy}) processing completed for all events.")