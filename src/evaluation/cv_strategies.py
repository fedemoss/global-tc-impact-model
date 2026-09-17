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


def run_loocv_pipeline(df, events, model, strategy="global", output_folder="loocv_results",
                       event_col="DisNo."):
    """
    Executes LOOCV strategies based exactly on the original paper parameters.

    `event_col` selects the unit left out per fold, and `events` must always be
    `df[event_col].unique()`:

    - "DisNo." (default) -- the EM-DAT country-event record, as in the paper.
    - "sid"              -- the physical cyclone (IBTrACS storm id). A storm
      that hits several countries contributes several `DisNo.` records, so
      grouping by `sid` keeps all of them on one side of every fold. This is
      stricter, and only differs from the default once the dataset contains
      multi-country storms.

    Note that the per-fold output filenames are built from `events`, so the two
    settings write different filenames -- use a separate `output_folder` when
    switching, or old folds will be picked up by the resume check and mixed into
    `all_predictions_compiled.csv`.

    Strategies:
    - 'global': Standard LOOCV (train on all except test cyclone).
    - 'walk_forward': Train only on cyclones strictly before the test cyclone's
      earliest record date.
    - 'geo_constrained': Train only on cyclones in the same cyclone_basin as the
      test cyclone (basin of its first record, for the rare basin-crossers).
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
            # Earliest record date of the storm -- multi-country storms have several
            event_date = df.loc[df[event_col] == ev, "date"].min()
            df_train = df[(df[event_col] != ev) & (df["date"] < event_date)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no past data to train on for walk-forward.")
                continue
                
        elif strategy == "geo_constrained":
            event_basin = df.loc[df[event_col] == ev, "cyclone_basin"].iloc[0]
            # Train only on cyclones in the exact same basin
            df_train = df[(df[event_col] != ev) & (df["cyclone_basin"] == event_basin)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no other events in basin {event_basin} to train on.")
                continue
                
        elif strategy == "global":
            # Train on all cyclones except the target
            df_train = df[df[event_col] != ev].copy()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Test set is always the isolated cyclone
        df_test = df[df[event_col] == ev].copy()
        if df_test.empty:
            raise ValueError(
                f"no rows with {event_col} == {ev!r} -- `events` must be "
                f"df[{event_col!r}].unique(), not another identifier"
            )

        # A physical cyclone must never sit on both sides of a fold.
        overlap = set(df_train[event_col].unique()) & set(df_test[event_col].unique())
        assert not overlap, f"fold leakage: {sorted(overlap)} in both train and test"

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