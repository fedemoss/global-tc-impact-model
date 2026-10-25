import gc
import json
import logging

import joblib
import pandas as pd

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def _claim_output_folder(out_dir, strategy, event_col, exclude_col):
    """Record how this folder's folds are defined, and refuse to mix two.

    Fold filenames carry the predicted unit alone, so predictions left by a run
    that withheld a different unit are neither overwritten nor caught by the
    resume check -- recompute included, since that only rewrites the filenames
    the current definition produces. They are simply picked up by the final
    concatenation. Recording the definition turns that into an error, which the
    caller resolves by emptying the folder or naming another; the stale
    predictions are a run's worth of compute, so they are never removed here.
    """
    marker = out_dir / "fold_config.json"
    current = {"strategy": strategy, "event_col": event_col,
               "exclude_col": exclude_col or event_col}
    if not any(out_dir.glob("predictions_event_*.csv")):
        marker.write_text(json.dumps(current, indent=2))
        return

    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous != current:
        raise ValueError(
            f"{out_dir} holds predictions made with "
            f"{previous or 'an unrecorded fold definition'}, which cannot be "
            f"combined with {current}. Empty or rename the folder, or send this "
            f"run to another one."
        )


def _save_final_model(model, out_dir):
    """Persist trained components so SHAP can load them later."""
    if hasattr(model, "classifier"):
        joblib.dump(model.classifier, out_dir / "classifier.joblib")
    if hasattr(model, "regressor"):
        joblib.dump(model.regressor, out_dir / "regressor.joblib")
    if hasattr(model, "model"):
        joblib.dump(model.model, out_dir / "model.joblib")


def run_loocv_pipeline(df, events, model, strategy="global", output_folder="loocv_results",
                       event_col="sid", exclude_col="sid", recompute=False):
    """
    Executes LOOCV strategies based exactly on the original paper parameters.

    `event_col` selects the unit predicted per fold, and `events` must always be
    `df[event_col].unique()`:

    - "sid"     -- the physical cyclone (IBTrACS storm id); one prediction per
      storm.
    - "DisNo."  -- the EM-DAT country-event record; one prediction per country
      struck, which is the unit impact is reported at.

    `exclude_col` names the unit withheld from training, which need not be the
    unit predicted. A storm making landfall in several countries produces one
    EM-DAT record per country, all sharing a `sid` and describing the same
    hazard field, so leaving out a single country-event would leave its
    siblings in training. Holding out `sid` while predicting `DisNo.` keeps the
    per-country granularity without training on the storm under test. Setting
    it to `event_col`, or to None, withholds only the unit predicted.

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
    _claim_output_folder(out_dir, strategy, event_col, exclude_col)

    widen = bool(exclude_col) and exclude_col != event_col
    if widen:
        if exclude_col not in df.columns:
            raise ValueError(f"exclude_col {exclude_col!r} is not a column of the dataset")
        # A null here would silently exempt those rows from the exclusion,
        # leaving the test storm's own records in training.
        if df[exclude_col].isna().any():
            raise ValueError(
                f"rows with a null {exclude_col} cannot be withheld from training"
            )

    for ev in events:
        out_file = out_dir / f"predictions_event_{ev}.csv"
        
        # Existing predictions are reused by default, which makes an interrupted
        # run resumable. Pass recompute=True after changing the model or its
        # hyperparameters, or every event is skipped and the previous
        # predictions are silently kept.
        if out_file.exists() and not recompute:
            logger.info(f"Skipping event {ev}: file already exists.")
            continue

        is_test = df[event_col] == ev
        # Every row the fold is forbidden to train on: the unit under test, and
        # -- when the two differ -- the whole of whatever `exclude_col` unit it
        # belongs to. This is applied before each strategy's own restriction, so
        # none of them can readmit a withheld row: the test storm's sibling
        # records precede it by hours, which clears walk-forward's date cutoff,
        # and share its basin by construction.
        trainable = ~is_test
        if widen:
            trainable &= ~df[exclude_col].isin(df.loc[is_test, exclude_col].unique())

        # -------------------------------------------------------------
        # Exact Exclusion Logic per Strategy
        # -------------------------------------------------------------
        if strategy == "walk_forward":
            # Earliest record date of the storm -- multi-country storms have several
            event_date = df.loc[is_test, "date"].min()
            df_train = df[trainable & (df["date"] < event_date)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no past data to train on for walk-forward.")
                continue

        elif strategy == "geo_constrained":
            event_basin = df.loc[is_test, "cyclone_basin"].iloc[0]
            if pd.isna(event_basin):
                raise ValueError(
                    f"{ev} has no cyclone_basin. Every storm in IBTrACS has "
                    f"one, so a missing value means the column was dropped or "
                    f"overwritten before this point -- skipping the event here "
                    f"would quietly shrink the evaluation instead."
                )
            # Train only on cyclones in the exact same basin
            df_train = df[trainable & (df["cyclone_basin"] == event_basin)].copy()
            if df_train.empty:
                logger.info(f"Skipping {ev} — no other events in basin {event_basin} to train on.")
                continue

        elif strategy == "global":
            # Train on all cyclones except the target
            df_train = df[trainable].copy()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Test set is always the isolated cyclone
        df_test = df[is_test].copy()
        if df_test.empty:
            raise ValueError(
                f"no rows with {event_col} == {ev!r} -- `events` must be "
                f"df[{event_col!r}].unique(), not another identifier"
            )

        # A physical cyclone must never sit on both sides of a fold. Checked on
        # the withheld unit rather than the predicted one: with the two apart,
        # agreement on `event_col` alone would pass a fold whose training set
        # still held the test storm under a sibling country's record.
        checked = exclude_col or event_col
        overlap = set(df_train[checked].unique()) & set(df_test[checked].unique())
        assert not overlap, f"fold leakage: {checked} {sorted(overlap)} in both train and test"

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