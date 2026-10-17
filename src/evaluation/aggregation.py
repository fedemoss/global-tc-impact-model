"""Aggregate grid-cell predictions to the ADM1 units where impact is reported.

The model is fitted on 0.1 degree grid cells, but EM-DAT reports impact per
administrative unit, so every evaluation and every comparison against the
baselines happens at ADM1. The conversion is people-weighted rather than a mean
of percentages: a cell's predicted percentage is turned into people, summed over
the unit, and divided by the unit's population. Averaging the percentages
instead would weight a near-empty cell the same as a city.

This module is the single definition of that step. The hyperparameter search
scores candidates with it, and the post-training evaluation writes its output,
so the number a configuration is chosen on and the number finally reported
cannot drift apart.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET_NAME = "perc_affected_pop_grid_region"

# The two cutoffs the model is built around: 0 separates "affected at all" from
# untouched, 15% separates the high-impact class used for the f1_high objective.
IMPACT_THRESHOLD_LOW = 0
IMPACT_THRESHOLD_HIGH = 15

ADM1_KEYS = ["DisNo.", "GID_0", "GID_1"]


def aggregate_predictions_to_adm1(df_pred, target_name=TARGET_NAME,
                                  pred_col="prediction_perc"):
    """Grid-cell predictions -> one row per (event, ADM1 unit).

    Returns a frame with `reported` and `predicted` as percentages of the
    unit's population, plus the populations and people counts behind them.
    Units with no population are dropped: their percentage is undefined.
    """
    df = df_pred.copy()
    df[pred_col] = df[pred_col].clip(lower=0, upper=100)
    df["prediction_ppl"] = df[pred_col] * df["population"] / 100
    df["actual_ppl"] = df[target_name] * df["population"] / 100

    # One row per (event, cell). Duplicates would double a cell's contribution
    # to its unit's total.
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["DisNo.", "id"])

    grouped = (
        df.groupby(ADM1_KEYS, observed=True)
        .agg(prediction_ppl=("prediction_ppl", "sum"),
             actual_ppl=("actual_ppl", "sum"),
             actual_perc=(target_name, "max"),
             population=("population", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["population"] > 0]
    grouped["reported"] = grouped["actual_perc"].clip(upper=100)
    grouped["predicted"] = (
        100 * grouped["prediction_ppl"] / grouped["population"]).clip(upper=100)
    return grouped.dropna(subset=["reported", "predicted"])


def categorise(series, low=IMPACT_THRESHOLD_LOW, high=IMPACT_THRESHOLD_HIGH):
    """0 = no impact, 1 = up to `high`%, 2 = above `high`%."""
    import numpy as np
    bins = [-np.inf, low, high, np.inf]
    return pd.cut(series, bins=bins, labels=[0, 1, 2]).astype(int)


def aggregate_output_folder(grid_folder, adm1_folder, target_name=TARGET_NAME,
                            pred_col="prediction_perc"):
    """Write ADM1-level predictions from a folder of grid-level ones.

    The model trains and predicts per grid cell; `metrics.load_and_evaluate_all_models`
    reads whichever level it is asked for straight off disk. This is the step
    between them, and it keeps the per-event file layout so the evaluation does
    not need to know which level produced it.

    Returns the number of events written.
    """
    grid_folder, adm1_folder = Path(grid_folder), Path(adm1_folder)
    files = sorted(grid_folder.glob("predictions_event_*.csv"))
    if not files:
        logger.warning(f"No grid predictions in {grid_folder}")
        return 0

    adm1_folder.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in files:
        df = pd.read_csv(f)
        missing = {target_name, pred_col, "population", *ADM1_KEYS} - set(df.columns)
        if missing:
            logger.warning(f"{f.name}: missing {sorted(missing)}; skipped")
            continue
        grouped = aggregate_predictions_to_adm1(
            df, target_name=target_name, pred_col=pred_col)
        if grouped.empty:
            continue
        # Keep the column names the evaluation expects alongside the ADM1 ones.
        grouped[target_name] = grouped["reported"]
        grouped[pred_col] = grouped["predicted"]
        for col in ("sid", "method", "event"):
            if col in df.columns and df[col].nunique(dropna=False) == 1:
                grouped[col] = df[col].iloc[0]
        grouped.to_csv(adm1_folder / f.name, index=False)
        written += 1

    logger.info(f"Aggregated {written} events: {grid_folder.name} -> {adm1_folder.name}")
    return written
