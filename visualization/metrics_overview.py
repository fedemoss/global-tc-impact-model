"""Shared loading and plotting helpers for the visualization notebooks.

Kept out of the notebooks so the figures and the tables are built from one
definition of what a model output is and how a prediction is categorised.
"""
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR
from src.evaluation.aggregation import (IMPACT_THRESHOLD_HIGH,
                                        IMPACT_THRESHOLD_LOW)

MODEL_ORDER = ["historical", "windspeed-exposed", "windspeed-historical",
               "2-stage-XGBoost"]
MODEL_LABELS = {
    "historical": "Historical",
    "windspeed-exposed": "Windspeed-exposed",
    "windspeed-historical": "Windspeed-historical",
    "2-stage-XGBoost": "2-stage XGBoost",
}
CATEGORY_LABELS = ["no impact", f"0-{IMPACT_THRESHOLD_HIGH}%",
                   f">{IMPACT_THRESHOLD_HIGH}%"]
STRATEGIES = ["global", "walk_forward", "geo_constrained"]


def output_folder(model_name, strategy="global", level="adm1"):
    return OUTPUT_DIR / f"model_output/{model_name}_{strategy}_{level}"


def load_model(model_name, strategy="global", level="adm1"):
    """Concatenate one model's per-event predictions, or None if absent."""
    folder = output_folder(model_name, strategy, level)
    files = sorted(folder.glob("predictions_event_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "GID_0" not in df.columns and "DisNo." in df.columns:
        df["GID_0"] = df["DisNo."].astype(str).str[-3:]
    return df


def load_all(strategy="global", level="adm1", models=MODEL_ORDER):
    """{model_name: frame} for whichever models have been scored."""
    out = {}
    for name in models:
        df = load_model(name, strategy, level)
        if df is not None:
            out[name] = df
    return out


def categorise(series, low=IMPACT_THRESHOLD_LOW, high=IMPACT_THRESHOLD_HIGH):
    """0 = no impact, 1 = up to `high`%, 2 = above `high`%."""
    return pd.cut(series, bins=[-np.inf, low, high, np.inf],
                  labels=[0, 1, 2]).astype(int)


def coverage(models_dict, keys=("DisNo.", "GID_1")):
    """Rows, events and shared keys per model.

    Metrics are only comparable across models scored on the same units, so
    this is the check to run before reading any table below it.
    """
    rows, keysets = [], {}
    for name, df in models_dict.items():
        present = [k for k in keys if k in df.columns]
        ks = set(map(tuple, df[present].astype(str).values)) if present else set()
        keysets[name] = ks
        rows.append({
            "model": MODEL_LABELS.get(name, name),
            "rows": len(df),
            "events": df["DisNo."].nunique() if "DisNo." in df else np.nan,
            "units": len(ks),
        })
    table = pd.DataFrame(rows)
    if len(keysets) > 1:
        shared = set.intersection(*keysets.values())
        table["shared_units"] = len(shared)
        table["only_this_model"] = [len(keysets[n] - shared) for n in models_dict]
    return table


def restrict_to_common(models_dict, keys=("DisNo.", "GID_1")):
    """Reduce every model to the units all of them scored."""
    present = [k for k in keys if all(k in df.columns for df in models_dict.values())]
    if not present or len(models_dict) < 2:
        return models_dict
    sets = [set(map(tuple, df[present].astype(str).values))
            for df in models_dict.values()]
    shared = set.intersection(*sets)
    out = {}
    for name, df in models_dict.items():
        mask = [tuple(r) in shared for r in df[present].astype(str).values]
        out[name] = df[mask].reset_index(drop=True)
    return out


def confusion_counts(df, target_col="reported", pred_col="predicted"):
    """3x3 category confusion matrix, rows reported and columns predicted."""
    t = categorise(df[target_col])
    p = categorise(df[pred_col])
    return (pd.crosstab(t, p)
            .reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
            .to_numpy())


def absolute_errors(df, target_col="reported", pred_col="predicted"):
    return (df[pred_col] - df[target_col]).abs().to_numpy()
