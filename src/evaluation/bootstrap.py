"""Event-level bootstrap confidence intervals for the validation metrics.

Point estimates on their own say nothing about how much of a difference between
two models is real. These helpers attach nonparametric confidence intervals by
resampling.

**Resampling happens over whole events, not over rows.** ADM1 rows belonging to
one cyclone share a hazard field and a single EM-DAT report, so they are not
independent observations. Resampling rows would treat correlated data as
independent and produce intervals that are far too narrow - the usual way a
bootstrap flatters a model. Drawing whole events with replacement keeps each
storm's rows together and gives an interval that reflects the number of storms
actually observed, which is the real sample size.

Undefined metrics are handled differently in the point estimate and in the
replicates, deliberately:

* the point estimate scores an undefined ratio (0/0, e.g. precision when nothing
  is predicted positive) as 0.0, matching `sklearn`'s `zero_division=0` and the
  conventional way these numbers are reported;
* a replicate where the metric is undefined is *excluded* from the percentile
  calculation and counted in `<metric>_n_undefined_boot`. Scoring it 0.0 would
  drag the lower bound down for a reason that has nothing to do with model
  quality, and dropping it silently would hide how often it happened. Check that
  column before quoting an interval: one resting on few valid replicates is not
  worth much.

Intervals are percentile-method, so they are neither bias- nor skew-corrected
(no BCa). That is fine for these metrics but worth stating in a methods section.
"""
import numpy as np
import pandas as pd

METRIC_NAMES = [
    "precision", "recall", "specificity", "accuracy",
    "fpr", "fnr", "f1", "csi", "mean_abs_error", "median_abs_error",
]


def compute_metrics_at_threshold(
    df,
    threshold,
    target_col="perc_affected_pop_grid_region",
    pred_col="prediction_perc",
    nan_on_undefined=True,
):
    """Binary and error metrics for one threshold.

    With `nan_on_undefined=True` an undefined ratio returns NaN, so a bootstrap
    replicate can be dropped rather than counted as zero; with False it returns
    0.0, reproducing `sklearn`'s `zero_division=0` for the point estimate.
    """
    reported_bin = (df[target_col] > threshold).astype(int)
    predicted_bin = (df[pred_col] > threshold).astype(int)

    TP = int(((predicted_bin == 1) & (reported_bin == 1)).sum())
    TN = int(((predicted_bin == 0) & (reported_bin == 0)).sum())
    FP = int(((predicted_bin == 1) & (reported_bin == 0)).sum())
    FN = int(((predicted_bin == 0) & (reported_bin == 1)).sum())

    def safe_div(num, denom):
        if denom == 0:
            return np.nan if nan_on_undefined else 0.0
        return num / denom

    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    specificity = safe_div(TN, TN + FP)
    fpr = safe_div(FP, FP + TN)
    fnr = safe_div(FN, FN + TP)
    accuracy = safe_div(TP + TN, TP + TN + FP + FN)
    csi = safe_div(TP, TP + FP + FN)

    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    abs_error = df["abs_error"] if "abs_error" in df.columns else (df[target_col] - df[pred_col]).abs()

    return {
        "precision": precision, "recall": recall, "specificity": specificity,
        "accuracy": accuracy, "fpr": fpr, "fnr": fnr, "f1": f1, "csi": csi,
        "mean_abs_error": abs_error.mean(), "median_abs_error": abs_error.median(),
    }


def bootstrap_binary_metrics(
    df,
    event_col="DisNo.",
    thresholds=(0, 15),
    target_col="perc_affected_pop_grid_region",
    pred_col="prediction_perc",
    model_name=None,
    n_boot=1000,
    ci=95,
    random_state=None,
):
    """Point estimates plus event-level bootstrap CIs, one row per threshold.

    `event_col` is the unit resampled - the cyclone/event identifier, never the
    row index. Returns `<metric>`, `<metric>_ci_low`, `<metric>_ci_high` and
    `<metric>_n_undefined_boot` for each metric in METRIC_NAMES.
    """
    if event_col not in df.columns:
        raise KeyError(
            f"{event_col!r} not in the predictions; bootstrap must resample whole "
            f"events, so an event identifier is required (columns: {list(df.columns)[:12]})"
        )

    df = df.copy()
    if "abs_error" not in df.columns:
        df["abs_error"] = (df[target_col] - df[pred_col]).abs()

    rng = np.random.default_rng(random_state)
    events = df[event_col].unique()
    n_events = len(events)
    # Grouping once keeps the replicate loop to a concat of pre-built frames
    grouped = {event: group for event, group in df.groupby(event_col)}

    results = []
    for threshold in thresholds:
        point = compute_metrics_at_threshold(
            df, threshold, target_col, pred_col, nan_on_undefined=False)
        point["model"] = model_name
        point["threshold"] = threshold
        point["n_events"] = n_events
        point["n_rows"] = len(df)

        boot_values = {k: [] for k in METRIC_NAMES}
        n_undefined = {k: 0 for k in METRIC_NAMES}

        for _ in range(n_boot):
            sampled = rng.choice(events, size=n_events, replace=True)
            boot_df = pd.concat([grouped[e] for e in sampled], ignore_index=True)
            m = compute_metrics_at_threshold(
                boot_df, threshold, target_col, pred_col, nan_on_undefined=True)
            for k in METRIC_NAMES:
                if pd.isna(m[k]):
                    n_undefined[k] += 1
                else:
                    boot_values[k].append(m[k])

        alpha = (100 - ci) / 2
        for k in METRIC_NAMES:
            values = np.asarray(boot_values[k], dtype="float64")
            if values.size == 0:
                point[f"{k}_ci_low"] = np.nan
                point[f"{k}_ci_high"] = np.nan
            else:
                point[f"{k}_ci_low"] = float(np.percentile(values, alpha))
                point[f"{k}_ci_high"] = float(np.percentile(values, 100 - alpha))
            point[f"{k}_n_undefined_boot"] = n_undefined[k]

        results.append(point)

    return pd.DataFrame(results)


def format_ci(row, metric, decimals=3):
    """"0.412 [0.350, 0.478]" - a metric with its interval, for tables."""
    lo, hi = row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    if pd.isna(lo) or pd.isna(hi):
        return f"{row[metric]:.{decimals}f} [n/a]"
    return f"{row[metric]:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
