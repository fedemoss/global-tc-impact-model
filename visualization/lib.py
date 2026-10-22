"""Loading and plotting helpers shared by the notebooks in this folder.

Kept out of the notebooks so that the figures and the tables are built from
one definition of what a model output is and how a prediction is categorised,
and so that the plotting, which is the bulk of the code, does not crowd out
the analysis.
"""
import os

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

#: Variants of the model itself, as opposed to the cross-validation strategy.
#: Each is a folder suffix under model_output, and each is a different model
#: of the same administrative units.
MODEL_VARIANTS = {
    "grid-fitted": "2-stage-XGBoost_global_adm1",
    "ADM1-fitted": "2-stage-XGBoost_global_adm1_trained",
    "no prev. events": "2-stage-XGBoost_global_grid_no_N_events_5_years",
}


#: Appended to the model's output folders, so that a re-run kept under its own
#: name can be read with these notebooks while the folders an earlier run wrote
#: stay where they are. Empty reads the original run.
RUN_SUFFIX = os.environ.get("TC_RUN_SUFFIX", "")


def output_folder(model_name, strategy="global", level="adm1"):
    # The rule baselines have no fitted parameters and are not re-run with the
    # model, so they keep their single set of folders.
    suffix = RUN_SUFFIX if model_name == "2-stage-XGBoost" else ""
    return OUTPUT_DIR / f"model_output/{model_name}_{strategy}_{level}{suffix}"


def load_model(model_name, strategy="global", level="adm1"):
    """One model's predictions, or None if it has not been scored.

    A fitted model writes a file per fold; a rule has no folds and is written
    whole, so both layouts are accepted.
    """
    folder = output_folder(model_name, strategy, level)
    bundle = folder.with_suffix(".parquet")
    if bundle.exists():
        df = pd.read_parquet(bundle)
    else:
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


# --------------------------------------------------------------------------
# Paper figures
# --------------------------------------------------------------------------
# The categorical axis labels and the figure styling follow the paper, so a
# figure built here is the same one the paper carries.
CATEGORY_ORDER = [r"$\it{no-imp.}$", r"$\it{low}$", r"$\it{high}$"]
REPORTED_BANDS = ["0.0 %", "0.0-15.0%", ">15.0 %"]
MODEL_COLORS = ["#A0B0F3", "#AEC6CF", "#E6E6FA", "#E24916"]
PAPER_MODEL_NAMES = ["historical", "windspeed-exposed", "windspeed-historical",
                     r"$\bf{2\text{-}stage\text{-}XGBoost}$"]


def categorize_impact(val):
    """The predicted/reported category, labelled as the figures label it."""
    if val == 0:
        return CATEGORY_ORDER[0]
    if 0 < val < IMPACT_THRESHOLD_HIGH:
        return CATEGORY_ORDER[1]
    return CATEGORY_ORDER[2]


def categorize_reported(val):
    """The reported band, used to split the error distribution."""
    if val == 0:
        return REPORTED_BANDS[0]
    if 0 < val < IMPACT_THRESHOLD_HIGH:
        return REPORTED_BANDS[1]
    return REPORTED_BANDS[2]


def plot_categorical_confusion_matrices(dfs_adm1, model_names, savefig=None):
    """One matrix per model, rows reported and columns predicted.

    Colour is normalised within a row, so a row's largest cell is saturated
    whatever the row totals are, and the three rows carry separate scales --
    otherwise the no-impact row, two orders of magnitude larger, flattens the
    rest to white.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix

    matrices = []
    for df in dfs_adm1:
        y_true = df["reported"].apply(categorize_impact)
        y_pred = df["predicted"].apply(categorize_impact)
        matrices.append(np.flipud(
            confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)))

    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    axes = np.atleast_1d(axes)

    row_colormaps = [
        LinearSegmentedColormap.from_list("pastel_reds", ["#ffe5e5", "#ff9999"]),
        LinearSegmentedColormap.from_list("pastel_yellow_gold", ["#fff9cc", "#ffe066"]),
        LinearSegmentedColormap.from_list("pastel_greens", ["#e6ffe6", "#99e699"]),
    ]

    for i, ax in enumerate(axes):
        data = matrices[i]
        ax.set_title(model_names[i], size=15)

        normed = np.zeros_like(data, dtype=float)
        for r in range(data.shape[0]):
            row_max = data[r].max()
            normed[r] = data[r] / row_max if row_max else data[r]

        rgb = np.zeros((3, 3, 4))
        for r in range(3):
            for c in range(3):
                rgb[r, c] = row_colormaps[r](normed[r, c])
        ax.imshow(rgb, aspect="equal")

        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(CATEGORY_ORDER, fontsize=12)
        ax.set_yticklabels(CATEGORY_ORDER[::-1] if i == 0 else [], fontsize=12)
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        ax.grid(False)

        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{data[r, c]}", ha="center", va="center",
                        fontsize=10, color="black")
                if r + c == 2:          # the diagonal, after the vertical flip
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           linewidth=2.5, edgecolor="black",
                                           facecolor="none"))
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.text(0.05, 0.5, "True Category", va="center", rotation="vertical", fontsize=14)
    fig.text(0.5, -0.1, "Predicted Category", ha="center", fontsize=14)
    fig.text(0.05, -0.22,
             r"$\it{0\%}$ : $\it{no-imp.}$   |   "
             r"$\it{0\%-15\%}$ : $\it{low}$   |   "
             r"$\it{\geq 15\%}$ : $\it{high}$",
             ha="left", fontsize=13,
             bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.4"))
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight", pad_inches=0.5)
    return fig, matrices


def plot_models_violin(ax, dfs, model_names, colors=None, show_legend=True,
                       show_ylabel=True, show_info=True):
    """Absolute error per model, split by the reported band.

    A model is also shown over every row, as the fourth group, so the bands
    can be read against the whole.
    """
    import numpy as np
    import seaborn as sns

    colors = colors or MODEL_COLORS
    category_order = REPORTED_BANDS + ["all-data"]

    frames = []
    for df, name in zip(dfs, model_names):
        temp = df.copy()
        if "abs_error" not in temp.columns:
            temp["abs_error"] = np.abs(temp["reported"] - temp["predicted"])
        temp["category"] = temp["reported"].apply(categorize_reported)
        temp["Model"] = name
        everything = temp.copy()
        everything["category"] = "all-data"
        frames += [temp, everything]
    combined = pd.concat(frames, ignore_index=True)

    sns.violinplot(
        data=combined, x="category", y="abs_error", hue="Model",
        order=category_order, hue_order=model_names,
        palette=colors, split=False, ax=ax,
        cut=0, fill=True, linewidth=0.3, inner="box",
        inner_kws={"linewidth": 1.5, "alpha": 0.8, "box_width": 5},
        legend=show_legend,
    )

    for i in range(1, len(category_order)):
        ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("Absolute Error (%)" if show_ylabel else "", size=12)
    if show_legend:
        ax.legend(title="Model", loc="upper left")

    if show_info:
        every = pd.concat(dfs, ignore_index=True)
        every["category"] = every["reported"].apply(categorize_reported)
        total = len(every)
        info = [f"{100 * (every['category'] == c).sum() / total:.1f}%"
                for c in REPORTED_BANDS] + ["100%"]
        ax.set_xticks(range(len(category_order)))
        ax.set_xticklabels([f"{c}\n({p})" for c, p in zip(category_order, info)],
                           fontsize=11)
    return ax


def load_variant(folder_name, level="adm1"):
    """One model variant's predictions, by output folder name.

    A variant fitted at grid level is read from its ADM1 aggregation, so every
    variant is compared on the administrative units impact is reported at.
    """
    from src.config import OUTPUT_DIR

    base = OUTPUT_DIR / "model_output"
    # Every variant is a fitting of the model itself, so each follows the run.
    folder_name = f"{folder_name}{RUN_SUFFIX}"
    # A grid-fitted variant is compared through its ADM1 aggregation, which
    # the training run writes once it has finished. Falling back to the grid
    # output would put cells and administrative units in the same comparison.
    if folder_name.endswith("_grid") or "_grid_" in folder_name:
        aggregated = base / folder_name.replace("_grid", "_adm1", 1)
        if not (aggregated.exists() or aggregated.with_suffix(".parquet").exists()):
            return None
        folder_name = aggregated.name

    folder = base / folder_name
    bundle = folder.with_suffix(".parquet")
    if bundle.exists():
        return _with_comparison_columns(pd.read_parquet(bundle))
    files = sorted(folder.glob("predictions_event_*.csv"))
    if not files:
        return None
    return _with_comparison_columns(
        pd.concat([pd.read_csv(f) for f in files], ignore_index=True))


def _with_comparison_columns(df):
    """Give a variant the reported/predicted pair every comparison here reads.

    A run fitted at ADM1 is scored where it was fitted, so it carries the
    model's own column names. A grid run reaches ADM1 through the aggregation,
    which renames them. Both are a reported and a predicted share of a
    population, and both are bounded by it.
    """
    if "reported" in df.columns or "perc_affected_pop_grid_region" not in df.columns:
        return df
    df = df.copy()
    df["reported"] = df["perc_affected_pop_grid_region"].clip(0, 100)
    df["predicted"] = df["prediction_perc"].clip(0, 100)
    return df


def load_variants(variants=None, level="adm1"):
    """{label: frame} for whichever model variants have been scored."""
    out = {}
    for label, folder in (variants or MODEL_VARIANTS).items():
        df = load_variant(folder, level=level)
        if df is not None:
            out[label] = df
    return out


# --------------------------------------------------------------------------
# Walk-forward against leave-one-out, over time
# --------------------------------------------------------------------------
COLOR_WALK_FORWARD = "#817F7D"
COLOR_LOOCV = "#E24916"


def attach_event_time(df):
    """Add `year` and `month` from the EM-DAT record of each event."""
    import pandas as pd

    from src.config import INPUT_DIR

    emdat = pd.read_csv(INPUT_DIR / "EMDAT" / "emdat.csv")
    cols = ["DisNo.", "Start Year", "Start Month"]
    ev = emdat[cols].drop_duplicates().rename(
        columns={"Start Year": "year", "Start Month": "month"})
    out = df.drop(columns=[c for c in ("year", "month") if c in df.columns])
    out = out.merge(ev, on="DisNo.", how="left")
    # an event the registry no longer carries still has its year in the id
    missing = out["year"].isna()
    if missing.any():
        out.loc[missing, "year"] = (
            out.loc[missing, "DisNo."].astype(str).str[:4].astype(float))
    out["month"] = out["month"].fillna(1)
    return out


def rolling_metrics_with_bootstrap(df, event_col="sid", threshold=0,
                                   resolution="month", window_size=5,
                                   n_boot=30, ci=0.95, seed=None,
                                   from_year=2000):
    """Metrics over a trailing window, with intervals from resampling cyclones.

    Each point covers the `window_size` years ending at it. Resampling is over
    the cyclone rather than the row, since administrative units of one storm
    share a hazard field and a single report.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(seed)
    df = df[df["year"] >= from_year].copy()
    df["reported_bin"] = (df["reported"] > threshold).astype(int)
    df["pred_bin"] = (df["predicted"] > threshold).astype(int)

    year = df["year"].astype(int).astype(str)
    if resolution == "year":
        df["time_index"] = pd.to_datetime(year + "-01-01")
    elif resolution == "month":
        month = df["month"].fillna(1).astype(int).astype(str)
        df["time_index"] = pd.to_datetime(year + "-" + month + "-01")
    else:
        raise ValueError("resolution must be 'year' or 'month'")

    df = df.sort_values("time_index")
    bounds = lambda a: (
        (np.nan, np.nan) if not len(a[~np.isnan(a)])
        else (np.percentile(a[~np.isnan(a)], (1 - ci) / 2 * 100),
              np.percentile(a[~np.isnan(a)], (1 + ci) / 2 * 100)))

    rows = []
    for t in df["time_index"].dropna().unique():
        window = df[(df["time_index"] > t - pd.DateOffset(years=window_size))
                    & (df["time_index"] <= t)]
        if window.empty:
            continue
        events = window[event_col].unique()
        grouped = {e: g for e, g in window.groupby(event_col, observed=True)}

        prec, rec, f1 = [], [], []
        for _ in range(n_boot):
            sample = pd.concat(
                [grouped[e] for e in rng.choice(events, size=len(events), replace=True)],
                ignore_index=True)
            if sample["reported_bin"].sum() == 0:
                prec.append(np.nan)
            else:
                prec.append(precision_score(sample["reported_bin"],
                                            sample["pred_bin"], zero_division=0))
            rec.append(recall_score(sample["reported_bin"], sample["pred_bin"],
                                    zero_division=0))
            f1.append(f1_score(sample["reported_bin"], sample["pred_bin"],
                               zero_division=0))

        entry = {"time_index": t, "n_events_in_window": len(events)}
        for name, vals in (("precision", prec), ("recall", rec), ("f1", f1)):
            arr = np.asarray(vals, dtype=float)
            lo, hi = bounds(arr)
            entry[f"{name}_mean"] = np.nanmean(arr)
            entry[f"{name}_lower"] = lo
            entry[f"{name}_upper"] = hi
        rows.append(entry)

    return pd.DataFrame(rows)


def plot_rolling_comparison(walk, loocv, from_date="2005-01-01", savefig=None):
    """Precision and recall over time, walk-forward against leave-one-out.

    `walk` and `loocv` are {threshold: rolling-metrics frame}. Two thresholds
    across the columns, precision above recall down the rows.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    thresholds = sorted(walk)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=False)

    for col, t in enumerate(thresholds):
        for row, metric in enumerate(("precision", "recall")):
            a = ax[row, col]
            for frame, colour, label, width in (
                    (walk[t], COLOR_WALK_FORWARD, "Walk-Forward", 3),
                    (loocv[t], COLOR_LOOCV, "LOOCV", 2)):
                d = frame[frame["time_index"] >= from_date]
                a.plot(d["time_index"], d[f"{metric}_mean"], color=colour,
                       linewidth=width, alpha=0.7,
                       label=label if (row == 0 and col == 0) else None)
                a.fill_between(d["time_index"], d[f"{metric}_lower"],
                               d[f"{metric}_upper"], color=colour, alpha=0.2)
            a.set_ylim([0, 1])
            a.grid(True, linestyle="--", alpha=0.6)
            a.tick_params(labelsize=12)
            if row == 0:
                a.set_title(f"Threshold = {t}%", fontsize=15)
            if col == 0:
                a.set_ylabel(metric.capitalize(), fontsize=15)
            else:
                a.tick_params(left=False, labelleft=False)
    ax[0, 0].legend(loc="lower center", ncol=2, fontsize=13, frameon=True)

    fig.text(0.5, 0.04, "Time", ha="center", fontsize=15)
    for a in ax[1, :]:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    return fig


def plot_cumulative_events(df, from_year=2012, savefig=None):
    """Cyclones available to train on, accumulating over the record."""
    import matplotlib.pyplot as plt

    per_year = df.groupby("year")["DisNo."].nunique()
    cumulative = per_year.cumsum()
    cumulative = cumulative[cumulative.index >= from_year]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(cumulative.index, cumulative.values, marker="o", linestyle="-",
            color="#C5C4C3", linewidth=2, label="Training Data")
    ax.set_ylabel("TCs", fontsize=15)
    ax.set_xlabel("Year", fontsize=15)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=15, frameon=True)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches="tight", pad_inches=0.2)
    return fig, cumulative
