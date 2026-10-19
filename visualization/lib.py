"""Loading and plotting helpers shared by the notebooks in this folder.

Kept out of the notebooks so that the figures and the tables are built from
one definition of what a model output is and how a prediction is categorised,
and so that the plotting, which is the bulk of the code, does not crowd out
the analysis.
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
