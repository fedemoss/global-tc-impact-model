#!/usr/bin/env python3
"""Joint hyperparameter search for both stages of the two-stage impact model.

Writes the winning configuration to `HYPERPARAMETERS_PATH`
(`data/model_hyperparameters.json` by default), which
`src/models/two_stage_xgb.py` loads automatically - so tuning and production use
the same numbers without anyone copying them by hand.

Design, and why it is this way
------------------------------

**Both stages are searched together.** Tuning the classifier first and the
regressor afterwards would miss their interaction: how aggressively stage 1
flags cells changes which rows stage 2 ever sees, so the best regressor depends
on the classifier it sits behind. The search space therefore spans both stages
plus the model's own class-balancing ratios (`u1`, `u2`) and the stage-1
decision threshold.

**Folds are split by event, never by grid cell.** Grid cells within one cyclone
are strongly correlated - same storm, same country, often adjacent land. Random
row-wise folds would put near-duplicate rows on both sides of the split and
report a score the model cannot reproduce on a genuinely unseen storm.
`StratifiedGroupKFold` on `DisNo.` keeps every event whole and keeps the
severity mix even across folds.

**A held-out set of events is scored exactly once.** Hyperparameters are chosen
on cross-validated folds; the held-out events are touched only at the end. The
cross-validated score is optimistic by construction (it is what was optimised),
so the held-out number is the one to quote.

**Metrics are computed on pooled out-of-fold predictions, aggregated to ADM1.**
Impact is reported at admin level, not per grid cell, so the metric is computed
where the labels actually live. Pooling across folds rather than averaging
per-fold scores avoids unstable precision/recall on folds that contain few
positive events.

**Random search by default.** The space below has ~17 dimensions; an exhaustive
grid over it is not feasible, and random search covers the important dimensions
far better for the same budget. `--search grid` is available for a small,
deliberately reduced space.

Usage
-----
    python test/hyperparameter_search.py --n-iter 80 --n-events 200
    python test/hyperparameter_search.py --dry-run            # tiny, for a smoke test
    python test/hyperparameter_search.py --search grid --n-jobs 4

Nothing is written until the search completes; pass `--output` to write
somewhere other than the configured path.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    f1_score, mean_absolute_error, precision_score, recall_score,
)
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:  # scikit-learn < 0.24
    from sklearn.model_selection import GroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES, HYPERPARAMETERS_PATH, INPUT_DIR
from src.models.two_stage_xgb import TwoStageXGBoost

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

TARGET_NAME = "perc_affected_pop_grid_region"
RANDOM_STATE = 0

# The two thresholds the model itself is built around: 0 separates "affected at
# all" (stage 1's target) and 15 separates "high impact" (the cutoff stage 2's
# balancing uses, and the paper's "highly affected" class).
IMPACT_THRESHOLD_LOW = 0
IMPACT_THRESHOLD_HIGH = 15


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
# Stage 1 ("clf_"), stage 2 ("reg_"), the balancing ratios and the decision
# threshold, all drawn jointly.
PARAM_SPACE = {
    "clf_learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf_max_depth": [3, 4, 5, 6],
    "clf_min_child_weight": [1, 4, 8],
    "clf_gamma": [0, 0.5, 1, 3],
    "clf_subsample": [0.6, 0.8, 1.0],
    "clf_colsample_bytree": [0.6, 0.8, 1.0],
    "clf_n_estimators": [100, 200, 400],
    "reg_learning_rate": [0.01, 0.03, 0.05, 0.1],
    "reg_max_depth": [3, 4, 5, 6],
    "reg_min_child_weight": [1, 4, 8],
    "reg_gamma": [0, 0.5, 1, 3],
    "reg_subsample": [0.6, 0.8, 1.0],
    "reg_colsample_bytree": [0.6, 0.8, 1.0],
    "reg_n_estimators": [100, 200, 400],
    # Undersampling ratios: how many majority rows are kept per minority row in
    # each stage (TwoStageXGBoost.oversample keeps every minority row and
    # downsamples the majority to u x that count, despite the method's name).
    # They are part of the model's design, not a preprocessing detail: the
    # classes are very unbalanced, so how hard each stage is rebalanced trades
    # recall against precision directly, and interacts with clf_threshold.
    # u = 1 is a fully balanced training set; larger keeps the data closer to
    # its natural imbalance.
    "u1": [1, 2, 3, 5, 8, 10],   # stage 1, affected vs not affected
    "u2": [1, 2, 3, 5, 8, 10],   # stage 2, high impact vs low impact
    "clf_threshold": [0.30, 0.40, 0.50, 0.60, 0.70],
}

# A deliberately small space for --search grid, which is exhaustive and so can
# only afford a few dimensions.
# An exhaustive grid can only afford a few dimensions, so the ones kept here are
# those the model is most sensitive to - including both undersampling ratios,
# which are searched rather than pinned. This is 2*2*2*2*6*6*3 = 1728
# candidates, i.e. 8640 two-stage fits at the default 5 folds: run it with
# --n-jobs, or prefer the random search, which covers the same space far more
# cheaply.
PARAM_GRID_SMALL = {
    "clf_learning_rate": [0.01, 0.05],
    "clf_max_depth": [3, 5],
    "reg_learning_rate": [0.01, 0.05],
    "reg_max_depth": [3, 5],
    "u1": [1, 2, 3, 5, 8, 10],
    "u2": [1, 2, 3, 5, 8, 10],
    "clf_threshold": [0.4, 0.5, 0.6],
}
# Held fixed across the grid, at the production defaults, so the exhaustive
# sweep stays tractable. The random search varies all of these.
GRID_DEFAULTS = {
    "clf_min_child_weight": 1, "clf_gamma": 0.5, "clf_subsample": 0.8,
    "clf_colsample_bytree": 0.8, "clf_n_estimators": 200,
    "reg_min_child_weight": 1, "reg_gamma": 0.5, "reg_subsample": 0.8,
    "reg_colsample_bytree": 0.8, "reg_n_estimators": 200,
}


def xgb_params_from_candidate(cand, nthread=1):
    """Split one flat candidate into the two stages' XGBoost keyword dicts."""
    common = {"booster": "gbtree", "verbosity": 0, "random_state": RANDOM_STATE, "nthread": nthread}
    clf_params = {
        **common, "objective": "binary:logistic", "eval_metric": "logloss",
        "learning_rate": cand["clf_learning_rate"], "max_depth": int(cand["clf_max_depth"]),
        "min_child_weight": cand["clf_min_child_weight"], "gamma": cand["clf_gamma"],
        "subsample": cand["clf_subsample"], "colsample_bytree": cand["clf_colsample_bytree"],
        "n_estimators": int(cand["clf_n_estimators"]),
    }
    reg_params = {
        **common, "objective": "reg:squarederror", "eval_metric": "rmse",
        "learning_rate": cand["reg_learning_rate"], "max_depth": int(cand["reg_max_depth"]),
        "min_child_weight": cand["reg_min_child_weight"], "gamma": cand["reg_gamma"],
        "subsample": cand["reg_subsample"], "colsample_bytree": cand["reg_colsample_bytree"],
        "n_estimators": int(cand["reg_n_estimators"]),
    }
    return clf_params, reg_params


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(input_path):
    """Load the model input and apply the paper's event-selection filters.

    The same three constraints the production evaluation uses: more than 100
    people affected, the event reported subnationally, and not every GID_1 in
    the country flagged as affected (which carries no spatial information).
    """
    df = pd.read_parquet(input_path).drop_duplicates().reset_index(drop=True)

    c1 = df[df["Total Affected"] > 100]["DisNo."].unique()
    c2 = df[df["level"] != "ADM0"]["DisNo."].unique()
    proportions = (
        df[df["level"] != "ADM0"]
        .assign(is_affected=lambda x: x["Total Affected"] > 0)
        .groupby(["DisNo.", "GID_0", "GID_1"])["is_affected"].max()
        .groupby(["DisNo.", "GID_0"]).mean()
        .reset_index(name="proportion_affected_gid1")
    )
    c3 = proportions[proportions.proportion_affected_gid1 < 1]["DisNo."].unique()

    keep = set(c1) & set(c2) & set(c3)
    df = df[df["DisNo."].isin(keep)].drop_duplicates().reset_index(drop=True)
    logging.info(f"{df['DisNo.'].nunique()} events, {len(df):,} grid-cell rows after filtering")

    if df.empty:
        raise ValueError(
            f"No events survive the selection filters in {input_path}. "
            f"Of the input's events, {len(c1)} have >100 affected, {len(c2)} are reported "
            f"subnationally and {len(c3)} do not have every GID_1 affected. A small "
            f"single-country extract (e.g. the ATG/FJI/HTI development subset) will "
            f"usually fail all three - point --input at the full dataset."
        )
    return df


def event_severity_labels(df, target_name=TARGET_NAME):
    """Per-event severity class, used only to stratify splits - never a feature."""
    def categorize(x):
        if x == 0:
            return 0
        return 1 if x < IMPACT_THRESHOLD_HIGH else 2
    return df.groupby("DisNo.")[target_name].max().apply(categorize)


def split_events(df, severity, n_events, test_fraction, random_state=RANDOM_STATE):
    """A representative event subsample, split into a search pool and a held-out set."""
    all_events = df["DisNo."].unique().tolist()

    if 0 < n_events < len(all_events):
        events_for_search, _ = train_test_split(
            all_events, train_size=n_events, random_state=random_state,
            stratify=severity.loc[all_events],
        )
    else:
        events_for_search = all_events

    search_events, holdout_events = train_test_split(
        events_for_search, test_size=test_fraction, random_state=random_state,
        stratify=severity.loc[events_for_search],
    )
    return list(search_events), list(holdout_events)


def make_folds(df_search, severity, n_folds, random_state=RANDOM_STATE):
    """Event-grouped CV folds, returned as (train_events, val_events) pairs."""
    groups = df_search["DisNo."]
    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(df_search, df_search["DisNo."].map(severity), groups)
    else:
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(df_search, groups=groups)

    return [
        (df_search.iloc[tr]["DisNo."].unique().tolist(),
         df_search.iloc[va]["DisNo."].unique().tolist())
        for tr, va in split_iter
    ]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_predictions(df_pred, target_name=TARGET_NAME, return_grouped=False):
    """Metrics at ADM1 level, where impact is actually reported.

    Grid-cell percentages are converted to people, summed per (event, GID_1),
    and turned back into a percentage of that unit's population before being
    binned. Both the "affected at all" and "high impact" thresholds are scored.
    """
    df = df_pred.copy()
    df["prediction_perc"] = df["prediction_perc"].clip(lower=0, upper=100)
    df["prediction_ppl"] = df["prediction_perc"] * df["population"] / 100
    df["actual_ppl"] = df[target_name] * df["population"] / 100
    df = df.drop_duplicates(subset=["DisNo.", "id"])

    grouped = (
        df.groupby(["DisNo.", "GID_0", "GID_1"])
        .agg(prediction_ppl=("prediction_ppl", "sum"),
             actual_perc=(target_name, "max"),
             population=("population", "sum"))
        .reset_index()
    )
    grouped = grouped[grouped["population"] > 0]
    grouped["reported"] = grouped["actual_perc"].clip(upper=100)
    grouped["predicted"] = (100 * grouped["prediction_ppl"] / grouped["population"]).clip(upper=100)
    grouped = grouped.dropna(subset=["reported", "predicted"])

    if grouped.empty:
        empty = {"mae": float("nan"), "n_adm1": 0}
        for suffix in ("", "_high"):
            empty.update({f"precision{suffix}": 0.0, f"recall{suffix}": 0.0, f"f1{suffix}": 0.0})
        return (empty, grouped) if return_grouped else empty

    bins = [-np.inf, IMPACT_THRESHOLD_LOW, IMPACT_THRESHOLD_HIGH, np.inf]
    grouped["reported_cat"] = pd.cut(grouped["reported"], bins=bins, labels=[0, 1, 2]).astype(int)
    grouped["predicted_cat"] = pd.cut(grouped["predicted"], bins=bins, labels=[0, 1, 2]).astype(int)

    metrics = {
        "mae": mean_absolute_error(grouped["reported"], grouped["predicted"]),
        "n_adm1": len(grouped),
    }
    for suffix, cutoff in [("", 1), ("_high", 2)]:
        y_true = (grouped["reported_cat"] >= cutoff).astype(int)
        y_pred = (grouped["predicted_cat"] >= cutoff).astype(int)
        metrics[f"precision{suffix}"] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f"recall{suffix}"] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f"f1{suffix}"] = f1_score(y_true, y_pred, zero_division=0)
    return (metrics, grouped) if return_grouped else metrics


def run_two_stage(df_all, features, train_events, test_events, cand, nthread=1):
    """One fit/predict of the production model under one candidate configuration."""
    clf_params, reg_params = xgb_params_from_candidate(cand, nthread=nthread)
    model = TwoStageXGBoost(
        clf_params=clf_params, reg_params=reg_params, features=features,
        target_name=TARGET_NAME, u1=int(cand["u1"]), u2=int(cand["u2"]),
        clf_threshold=float(cand["clf_threshold"]),
        use_tuned=False,  # never let a previously-saved file leak into the search
    )
    df_train = df_all[df_all["DisNo."].isin(train_events)]
    df_test = df_all[df_all["DisNo."].isin(test_events)]
    return model.train_and_predict(df_train, df_test)


def score_candidate(cand, df_search, folds, features, nthread=1):
    """Pooled out-of-fold metrics for one candidate."""
    oof = []
    for train_events, val_events in folds:
        try:
            oof.append(run_two_stage(df_search, features, train_events, val_events, cand, nthread))
        except Exception as e:  # a candidate that cannot fit should not kill the search
            logging.warning(f"candidate failed on a fold ({e}); scoring it as 0")
            return {**cand, "mae": float("nan"), "f1": 0.0, "f1_high": 0.0,
                    "precision": 0.0, "recall": 0.0, "precision_high": 0.0,
                    "recall_high": 0.0, "n_adm1": 0, "failed": True}
    metrics = evaluate_predictions(pd.concat(oof, axis=0))
    return {**cand, **metrics, "failed": False}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def build_candidates(search, n_iter, random_state=RANDOM_STATE):
    if search == "grid":
        candidates = [{**GRID_DEFAULTS, **c} for c in ParameterGrid(PARAM_GRID_SMALL)]
        logging.info(f"exhaustive grid over {len(PARAM_GRID_SMALL)} dimensions: {len(candidates)} candidates")
        return candidates
    candidates = list(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=random_state))
    logging.info(f"random search over {len(PARAM_SPACE)} dimensions: {len(candidates)} candidates")
    return candidates


MIN_EVENTS = 6  # below this, grouped folds cannot be formed meaningfully


def main(args):
    df = load_dataset(args.input)
    severity = event_severity_labels(df)

    n_events_available = df["DisNo."].nunique()
    if n_events_available < MIN_EVENTS:
        logging.error(
            f"only {n_events_available} events available; at least {MIN_EVENTS} are needed to "
            f"form event-grouped folds and a held-out set. Point --input at the full dataset."
        )
        return 1

    search_events, holdout_events = split_events(
        df, severity, args.n_events, args.test_fraction, args.random_state)
    logging.info(f"search pool: {len(search_events)} events | held-out: {len(holdout_events)} events")

    df_search = df[df["DisNo."].isin(search_events)].copy()
    df_holdout = df[df["DisNo."].isin(holdout_events)].copy()

    n_folds = min(args.n_folds, max(2, len(search_events)))
    folds = make_folds(df_search, severity, n_folds, args.random_state)
    for i, (tr, va) in enumerate(folds):
        logging.info(f"  fold {i}: {len(tr)} train events, {len(va)} val events")

    candidates = build_candidates(args.search, args.n_iter, args.random_state)
    logging.info(f"{len(candidates)} candidates x {len(folds)} folds = "
                 f"{len(candidates) * len(folds)} two-stage fits")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(score_candidate)(c, df_search, folds, FEATURES, args.nthread) for c in candidates
    )
    results_df = pd.DataFrame(results).sort_values(args.select_on, ascending=False).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_path = args.output.with_name(args.output.stem + "_search_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"all candidate scores -> {results_path}")

    if results_df["failed"].all():
        logging.error("every candidate failed; nothing written")
        return 1

    param_keys = list(PARAM_SPACE.keys())
    best = results_df.iloc[0]
    best_cand = {k: best[k] for k in param_keys if k in results_df.columns}
    for k in ("clf_max_depth", "clf_min_child_weight", "clf_n_estimators",
              "reg_max_depth", "reg_min_child_weight", "reg_n_estimators", "u1", "u2"):
        if k in best_cand:
            best_cand[k] = int(best_cand[k])
    best_cand["clf_threshold"] = float(best_cand["clf_threshold"])

    logging.info(f"best candidate by {args.select_on}:\n{json.dumps(best_cand, indent=2, default=str)}")

    # The held-out events have not influenced anything above; this is the only
    # time they are used, and the resulting number is the one worth quoting.
    holdout_pred = run_two_stage(
        pd.concat([df_search, df_holdout], axis=0), FEATURES,
        search_events, holdout_events, best_cand, nthread=args.nthread or 0,
    )
    holdout_metrics = evaluate_predictions(holdout_pred)
    logging.info(f"held-out metrics (evaluated once):\n{json.dumps(holdout_metrics, indent=2, default=str)}")

    clf_params, reg_params = xgb_params_from_candidate(best_cand, nthread=args.nthread or 0)
    payload = {
        "clf_params": clf_params,
        "reg_params": reg_params,
        "u1": best_cand["u1"],
        "u2": best_cand["u2"],
        "clf_threshold": best_cand["clf_threshold"],
        "candidate": best_cand,
        "selected_on": args.select_on,
        "cv_metrics": {k: (None if pd.isna(best[k]) else float(best[k]))
                       for k in ("mae", "precision", "recall", "f1",
                                 "precision_high", "recall_high", "f1_high", "n_adm1")
                       if k in results_df.columns},
        "holdout_metrics": {k: (None if pd.isna(v) else float(v)) for k, v in holdout_metrics.items()},
        "search": {
            "input": str(args.input), "search": args.search, "n_candidates": len(candidates),
            "n_events": len(search_events) + len(holdout_events),
            "n_search_events": len(search_events), "n_holdout_events": len(holdout_events),
            "n_folds": len(folds), "random_state": args.random_state,
        },
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"chosen hyperparameters -> {args.output} (picked up automatically by TwoStageXGBoost)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=INPUT_DIR / "model_input_dataset" / "training_dataset.parquet",
                   help="model input parquet (default: the pipeline's own output)")
    p.add_argument("--output", type=Path, default=HYPERPARAMETERS_PATH,
                   help="where to write the chosen hyperparameters JSON")
    p.add_argument("--search", choices=["random", "grid"], default="random",
                   help="random search over the full space (default) or an exhaustive small grid")
    p.add_argument("--n-iter", type=int, default=80, help="random-search candidates")
    p.add_argument("--n-events", type=int, default=200,
                   help="events sampled for the whole search; 0 uses all of them")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="fraction of sampled events held out and scored once")
    p.add_argument("--n-folds", type=int, default=5, help="event-grouped CV folds")
    p.add_argument("--n-jobs", type=int, default=4, help="candidates evaluated in parallel")
    p.add_argument("--nthread", type=int, default=1,
                   help="xgboost threads per fit; keep n_jobs * nthread <= cores")
    p.add_argument("--select-on", default="f1",
                   help="metric that decides the winner (pre-specify it; do not eyeball several)")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--dry-run", action="store_true",
                   help="tiny budget for a smoke test: 2 candidates, 2 folds, 20 events")
    args = p.parse_args(argv)
    if args.dry_run:
        args.n_iter, args.n_folds, args.n_events, args.n_jobs = 2, 2, 20, 1
    return args


if __name__ == "__main__":
    sys.exit(main(parse_args()))
