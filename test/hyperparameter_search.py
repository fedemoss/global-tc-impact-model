#!/usr/bin/env python3
"""Joint hyperparameter search for both stages of the two-stage impact model.

Writes the winning configuration to `HYPERPARAMETERS_PATH`
(`data/model_hyperparameters.json` by default), which
`src/models/two_stage_xgb.py` loads automatically - so tuning and production
use the same numbers without anyone copying them by hand.

Design, and why it is this way
------------------------------

**The decision threshold is never searched - it is swept for free.** Stage 1's
cutoff (`clf_threshold`) changes nothing that is trained: stage 1 is fitted
without it and stage 2 is fitted on *observed* positives, so the threshold
only gates which test cells receive the stage-2 estimate at prediction time.
Every candidate is therefore fitted once, its out-of-fold probabilities and
stage-2 estimates are kept, and ALL thresholds are scored from them. Earlier
versions sampled the threshold as a search dimension, which paid ten XGBoost
fits to compare two cutoffs that one fit and two cheap scoring passes settle.

**Two phases, matched to what each dimension is.**

* ``--search random`` (default) draws candidates over the fourteen tree
  parameters and the two balancing ratios. High-dimensional, plateau-shaped:
  random search is the right tool, and its job is to land on the plateau.
* ``--search ratios`` grids u1 x u2 exhaustively with the tree parameters
  frozen (from an earlier winner's JSON, or the built-in defaults). The
  ratios and the threshold set the precision/recall operating point the
  paper's story rests on, and they are only two training dimensions - mapping
  the whole surface is affordable and defensible where a sampled point is
  not. The grid is also factorized: u1 touches only stage 1 and u2 only
  stage 2, so a fold needs one classifier per u1 and one regressor per u2
  (|u1| + |u2| fits), not |u1| x |u2| pairs.

**Folds are split by physical cyclone, never by grid cell or country-event
record.** Grid cells within one cyclone are strongly correlated - same storm,
same country, often adjacent land - and a storm that hits several countries
contributes several EM-DAT `DisNo.` records. Random row-wise folds would put
near-duplicate rows on both sides of the split, and `DisNo.`-wise folds would
put the same storm on both sides through different countries.
`StratifiedGroupKFold` on the IBTrACS `sid` keeps every cyclone whole and
keeps the severity mix even across folds.

**A held-out set of cyclones is scored exactly once**, after selection. The
cross-validated score is optimistic by construction (it is what was
optimised), so the held-out number is the one to quote. The holdout's full
threshold curve is stored for the response-surface figure, but the quotable
value is the one at the selected threshold.

**Metrics are computed on pooled out-of-fold predictions, aggregated to
ADM1**, where impact is actually reported, at both the "affected at all" (0%)
and "highly affected" (15%) thresholds. Selection is on `--select-on`
(default `f1_high`, the paper's objective); the top candidates' spread is
written out so a winner inside the noise band is visible as such.

Usage
-----
    python test/hyperparameter_search.py --n-iter 80 --n-events 200
    python test/hyperparameter_search.py --search ratios
    python test/hyperparameter_search.py --dry-run     # tiny, for a smoke test

Nothing is written until the search completes; pass `--output` to write
somewhere other than the configured path.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterSampler, train_test_split
from xgboost import XGBClassifier, XGBRegressor

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

# The two thresholds the model itself is built around: 0 separates "affected
# at all" (stage 1's target) and 15 separates "high impact" (strictly >15,
# the paper's category boundary).
IMPACT_THRESHOLD_LOW = 0
IMPACT_THRESHOLD_HIGH = 15

# Stage-1 cutoffs scored for every candidate. Not a search dimension: every
# value is evaluated from each candidate's stored probabilities.
CLF_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

# Balancing ratios. The last search selected u2 = 10 and threshold 0.70, both
# at the edge of the previous grids; these extend past them so a boundary
# optimum is either confirmed interior or exposed as censored. A large u
# saturates once it asks for more majority rows than exist - the saturation
# share is reported, and a saturated winner simply means "no undersampling".
U_VALUES = (1, 2, 3, 5, 8, 10, 15, 20)

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
    "u1": list(U_VALUES),
    "u2": list(U_VALUES),
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


def candidate_from_params(clf_params, reg_params):
    """The flat candidate encoding of two existing parameter dicts."""
    cand = {}
    for prefix, params in (("clf", clf_params), ("reg", reg_params)):
        for key in ("learning_rate", "max_depth", "min_child_weight", "gamma",
                    "subsample", "colsample_bytree", "n_estimators"):
            cand[f"{prefix}_{key}"] = params[key]
    return cand


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(input_path):
    """Load the model input and apply the paper's event-selection filters."""
    df = pd.read_parquet(input_path).drop_duplicates().reset_index(drop=True)

    c1 = df[df["Total Affected"] > 100]["DisNo."].unique()
    c2 = df[df["level"] != "ADM0"]["DisNo."].unique()
    proportions = (
        df[df["level"] != "ADM0"]
        .assign(is_affected=lambda x: x["Total Affected"] > 0)
        .groupby(["DisNo.", "GID_0", "GID_1"], observed=True)["is_affected"].max()
        .groupby(["DisNo.", "GID_0"], observed=True).mean()
        .reset_index(name="proportion_affected_gid1")
    )
    c3 = proportions[proportions.proportion_affected_gid1 < 1]["DisNo."].unique()

    keep = set(c1) & set(c2) & set(c3)
    df = df[df["DisNo."].isin(keep)].drop_duplicates().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No events survive the selection filters in {input_path}. "
            f"Of the input's events, {len(c1)} have >100 affected, {len(c2)} are reported "
            f"subnationally and {len(c3)} do not have every GID_1 affected. A small "
            f"single-country extract will usually fail all three - point --input at the "
            f"full dataset."
        )
    if "sid" not in df.columns or df["sid"].isna().any():
        raise ValueError(
            "the grouping unit is the physical cyclone: every row needs a non-null "
            "'sid' (IBTrACS storm id) -- see the EM-DAT input schema in the README"
        )
    multi = df.groupby("DisNo.", observed=True)["sid"].nunique()
    if (multi > 1).any():
        raise ValueError(
            f"these DisNo. map to more than one sid and must be resolved first: "
            f"{sorted(multi[multi > 1].index.tolist())}"
        )
    logging.info(
        f"{df['sid'].nunique()} physical cyclones / {df['DisNo.'].nunique()} country-event "
        f"records / {len(df):,} grid-cell rows after filtering"
    )
    return df


def event_severity_labels(df, target_name=TARGET_NAME):
    """Per-cyclone severity class, used only to stratify splits - never a feature."""
    def categorize(x):
        if x == 0:
            return 0
        return 1 if x <= IMPACT_THRESHOLD_HIGH else 2
    return df.groupby("sid", observed=True)[target_name].max().apply(categorize)


def split_events(df, severity, n_events, test_fraction, random_state=RANDOM_STATE):
    """A representative cyclone subsample, split into a search pool and a held-out set."""
    all_events = df["sid"].unique().tolist()
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
    """Cyclone-grouped CV folds, returned as (train_sids, val_sids) pairs."""
    groups = df_search["sid"]
    if HAS_STRATIFIED_GROUP_KFOLD:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        split_iter = splitter.split(df_search, df_search["sid"].map(severity), groups)
    else:
        splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(df_search, groups=groups)
    return [
        (df_search.iloc[tr]["sid"].unique().tolist(),
         df_search.iloc[va]["sid"].unique().tolist())
        for tr, va in split_iter
    ]


# --------------------------------------------------------------------------
# Stage fits (the semantics of TwoStageXGBoost.train_and_predict, split so
# the two stages can be fitted and reused independently)
# --------------------------------------------------------------------------
def fit_stage1(df_train, features, clf_params, u1):
    """Classifier fitted on the u1-rebalanced training set."""
    df_train = df_train.copy()
    df_train["reported_bin"] = (df_train[TARGET_NAME] > 0).astype(int)
    n_minority = int(df_train["reported_bin"].sum())
    n_majority = int((df_train["reported_bin"] == 0).sum())
    balanced = TwoStageXGBoost.oversample(df_train, target="reported_bin", u=u1)
    clf = XGBClassifier(**clf_params)
    clf.fit(balanced[features], balanced["reported_bin"])
    return clf, u1 * n_minority > n_majority


def fit_stage2(df_train, features, reg_params, u2):
    """Regressor fitted on observed positives, u2-rebalanced - threshold-free."""
    df_high = df_train[df_train[TARGET_NAME] > 0].copy()
    # Strictly > 15, matching the paper's category definition.
    df_high["impact_high"] = (df_high[TARGET_NAME] > IMPACT_THRESHOLD_HIGH).astype(int)
    n_minority = int(df_high["impact_high"].sum())
    n_majority = int((df_high["impact_high"] == 0).sum())
    balanced = TwoStageXGBoost.oversample(df_high, target="impact_high", u=u2)
    reg = XGBRegressor(**reg_params)
    reg.fit(balanced[features], balanced[TARGET_NAME])
    return reg, u2 * n_minority > n_majority


# --------------------------------------------------------------------------
# Threshold-sweep evaluation
# --------------------------------------------------------------------------
# For one fitted candidate, predictions at every threshold share the same
# probabilities and the same stage-2 estimates; only the gate moves. Each test
# cell is binned by the highest threshold it clears, the stage-2
# people-estimates are summed per (ADM1 unit, bin) once, and a reverse
# cumulative sum over bins yields every threshold's ADM1 totals from that one
# groupby. Scoring all |CLF_THRESHOLDS| cutoffs therefore costs one
# aggregation, not one model fit each.

ADM1_KEYS = ["DisNo.", "GID_0", "GID_1"]


def oof_frame(df_test, proba, reg_pred):
    """The per-cell quantities every threshold shares, deduplicated per cell."""
    out = df_test[ADM1_KEYS + ["id", "population", TARGET_NAME]].copy()
    out["proba"] = np.asarray(proba, dtype="float64")
    out["ppl_reg"] = np.clip(np.asarray(reg_pred, dtype="float64"), 0, 100) * out["population"] / 100.0
    return out.drop_duplicates(subset=["DisNo.", "id"])


def score_thresholds(cells, thresholds=CLF_THRESHOLDS):
    """ADM1-level metrics at every stage-1 cutoff, from one aggregation.

    Returns a list (one dict per threshold) with precision/recall/F1 at the
    0% and 15% impact thresholds plus MAE and the pair count.
    """
    thresholds = np.asarray(thresholds, dtype="float64")
    # bin b: the cell clears thresholds[k] exactly for k < b
    cells = cells.assign(_bin=np.searchsorted(thresholds, cells["proba"].to_numpy(), side="right"))

    grouped = (
        cells.groupby(ADM1_KEYS, observed=True)
        .agg(population=("population", "sum"), actual_perc=(TARGET_NAME, "max"))
    )
    grouped = grouped[grouped["population"] > 0]
    by_bin = (
        cells.groupby(ADM1_KEYS + ["_bin"], observed=True)["ppl_reg"].sum()
        .unstack("_bin", fill_value=0.0)
        .reindex(columns=range(len(thresholds) + 1), fill_value=0.0)
        .reindex(grouped.index, fill_value=0.0)
    )
    # Threshold k admits bins > k: a reverse cumulative sum, dropping bin 0.
    ppl = by_bin.to_numpy()[:, ::-1].cumsum(axis=1)[:, ::-1][:, 1:]  # (n_adm1, K)

    population = grouped["population"].to_numpy()[:, None]
    predicted = np.clip(100.0 * ppl / population, 0, 100)            # (n_adm1, K)
    reported = np.clip(grouped["actual_perc"].to_numpy(), 0, 100)

    results = []
    for k, threshold in enumerate(thresholds):
        row = {"clf_threshold": float(threshold), "n_adm1": int(len(grouped)),
               "mae": float(np.abs(reported - predicted[:, k]).mean())}
        for suffix, cut in (("", IMPACT_THRESHOLD_LOW), ("_high", IMPACT_THRESHOLD_HIGH)):
            y_true = reported > cut
            y_pred = predicted[:, k] > cut
            tp = int(np.sum(y_true & y_pred))
            fp = int(np.sum(~y_true & y_pred))
            fn = int(np.sum(y_true & ~y_pred))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            row[f"precision{suffix}"] = precision
            row[f"recall{suffix}"] = recall
            row[f"f1{suffix}"] = f1
        results.append(row)
    return results


# --------------------------------------------------------------------------
# Candidate evaluation
# --------------------------------------------------------------------------
def _fold_cells(df_search, features, fold, cand, nthread):
    """Fit one candidate on one fold; return the per-cell OOF frame."""
    train_sids, val_sids = fold
    df_train = df_search[df_search["sid"].isin(train_sids)]
    df_test = df_search[df_search["sid"].isin(val_sids)]
    assert not set(df_train["sid"].unique()) & set(df_test["sid"].unique())
    clf_params, reg_params = xgb_params_from_candidate(cand, nthread)
    clf, sat1 = fit_stage1(df_train, features, clf_params, int(cand["u1"]))
    reg, sat2 = fit_stage2(df_train, features, reg_params, int(cand["u2"]))
    cells = oof_frame(df_test, clf.predict_proba(df_test[features])[:, 1],
                      reg.predict(df_test[features]))
    return cells, sat1, sat2


def score_candidate(cand, df_search, folds, features, select_on, nthread=1):
    """Pooled out-of-fold metrics for one candidate, at its best threshold."""
    try:
        parts = [_fold_cells(df_search, features, fold, cand, nthread) for fold in folds]
    except Exception as e:  # a candidate that cannot fit should not kill the search
        logging.warning(f"candidate failed on a fold ({e}); scoring it as 0")
        return {**cand, "clf_threshold": np.nan, select_on: 0.0, "failed": True}, []
    cells = pd.concat([p[0] for p in parts], ignore_index=True)
    curve = score_thresholds(cells)
    best = max(curve, key=lambda row: row[select_on])
    saturated = {"u1_saturated": any(p[1] for p in parts),
                 "u2_saturated": any(p[2] for p in parts)}
    return {**cand, **best, **saturated, "failed": False}, curve


# --------------------------------------------------------------------------
# The ratios grid, factorized: u1 touches only stage 1 and u2 only stage 2,
# so each fold fits |U| classifiers and |U| regressors (not |U|^2 pairs) and
# every (u1, u2) combination is assembled from stored predictions.
# --------------------------------------------------------------------------
def ratios_grid(df_search, folds, features, clf_params, reg_params, select_on,
                u_values=U_VALUES, n_jobs=1, nthread=1):
    clf_params = {**clf_params, "nthread": nthread}
    reg_params = {**reg_params, "nthread": nthread}

    def one_fit(fold_idx, stage, u):
        train_sids, val_sids = folds[fold_idx]
        df_train = df_search[df_search["sid"].isin(train_sids)]
        df_test = df_search[df_search["sid"].isin(val_sids)]
        if stage == "clf":
            model, saturated = fit_stage1(df_train, features, clf_params, u)
            pred = model.predict_proba(df_test[features])[:, 1]
        else:
            model, saturated = fit_stage2(df_train, features, reg_params, u)
            pred = model.predict(df_test[features])
        return (fold_idx, stage, u), (pred, saturated)

    jobs = [(i, stage, u) for i in range(len(folds)) for stage in ("clf", "reg") for u in u_values]
    logging.info(f"ratios grid: {len(jobs)} stage fits "
                 f"({len(folds)} folds x 2 stages x {len(u_values)} ratios) cover "
                 f"{len(u_values) ** 2} (u1, u2) combinations x {len(CLF_THRESHOLDS)} thresholds")
    fitted = dict(Parallel(n_jobs=n_jobs)(delayed(one_fit)(*job) for job in jobs))

    test_frames = [df_search[df_search["sid"].isin(folds[i][1])] for i in range(len(folds))]
    results, curves = [], {}
    for u1 in u_values:
        for u2 in u_values:
            cells = pd.concat([
                oof_frame(test_frames[i],
                          fitted[(i, "clf", u1)][0], fitted[(i, "reg", u2)][0])
                for i in range(len(folds))
            ], ignore_index=True)
            curve = score_thresholds(cells)
            best = max(curve, key=lambda row: row[select_on])
            results.append({
                **candidate_from_params(clf_params, reg_params), "u1": u1, "u2": u2, **best,
                "u1_saturated": any(fitted[(i, "clf", u1)][1] for i in range(len(folds))),
                "u2_saturated": any(fitted[(i, "reg", u2)][1] for i in range(len(folds))),
                "failed": False,
            })
            curves[(u1, u2)] = curve
    return results, curves


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
MIN_EVENTS = 6  # below this, grouped folds cannot be formed meaningfully


def frozen_tree_params(args):
    """Tree parameters for --search ratios: an earlier winner's, or the defaults."""
    source = args.from_json or HYPERPARAMETERS_PATH
    try:
        with open(source) as f:
            tuned = json.load(f)
        clf_params = dict(tuned["clf_params"])
        reg_params = dict(tuned["reg_params"])
        logging.info(f"ratios grid uses the tree parameters from {source}")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        clf_params = TwoStageXGBoost._default_clf_params()
        reg_params = TwoStageXGBoost._default_reg_params()
        logging.info(f"no usable tree parameters at {source}; ratios grid uses the built-in defaults")
    for params in (clf_params, reg_params):
        params.pop("nthread", None)
    return clf_params, reg_params


def main(args):
    df = load_dataset(args.input)
    severity = event_severity_labels(df)

    if df["sid"].nunique() < MIN_EVENTS:
        logging.error(
            f"only {df['sid'].nunique()} cyclones available; at least {MIN_EVENTS} are needed "
            f"to form grouped folds and a held-out set. Point --input at the full dataset."
        )
        return 1

    search_events, holdout_events = split_events(
        df, severity, args.n_events, args.test_fraction, args.random_state)
    logging.info(f"search pool: {len(search_events)} cyclones | held-out: {len(holdout_events)} cyclones")

    df_search = df[df["sid"].isin(search_events)].copy()
    df_holdout = df[df["sid"].isin(holdout_events)].copy()

    n_folds = min(args.n_folds, max(2, len(search_events)))
    folds = make_folds(df_search, severity, n_folds, args.random_state)
    for i, (tr, va) in enumerate(folds):
        logging.info(f"  fold {i}: {len(tr)} train cyclones, {len(va)} val cyclones")

    if args.search == "ratios":
        clf_params, reg_params = frozen_tree_params(args)
        results, curves = ratios_grid(
            df_search, folds, FEATURES, clf_params, reg_params, args.select_on,
            n_jobs=args.n_jobs, nthread=args.nthread)
    else:
        candidates = list(ParameterSampler(PARAM_SPACE, n_iter=args.n_iter,
                                           random_state=args.random_state))
        logging.info(f"random search: {len(candidates)} candidates x {len(folds)} folds "
                     f"= {len(candidates) * len(folds)} two-stage fits; "
                     f"{len(CLF_THRESHOLDS)} thresholds scored per candidate for free")
        scored = Parallel(n_jobs=args.n_jobs)(
            delayed(score_candidate)(c, df_search, folds, FEATURES, args.select_on, args.nthread)
            for c in candidates
        )
        results = [row for row, _ in scored]
        curves = {i: curve for i, (_, curve) in enumerate(scored)}

    results_df = pd.DataFrame(results).sort_values(args.select_on, ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_path = args.output.with_name(args.output.stem + "_search_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"all candidate scores (each at its best threshold) -> {results_path}")

    if results_df.empty or results_df["failed"].all():
        logging.error("every candidate failed; nothing written")
        return 1

    top = results_df.head(5)
    logging.info("top candidates by %s (selection noise check):\n%s", args.select_on,
                 top[["u1", "u2", "clf_threshold", args.select_on]].to_string(index=False))

    best = results_df.iloc[0]
    best_key = best.name  # original candidate index / (u1, u2) lookup key
    param_keys = list(PARAM_SPACE.keys())
    best_cand = {k: best[k] for k in param_keys if k in results_df.columns}
    for k in ("clf_max_depth", "clf_min_child_weight", "clf_n_estimators",
              "reg_max_depth", "reg_min_child_weight", "reg_n_estimators", "u1", "u2"):
        if k in best_cand:
            best_cand[k] = int(best_cand[k])
    selected_threshold = float(best["clf_threshold"])
    logging.info(f"best candidate by {args.select_on} (threshold {selected_threshold}):\n"
                 f"{json.dumps(best_cand, indent=2, default=str)}")

    # The held-out cyclones have influenced nothing above. They are fitted and
    # scored once; the quotable number is the one at the selected threshold,
    # and the full curve is kept for the response-surface figure.
    clf_params, reg_params = xgb_params_from_candidate(best_cand, nthread=args.nthread or 0)
    df_train = df_search
    df_test = df_holdout
    clf, _ = fit_stage1(df_train, FEATURES, clf_params, best_cand["u1"])
    reg, _ = fit_stage2(df_train, FEATURES, reg_params, best_cand["u2"])
    holdout_curve = score_thresholds(
        oof_frame(df_test, clf.predict_proba(df_test[FEATURES])[:, 1],
                  reg.predict(df_test[FEATURES])))
    holdout_metrics = next(row for row in holdout_curve
                           if row["clf_threshold"] == selected_threshold)
    logging.info(f"held-out metrics at the selected threshold (evaluated once):\n"
                 f"{json.dumps(holdout_metrics, indent=2, default=str)}")

    cv_curve = (curves[(best_cand["u1"], best_cand["u2"])]
                if args.search == "ratios" else curves[int(best_key)])
    payload = {
        "clf_params": clf_params,
        "reg_params": reg_params,
        "u1": best_cand["u1"],
        "u2": best_cand["u2"],
        "clf_threshold": selected_threshold,
        "candidate": best_cand,
        "selected_on": args.select_on,
        "cv_metrics": {k: (None if pd.isna(best[k]) else float(best[k]))
                       for k in ("mae", "precision", "recall", "f1",
                                 "precision_high", "recall_high", "f1_high", "n_adm1")
                       if k in results_df.columns},
        "cv_threshold_curve": cv_curve,
        "holdout_metrics": holdout_metrics,
        "holdout_threshold_curve": holdout_curve,
        "top_candidates": top.to_dict(orient="records"),
        "search": {
            "input": str(args.input), "search": args.search,
            "n_candidates": int(len(results_df)),
            "n_events": len(search_events) + len(holdout_events),
            "n_search_events": len(search_events), "n_holdout_events": len(holdout_events),
            "n_folds": len(folds), "random_state": args.random_state,
            "group_col": "sid",
            "clf_thresholds": list(CLF_THRESHOLDS),
            "search_events": sorted(map(str, search_events)),
            "holdout_events": sorted(map(str, holdout_events)),
        },
    }
    with open(args.output, "w") as f:
        # numpy scalars are not JSON serializable; .item() converts them.
        json.dump(payload, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
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
    p.add_argument("--search", choices=["random", "ratios"], default="random",
                   help="random search over tree parameters and ratios (default), or an "
                        "exhaustive, factorized u1 x u2 grid with the tree parameters "
                        "frozen (--from-json, else the tuned JSON, else the defaults)")
    p.add_argument("--from-json", type=Path, default=None,
                   help="for --search ratios: JSON whose clf_params/reg_params are frozen")
    p.add_argument("--n-iter", type=int, default=80, help="random-search candidates")
    p.add_argument("--n-events", type=int, default=200,
                   help="cyclones sampled for the whole search; 0 uses all of them")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="fraction of sampled cyclones held out and scored once")
    p.add_argument("--n-folds", type=int, default=5, help="cyclone-grouped CV folds")
    p.add_argument("--n-jobs", type=int, default=4, help="parallel workers (joblib)")
    p.add_argument("--nthread", type=int, default=1,
                   help="xgboost threads per fit; keep n_jobs * nthread <= cores")
    p.add_argument("--select-on", default="f1_high",
                   help="metric that decides the winner (pre-specify it; do not eyeball "
                        "several). Default f1_high: ADM1-level F1 at the 15%% threshold, "
                        "the selection objective reported in the paper.")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--dry-run", action="store_true",
                   help="tiny budget for a smoke test: 2 candidates, 2 folds, 20 cyclones")
    args = p.parse_args(argv)
    if args.dry_run:
        args.n_iter, args.n_folds, args.n_events, args.n_jobs = 2, 2, 20, 1
    return args


if __name__ == "__main__":
    sys.exit(main(parse_args()))
