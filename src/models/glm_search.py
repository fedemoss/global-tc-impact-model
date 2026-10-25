#!/usr/bin/env python3
"""Tuning for the two-part GLM baseline, under the XGBoost model's own protocol.

The baseline gets its own tuning, so that it is not compared at a
configuration chosen for the other model, and it is tuned the way the XGBoost
model was, so that neither had the easier selection:

* **The same cyclones.** By default the search pool and the held-out set are
  read from the XGBoost model's tuned JSON (`search.search_events`,
  `search.holdout_events`), and the folds are rebuilt with the same splitter
  and seed (`StratifiedGroupKFold` on the IBTrACS sid, stratified by cyclone
  severity). On the dataset file that search ran on, the folds are the same
  folds. `--resplit` draws a fresh split with the XGBoost search's procedure.
* **The same objective.** Pooled out-of-fold predictions are aggregated to
  ADM1 by `score_thresholds`, the function the XGBoost search scores with, and
  the winner is chosen on `--select-on` (default `f1_high`, ADM1-level F1 at
  the 15% threshold).
* **The same held-out check.** The winner is refitted on the search pool and
  scored once on the held-out cyclones; that number is the one to quote.

What is searched:

* the two L2 regularisation strengths, C1 (occurrence) and C2 (severity), on a
  log grid (`--c-values`);
* the stage-1 cutoff, swept for free from stored probabilities. The XGBoost
  cutoff (0.6) is not carried over: the GLM picks its own, from the same set
  of cutoffs the XGBoost search scored (`CLF_THRESHOLDS`, 0.4 to 0.9), so both
  models choose from the same options. Both stage-1 models are log-loss fits
  to the same rebalanced rows, so their probabilities are on comparable
  scales and the range means the same for both. `--wide-thresholds` sweeps
  0.01 to 0.99 instead; check the stored curve before preferring a cutoff from
  its tails, since on `f1_high` alone a very low cutoff can win by a hair by
  gating nearly every cell, at a large cost in detection;
* the two balancing ratios only with `--tune-ratios`. By default they stay at
  the XGBoost model's tuned values, so that the imbalance handling of the two
  models is identical and only the algorithm differs.

The grid is factorized as in the XGBoost ratios search: C1 and u1 touch only
stage 1, C2 and u2 only stage 2, so each fold fits one occurrence model per
(C1, u1) and one severity model per (C2, u2), and every combination is
assembled from stored predictions.

Usage
-----
    python src/models/glm_search.py
    python src/models/glm_search.py --tune-ratios
    python src/models/glm_search.py --dry-run   # tiny, for a smoke test

Nothing is written until the search completes.
"""
import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (FEATURES, GLM_HYPERPARAMETERS_PATH, HYPERPARAMETERS_PATH,
                        INPUT_DIR)
from src.models.hyperparameter_search import (ADM1_KEYS, MIN_EVENTS, RANDOM_STATE,
                                              CLF_THRESHOLDS, TARGET_NAME,
                                              U1_VALUES, U2_VALUES,
                                              event_severity_labels, load_dataset,
                                              make_folds, oof_frame,
                                              score_thresholds, split_events)
from src.models.two_stage_glm import (LOG_FEATURES, design_matrix, fit_stage1,
                                      fit_stage2, severity_percent)

#: Inverse regularisation strengths. Predictors are standardised, so the same
#: grid means the same thing for every predictor; the upper end is close to
#: unpenalised at the sample sizes involved.
C_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

#: The wider sweep behind --wide-thresholds: steps of 0.05, tails resolved finely.
WIDE_THRESHOLDS = tuple(float(t) for t in np.round(np.r_[
    np.arange(0.01, 0.05, 0.01), np.arange(0.05, 0.951, 0.05), 0.97, 0.99], 2))

CELL_COLUMNS = ADM1_KEYS + ["id", "population", TARGET_NAME]


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def split_from_xgboost(df, path):
    """The XGBoost search's pool and held-out cyclones, and its search record."""
    with open(path) as f:
        record = json.load(f)["search"]
    present = set(df["sid"].astype(str).unique())
    search = [s for s in record["search_events"] if s in present]
    holdout = [s for s in record["holdout_events"] if s in present]
    lost = len(record["search_events"]) + len(record["holdout_events"]) - len(search) - len(holdout)
    if lost:
        logging.warning(
            f"{lost} cyclones of the XGBoost split are not in this dataset; the "
            f"GLM is tuned on the {len(search)} + {len(holdout)} that are, so its "
            f"folds will not match the XGBoost search's exactly")
    if record.get("input"):
        logging.info(f"XGBoost split read from {path} (that search ran on {record['input']})")
    return search, holdout, record


# --------------------------------------------------------------------------
# Out-of-fold cells
# --------------------------------------------------------------------------
class FoldCells:
    """A validation fold's cells, ready to take any combination's predictions.

    Equivalent to calling `oof_frame` on the fold for every combination, but
    the column selection and the per-(event, cell) deduplication are done once
    instead of once per combination.
    """

    def __init__(self, df_test):
        keep = ~df_test.duplicated(subset=["DisNo.", "id"], keep="first").to_numpy()
        self.keep = keep
        self.base = df_test.loc[keep, CELL_COLUMNS].reset_index(drop=True)
        self.population = self.base["population"].to_numpy(dtype="float64")

    def frame(self, proba, severity):
        return self.base.assign(
            proba=np.asarray(proba, dtype="float64")[self.keep],
            ppl_reg=np.clip(np.asarray(severity, dtype="float64")[self.keep], 0, 100)
            * self.population / 100.0)


def _fit_one(df_search, folds, fold_idx, stage, C, u, features, max_iter):
    """Fit one stage on one fold; return its predictions for the fold's cells."""
    train_sids, val_sids = folds[fold_idx]
    df_train = df_search[df_search["sid"].isin(train_sids)]
    df_test = df_search[df_search["sid"].isin(val_sids)]
    X_test = design_matrix(df_test, features, LOG_FEATURES)
    if stage == "occurrence":
        model, saturated = fit_stage1(df_train, features, C, u, max_iter=max_iter)
        pred = model.predict_proba(X_test)[:, 1]
    else:
        model, saturated = fit_stage2(df_train, features, C, u, max_iter=max_iter)
        pred = severity_percent(model, X_test)
    # float32 halves what the grid holds in memory; the scores do not notice.
    return (fold_idx, stage, C, u), (pred.astype("float32"), saturated)


def factorized_grid(df_search, folds, features, c_values, u1_values, u2_values,
                    select_on, thresholds=CLF_THRESHOLDS, n_jobs=1, max_iter=1000):
    """Every (C1, u1, C2, u2) scored at every cutoff, from |stage-1| + |stage-2| fits per fold."""
    stage1 = [(C, u) for C in c_values for u in u1_values]
    stage2 = [(C, u) for C in c_values for u in u2_values]
    jobs = ([(i, "occurrence", C, u) for i in range(len(folds)) for C, u in stage1]
            + [(i, "severity", C, u) for i in range(len(folds)) for C, u in stage2])
    logging.info(
        f"GLM grid: {len(jobs)} stage fits ({len(folds)} folds x ({len(stage1)} "
        f"occurrence + {len(stage2)} severity)) cover {len(stage1) * len(stage2)} "
        f"combinations x {len(thresholds)} cutoffs")
    # Threads, not processes: the fits spend their time in numpy and scipy,
    # which release the GIL, and threads share the search frame instead of
    # each worker receiving a pickled copy of it.
    fitted = dict(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_one)(df_search, folds, i, stage, C, u, features, max_iter)
        for i, stage, C, u in jobs))

    cells = [FoldCells(df_search[df_search["sid"].isin(folds[i][1])])
             for i in range(len(folds))]
    results, curves = [], {}
    for C1, u1 in stage1:
        for C2, u2 in stage2:
            pooled = pd.concat([
                cells[i].frame(fitted[(i, "occurrence", C1, u1)][0],
                               fitted[(i, "severity", C2, u2)][0])
                for i in range(len(folds))], ignore_index=True)
            curve = score_thresholds(pooled, thresholds=thresholds)
            best = max(curve, key=lambda row: row[select_on])
            results.append({
                "C1": C1, "u1": u1, "C2": C2, "u2": u2, **best,
                "u1_saturated": any(fitted[(i, "occurrence", C1, u1)][1] for i in range(len(folds))),
                "u2_saturated": any(fitted[(i, "severity", C2, u2)][1] for i in range(len(folds))),
            })
            curves[(C1, u1, C2, u2)] = curve
    return results, curves


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(args):
    df = load_dataset(args.input)
    severity = event_severity_labels(df)
    needed = list(dict.fromkeys(list(FEATURES) + CELL_COLUMNS + ["sid"]))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"the dataset is missing {missing}")
    df = df[needed]
    df["sid"] = df["sid"].astype(str)
    severity.index = severity.index.astype(str)

    if df["sid"].nunique() < MIN_EVENTS:
        logging.error(f"only {df['sid'].nunique()} cyclones available; at least "
                      f"{MIN_EVENTS} are needed. Point --input at the full dataset.")
        return 1

    xgb_tuned = {}
    if args.xgb_json.exists():
        with open(args.xgb_json) as f:
            xgb_tuned = json.load(f)
    if not args.resplit and xgb_tuned.get("search", {}).get("search_events"):
        search_events, holdout_events, xgb_record = split_from_xgboost(df, args.xgb_json)
        split_source = f"xgboost:{args.xgb_json}"
        n_folds = args.n_folds or int(xgb_record.get("n_folds", 5))
        random_state = (args.random_state if args.random_state is not None
                        else int(xgb_record.get("random_state", RANDOM_STATE)))
    else:
        if not args.resplit:
            logging.warning(f"no XGBoost cyclone split in {args.xgb_json}; drawing a "
                            f"fresh split with the XGBoost search's procedure")
        random_state = args.random_state if args.random_state is not None else RANDOM_STATE
        search_events, holdout_events = split_events(
            df, severity, args.n_events, args.test_fraction, random_state)
        split_source = "resplit"
        n_folds = args.n_folds or 5
    logging.info(f"search pool: {len(search_events)} cyclones | held-out: "
                 f"{len(holdout_events)} cyclones ({split_source})")

    df_search = df[df["sid"].isin(search_events)].copy()
    df_holdout = df[df["sid"].isin(holdout_events)].copy()
    del df
    gc.collect()

    n_folds = min(n_folds, max(2, len(search_events)))
    folds = make_folds(df_search, severity, n_folds, random_state)
    for i, (tr, va) in enumerate(folds):
        logging.info(f"  fold {i}: {len(tr)} train cyclones, {len(va)} val cyclones")

    if args.tune_ratios:
        u1_values, u2_values = U1_VALUES, U2_VALUES
    else:
        u1_values = (int(xgb_tuned.get("u1", args.u1)),)
        u2_values = (int(xgb_tuned.get("u2", args.u2)),)
        logging.info(f"ratios fixed at the XGBoost model's: u1 = {u1_values[0]}, "
                     f"u2 = {u2_values[0]} (--tune-ratios to search them)")

    results, curves = factorized_grid(
        df_search, folds, FEATURES, args.c_values, u1_values, u2_values,
        args.select_on, thresholds=args.thresholds, n_jobs=args.n_jobs,
        max_iter=args.max_iter)

    results_df = pd.DataFrame(results).sort_values(args.select_on, ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_path = args.output.with_name(args.output.stem + "_search_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"all combinations (each at its best cutoff) -> {results_path}")

    top = results_df.head(5)
    logging.info("top combinations by %s (selection noise check):\n%s", args.select_on,
                 top[["C1", "u1", "C2", "u2", "clf_threshold", args.select_on]].to_string(index=False))

    best = results_df.iloc[0]
    C1, C2 = float(best["C1"]), float(best["C2"])
    u1, u2 = int(best["u1"]), int(best["u2"])
    threshold = float(best["clf_threshold"])
    for name, value, grid in (("C1", C1, args.c_values), ("C2", C2, args.c_values),
                              ("clf_threshold", threshold, args.thresholds)):
        if value in (min(grid), max(grid)):
            logging.warning(f"selected {name} = {value} is at the edge of its grid; "
                            f"the optimum may lie beyond it")

    # The held-out cyclones have influenced nothing above: fitted and scored once.
    stage1_model, _ = fit_stage1(df_search, FEATURES, C1, u1, max_iter=args.max_iter)
    stage2_model, _ = fit_stage2(df_search, FEATURES, C2, u2, max_iter=args.max_iter)
    X_holdout = design_matrix(df_holdout, FEATURES, LOG_FEATURES)
    holdout_curve = score_thresholds(
        oof_frame(df_holdout, stage1_model.predict_proba(X_holdout)[:, 1],
                  severity_percent(stage2_model, X_holdout)),
        thresholds=args.thresholds)
    holdout_metrics = next(row for row in holdout_curve
                           if np.isclose(row["clf_threshold"], threshold))
    logging.info(f"held-out metrics at the selected cutoff (evaluated once):\n"
                 f"{json.dumps(holdout_metrics, indent=2)}")

    metric_keys = ("mae", "precision", "recall", "f1",
                   "precision_high", "recall_high", "f1_high", "n_adm1")
    payload = {
        "model": "two-part GLM: logistic occurrence, fractional-logit severity",
        "C1": C1, "C2": C2, "u1": u1, "u2": u2, "clf_threshold": threshold,
        "log_features": list(LOG_FEATURES),
        "selected_on": args.select_on,
        "cv_metrics": {k: float(best[k]) for k in metric_keys if k in best},
        "cv_threshold_curve": curves[(C1, u1, C2, u2)],
        "holdout_metrics": holdout_metrics,
        "holdout_threshold_curve": holdout_curve,
        "top_candidates": top.to_dict(orient="records"),
        # The XGBoost model's numbers on the same pool and held-out set, for
        # a first comparison before any LOOCV run.
        "reference_xgboost": ({"cv_metrics": xgb_tuned.get("cv_metrics"),
                               "holdout_metrics": xgb_tuned.get("holdout_metrics"),
                               "clf_threshold": xgb_tuned.get("clf_threshold")}
                              if xgb_tuned else None),
        "search": {
            "input": str(args.input), "split_source": split_source,
            "c_values": list(args.c_values), "u1_values": list(u1_values),
            "u2_values": list(u2_values), "tune_ratios": bool(args.tune_ratios),
            "n_combinations": int(len(results_df)),
            "n_search_events": len(search_events), "n_holdout_events": len(holdout_events),
            "n_folds": len(folds), "random_state": random_state, "group_col": "sid",
            "clf_thresholds": list(args.thresholds),
            "search_events": sorted(search_events),
            "holdout_events": sorted(holdout_events),
        },
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    logging.info(f"chosen GLM configuration -> {args.output} (picked up by TwoStageGLM)")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=INPUT_DIR / "model_input_dataset" / "training_dataset.parquet")
    p.add_argument("--output", type=Path, default=GLM_HYPERPARAMETERS_PATH,
                   help="where to write the chosen GLM configuration")
    p.add_argument("--xgb-json", type=Path, default=HYPERPARAMETERS_PATH,
                   help="the XGBoost model's tuned JSON, whose cyclone split, fold "
                        "settings and ratios are reused")
    p.add_argument("--resplit", action="store_true",
                   help="draw a fresh search/held-out split instead of reusing the XGBoost one")
    p.add_argument("--n-events", type=int, default=200,
                   help="with --resplit: cyclones sampled for the search; 0 uses all")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="with --resplit: fraction of sampled cyclones held out")
    p.add_argument("--n-folds", type=int, default=None,
                   help="cyclone-grouped CV folds (default: the XGBoost search's)")
    p.add_argument("--random-state", type=int, default=None,
                   help="fold seed (default: the XGBoost search's)")
    p.add_argument("--c-values", type=float, nargs="+", default=list(C_VALUES),
                   help="inverse regularisation strengths searched for both stages")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(CLF_THRESHOLDS),
                   help="stage-1 cutoffs swept (default: the XGBoost search's, 0.4 to 0.9)")
    p.add_argument("--wide-thresholds", action="store_true",
                   help="sweep 0.01 to 0.99 instead of --thresholds")
    p.add_argument("--tune-ratios", action="store_true",
                   help="also grid u1 and u2 over the XGBoost search's ladders")
    p.add_argument("--u1", type=int, default=2,
                   help="u1 when it is neither tuned nor in the XGBoost JSON")
    p.add_argument("--u2", type=int, default=10,
                   help="u2 when it is neither tuned nor in the XGBoost JSON")
    p.add_argument("--select-on", default="f1_high",
                   help="metric that decides the winner; the default is the XGBoost "
                        "search's objective, ADM1-level F1 at 15%%")
    p.add_argument("--n-jobs", type=int, default=1, help="parallel stage fits (threads)")
    p.add_argument("--max-iter", type=int, default=1000, help="lbfgs iterations per fit")
    p.add_argument("--dry-run", action="store_true",
                   help="tiny budget for a smoke test: fresh split of 20 cyclones, "
                        "2 folds, 2 values of C")
    args = p.parse_args(argv)
    if args.wide_thresholds:
        args.thresholds = list(WIDE_THRESHOLDS)
    # score_thresholds bins cells with searchsorted, which needs ascending cutoffs.
    args.thresholds = sorted(set(args.thresholds))
    if args.dry_run:
        args.resplit, args.n_events, args.n_folds = True, 20, 2
        args.c_values = [0.1, 10.0]
    return args


if __name__ == "__main__":
    sys.exit(main(parse_args()))
