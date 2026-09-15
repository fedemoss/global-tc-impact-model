"""Two-stage XGBoost impact model.

Stage 1 is a binary classifier ("is this grid cell affected at all?"), stage 2 a
regressor for the affected percentage, fitted only on cells stage 1 flags as
positive. Both stages, the two class-balancing ratios (`u1`, `u2`) and the
stage-1 decision threshold are tuned jointly by `test/hyperparameter_search.py`
and read back here from `HYPERPARAMETERS_PATH`.
"""
import json
import logging

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.config import HYPERPARAMETERS_PATH


def load_tuned_hyperparameters(path=None):
    """Hyperparameters written by the search, or None if it has not been run.

    Returning None rather than raising keeps a fresh clone runnable: the model
    falls back to the defaults below, and the only consequence is that it is
    untuned.
    """
    path = path or HYPERPARAMETERS_PATH
    try:
        with open(path) as f:
            tuned = json.load(f)
    except FileNotFoundError:
        logging.info(f"No tuned hyperparameters at {path}; using built-in defaults")
        return None
    except json.JSONDecodeError as e:
        logging.warning(f"Could not parse {path} ({e}); using built-in defaults")
        return None

    missing = [k for k in ("clf_params", "reg_params") if k not in tuned]
    if missing:
        logging.warning(f"{path} is missing {missing}; using built-in defaults")
        return None
    logging.info(f"Loaded tuned hyperparameters from {path}")
    return tuned


class TwoStageXGBoost:
    """
    Custom 2-stage XGBoost model pipeline.
    Stage 1: Binary Classification (Impact > 0)
    Stage 2: Regression (Magnitude of impact, conditioned on predicted Impact > 0)

    Any of `clf_params`, `reg_params`, `u1`, `u2` and `clf_threshold` left as
    None is taken from the tuned JSON when it exists, and from the built-in
    defaults otherwise. Passing a value explicitly always wins, so callers that
    want to pin a configuration (the hyperparameter search itself, for one) are
    unaffected by whatever is on disk.
    """

    # Stage 1 flags a cell as affected when its predicted probability reaches
    # this; tuning it matters because the two classes are very unbalanced.
    DEFAULT_CLF_THRESHOLD = 0.5
    DEFAULT_U1 = 3
    DEFAULT_U2 = 3

    def __init__(self, clf_params=None, reg_params=None, features=None,
                 target_name="perc_affected_pop_grid_region",
                 u1=None, u2=None, clf_threshold=None, tuned=None,
                 use_tuned=True):
        tuned = tuned if tuned is not None else (load_tuned_hyperparameters() if use_tuned else None)
        tuned = tuned or {}

        self.clf_params = clf_params or tuned.get("clf_params") or self._default_clf_params()
        self.reg_params = reg_params or tuned.get("reg_params") or self._default_reg_params()
        self.u1 = u1 if u1 is not None else tuned.get("u1", self.DEFAULT_U1)
        self.u2 = u2 if u2 is not None else tuned.get("u2", self.DEFAULT_U2)
        self.clf_threshold = (
            clf_threshold if clf_threshold is not None
            else tuned.get("clf_threshold", self.DEFAULT_CLF_THRESHOLD)
        )

        self.features = features
        self.target_name = target_name
        self.classifier = XGBClassifier(**self.clf_params)
        self.regressor = XGBRegressor(**self.reg_params)

    @staticmethod
    def _default_clf_params():
        return {
            "booster": "gbtree", "colsample_bytree": 0.8, "gamma": 0.5, 
            "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 1, 
            "n_estimators": 100, "subsample": 0.8, "verbosity": 0, 
            "random_state": 0, "objective": "binary:logistic", "eval_metric": "logloss"
        }

    @staticmethod
    def _default_reg_params():
        return {
            "booster": "gbtree", "colsample_bytree": 0.8, "gamma": 0.5, 
            "learning_rate": 0.01, "max_depth": 4, "min_child_weight": 1, 
            "n_estimators": 100, "objective": "reg:squarederror", 
            "eval_metric": "rmse", "subsample": 0.8, "verbosity": 0, "random_state": 0
        }

    @staticmethod
    def oversample(df, target, u=1):
        """Rebalance `target` by keeping at most `u` majority rows per minority row."""
        minority_size = df[target].sum()
        majority_size = int(u * minority_size)
        
        df_balanced = (
            df.groupby(target, group_keys=False)
            .apply(lambda x: x.sample(n=min(minority_size if x.name == 1 else majority_size, len(x)), random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        return df_balanced

    def train_and_predict(self, df_train, df_test, u1=None, u2=None, clf_threshold=None):
        """Executes the custom 2-stage training and prediction flow with oversampling.

        `u1`, `u2` and `clf_threshold` fall back to the instance's tuned values.
        The inputs are copied rather than modified in place, so a caller can
        reuse the same frames across folds without them accumulating helper
        columns from a previous call.
        """
        u1 = u1 if u1 is not None else self.u1
        u2 = u2 if u2 is not None else self.u2
        clf_threshold = clf_threshold if clf_threshold is not None else self.clf_threshold

        df_train = df_train.copy()
        df_test = df_test.copy()

        # Stage 1: Classification
        df_train["reported_bin"] = (df_train[self.target_name] > 0).astype(int)
        df_test["reported_bin"] = (df_test[self.target_name] > 0).astype(int)
        
        df_train_balanced_1 = self.oversample(df_train, target="reported_bin", u=u1)
        self.classifier.fit(df_train_balanced_1[self.features], df_train_balanced_1["reported_bin"])
        
        y_proba = self.classifier.predict_proba(df_test[self.features])[:, 1]
        df_result = df_test.copy()
        df_result["predicted_proba"] = y_proba
        df_result["predicted_bin"] = (y_proba >= clf_threshold).astype(int)

        # Stage 2: Regression
        df_train_high = df_train[df_train[self.target_name] > 0].copy()
        df_train_high["impact_high"] = (df_train_high[self.target_name] >= 15).astype(int)
        
        df_train_balanced_2 = self.oversample(df_train_high, target="impact_high", u=u2)
        
        df_stage2_pos = df_result[df_result["predicted_bin"] == 1].copy()
        df_stage2_neg = df_result[df_result["predicted_bin"] == 0].copy()
        
        if not df_stage2_pos.empty:
            self.regressor.fit(df_train_balanced_2[self.features], df_train_balanced_2[self.target_name])
            df_stage2_pos["prediction_perc"] = self.regressor.predict(df_stage2_pos[self.features])
            
        df_stage2_neg["prediction_perc"] = 0
        
        return pd.concat([df_stage2_pos, df_stage2_neg], axis=0).sort_index()
