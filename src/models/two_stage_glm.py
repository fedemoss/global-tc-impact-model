"""Two-part generalized linear model: the statistical baseline for the two-stage XGBoost.

The question it answers is whether the learning algorithm matters: given the
same wind, rainfall, exposure and contextual predictors, does a model with
much simpler predictor-outcome relationships do as well? It therefore keeps
the two-stage structure and changes only what each stage is:

* stage 1, occurrence: L2-regularised logistic regression for "any reported
  impact" (target > 0);
* stage 2, severity: L2-regularised fractional logistic regression
  (Papke and Wooldridge, 1996) for the affected fraction, fitted only on cells
  with reported impact;
* prediction: zero where stage 1 does not flag the cell -- the same hard gate,
  so the model can predict exact zeros, which the 0% evaluation needs --
  otherwise the stage-2 mean, as a percentage.

Everything around the two stages is the XGBoost pipeline's own, so that a
difference in score is a difference between the algorithms: the same grid
predictors and training rows, the same undersampling (`TwoStageXGBoost.oversample`,
u1 on the occurrence label, u2 on the >= 15% label), and, through `train.py`,
the same cyclone-grouped folds and the same aggregation to ADM1. What a linear
model needs and a tree ensemble does not is added inside each stage:

* heavy-tailed, non-negative predictors enter as log1p (`LOG_FEATURES`);
* missing values are imputed with the training median, with an indicator
  column per imputed predictor, and every predictor is standardised.

Both steps are part of a pipeline fitted on each fold's training rows only, so
nothing about the test cyclone reaches the transformation.

Fractional logit with scikit-learn. For a fraction y in [0, 1], the Bernoulli
quasi-likelihood is y log p + (1 - y) log(1 - p), which is the log-loss of two
weighted rows: label 1 with weight y and label 0 with weight 1 - y. A weighted
`LogisticRegression` on the doubled rows therefore maximises exactly the
fractional-logit objective, boundary values included (a cell at 100% simply has
weight 0 on its label-0 row), and its prediction stays inside (0, 1). The L2
penalty applies to the coefficients as usual.

The regularisation strengths (`C1`, `C2`), the ratios and the stage-1 cutoff
are read from `GLM_HYPERPARAMETERS_PATH`, written by `src/models/glm_search.py`.
Without that file the ratios fall back to the XGBoost model's tuned values,
the controlled starting point, but the cutoff does not: it belongs to each
model's own probability scale, and an untuned GLM uses 0.5 with a warning.
"""
import json
import logging

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, GLM_HYPERPARAMETERS_PATH
from src.models.two_stage_xgb import TwoStageXGBoost, load_tuned_hyperparameters

logger = logging.getLogger(__name__)

#: Predictors that are counts or amounts with long right tails. A linear
#: predictor would otherwise be dominated by a handful of extreme cells.
LOG_FEATURES = (
    "population", "coast_length_meters", "rainfall_max_24h",
    "landslide_risk_sum", "N_events_5_years",
)

#: The high-impact boundary, inclusive, as everywhere else in the pipeline.
IMPACT_THRESHOLD_HIGH = 15

TARGET_NAME = "perc_affected_pop_grid_region"


def load_glm_hyperparameters(path=None):
    """Configuration written by the GLM search, or None if it has not been run."""
    path = path or GLM_HYPERPARAMETERS_PATH
    try:
        with open(path) as f:
            tuned = json.load(f)
    except FileNotFoundError:
        logger.info(f"No tuned GLM configuration at {path}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse {path} ({e}); the GLM runs untuned")
        return None
    logger.info(f"Loaded the tuned GLM configuration from {path}")
    return tuned


def design_matrix(df, features, log_features=LOG_FEATURES):
    """The predictors as a float matrix, with the heavy-tailed ones on log1p.

    Negative values of a logged predictor are clipped to zero first: the
    predictors in `LOG_FEATURES` are non-negative by construction, and a
    stray negative would otherwise become NaN rather than an error.
    """
    X = df[list(features)].to_numpy(dtype="float64", copy=True)
    for j, name in enumerate(features):
        if name in log_features:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return X


def make_stage(C, max_iter=1000, tol=1e-6):
    """One stage: median imputation with indicators, scaling, logistic regression.

    `tol` is tighter than scikit-learn's 1e-4: the fits are cheap, and the
    coefficients then agree with a dedicated GLM solver's to rounding rather
    than to the solver's stopping rule.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol)),
    ])


def fit_occurrence(X, y_any, C, max_iter=1000):
    """Stage 1: logistic regression on the (already rebalanced) occurrence label."""
    return make_stage(C, max_iter).fit(X, y_any)


def fit_severity(X, fraction, C, max_iter=1000):
    """Stage 2: fractional logit, as a weighted logistic regression on doubled rows."""
    fraction = np.clip(np.asarray(fraction, dtype="float64"), 0.0, 1.0)
    n = len(fraction)
    X2 = np.vstack([X, X])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    weights = np.concatenate([fraction, 1.0 - fraction])
    return make_stage(C, max_iter).fit(X2, labels, logit__sample_weight=weights)


def severity_percent(stage2, X):
    """The stage-2 mean as a percentage of population."""
    return 100.0 * stage2.predict_proba(X)[:, 1]


def fit_stage1(df_train, features, C, u1, target_name=TARGET_NAME,
               log_features=LOG_FEATURES, max_iter=1000):
    """Occurrence stage on the u1-rebalanced rows; returns (model, saturated).

    Only the predictors and the label travel into the rebalanced frame, so a
    fold's training set is not copied whole. `saturated` is True when u1 asks
    for more unaffected rows than exist, i.e. nothing was dropped.
    """
    label = (df_train[target_name] > 0).astype(int)
    n_minority = int(label.sum())
    frame = df_train[list(features)].assign(reported_bin=label)
    frame = TwoStageXGBoost.oversample(frame, target="reported_bin", u=u1)
    model = fit_occurrence(design_matrix(frame, features, log_features),
                           frame["reported_bin"].to_numpy(), C, max_iter)
    return model, u1 * n_minority > len(label) - n_minority


def fit_stage2(df_train, features, C, u2, target_name=TARGET_NAME,
               log_features=LOG_FEATURES, max_iter=1000):
    """Severity stage on observed positives, u2-rebalanced on the >= 15% label.

    Like the XGBoost regressor it never sees the stage-1 cutoff, so the cutoff
    can be swept afterwards without refitting. Returns (model, saturated).
    """
    positive = df_train[df_train[target_name] > 0]
    high = (positive[target_name] >= IMPACT_THRESHOLD_HIGH).astype(int)
    n_minority = int(high.sum())
    frame = positive[list(features)].assign(
        impact_high=high, fraction=positive[target_name].clip(0, 100) / 100.0)
    frame = TwoStageXGBoost.oversample(frame, target="impact_high", u=u2)
    model = fit_severity(design_matrix(frame, features, log_features),
                         frame["fraction"].to_numpy(), C, max_iter)
    return model, u2 * n_minority > len(high) - n_minority


class TwoStageGLM:
    """Logistic occurrence gate followed by a fractional-logit severity model.

    Same interface as `TwoStageXGBoost` (`fit`, `predict`, `train_and_predict`,
    and `classifier` / `regressor` for the artefacts the LOOCV run saves), so
    `train.py` runs it through the identical fold loop.

    Any of `C1`, `C2`, `u1`, `u2` and `clf_threshold` left as None is taken
    from the tuned GLM file when it exists; see the module docstring for the
    fallbacks when it does not. An explicit value always wins.
    """

    DEFAULT_C = 1.0
    DEFAULT_CLF_THRESHOLD = 0.5

    def __init__(self, features=None, target_name="perc_affected_pop_grid_region",
                 C1=None, C2=None, u1=None, u2=None, clf_threshold=None,
                 tuned=None, use_tuned=True, log_features=LOG_FEATURES,
                 max_iter=1000):
        tuned = tuned if tuned is not None else (
            load_glm_hyperparameters() if use_tuned else None)
        tuned = tuned or {}
        xgb = (load_tuned_hyperparameters() if use_tuned else None) or {}

        if use_tuned and not tuned and clf_threshold is None:
            logger.warning(
                "the GLM has no tuned configuration (run src/models/glm_search.py); "
                f"using C = {self.DEFAULT_C} and a stage-1 cutoff of "
                f"{self.DEFAULT_CLF_THRESHOLD}, not the XGBoost model's")

        self.C1 = C1 if C1 is not None else tuned.get("C1", self.DEFAULT_C)
        self.C2 = C2 if C2 is not None else tuned.get("C2", self.DEFAULT_C)
        self.u1 = u1 if u1 is not None else tuned.get(
            "u1", xgb.get("u1", TwoStageXGBoost.DEFAULT_U1))
        self.u2 = u2 if u2 is not None else tuned.get(
            "u2", xgb.get("u2", TwoStageXGBoost.DEFAULT_U2))
        self.clf_threshold = (clf_threshold if clf_threshold is not None
                              else tuned.get("clf_threshold", self.DEFAULT_CLF_THRESHOLD))
        self.log_features = tuple(tuned.get("log_features", log_features))

        self.features = list(features) if features is not None else list(FEATURES)
        self.target_name = target_name
        self.max_iter = max_iter
        self.classifier = None
        self.regressor = None

    def _design(self, df):
        return design_matrix(df, self.features, self.log_features)

    def fit(self, df_train, u1=None, u2=None):
        """Fit both stages, rebalancing each exactly as the XGBoost model does."""
        u1 = u1 if u1 is not None else self.u1
        u2 = u2 if u2 is not None else self.u2
        common = {"target_name": self.target_name, "log_features": self.log_features,
                  "max_iter": self.max_iter}
        self.classifier, _ = fit_stage1(df_train, self.features, self.C1, u1, **common)
        self.regressor, _ = fit_stage2(df_train, self.features, self.C2, u2, **common)
        return self

    def predict(self, df_test, clf_threshold=None):
        """Predict with both stages already fitted by `fit`.

        Returns the test frame with the same columns the XGBoost model adds:
        `reported_bin`, `predicted_proba`, `predicted_bin` and `prediction_perc`.
        """
        clf_threshold = clf_threshold if clf_threshold is not None else self.clf_threshold

        df_result = df_test.copy()
        df_result["reported_bin"] = (df_result[self.target_name] > 0).astype(int)

        X = self._design(df_result)
        proba = self.classifier.predict_proba(X)[:, 1]
        gate = proba >= clf_threshold
        df_result["predicted_proba"] = proba
        df_result["predicted_bin"] = gate.astype(int)

        prediction = np.zeros(len(df_result), dtype="float64")
        if gate.any():
            prediction[gate] = severity_percent(self.regressor, X[gate])
        df_result["prediction_perc"] = prediction
        return df_result

    def train_and_predict(self, df_train, df_test, u1=None, u2=None, clf_threshold=None):
        """Fit on `df_train` and predict `df_test`, the call the fold loop makes."""
        return self.fit(df_train, u1=u1, u2=u2).predict(df_test, clf_threshold=clf_threshold)
