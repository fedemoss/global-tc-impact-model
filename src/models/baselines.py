"""Reference baselines the model is compared against.

All three are deterministic rules, not fitted models: the comparison in the
paper is against what a practitioner could compute from a wind field and a
record of past events, with no training involved.

* ``windspeed-exposed``    everyone in a cell is affected once the wind exceeds
                           the threshold.
* ``historical``           the country's median past impact percentage, applied
                           regardless of this storm's wind.
* ``windspeed-historical``  that same median, but only where the wind exceeds
                           the threshold.

Each exposes the ``train_and_predict(df_train, df_test)`` interface the LOOCV
pipeline calls, so they run through the same evaluation path as the model. Only
the historical median actually consults ``df_train``; the wind rule ignores it.
"""
import logging

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

TARGET_NAME = "perc_affected_pop_grid_region"

# Sustained wind above which the exposed-population rules consider a cell hit.
# 33 m/s is the Saffir-Simpson category 1 threshold.
WIND_THRESHOLD = 33.0


def country_median_past_impact(df, target_name=TARGET_NAME):
    """Per (country, event): median non-zero impact % of that country's other events.

    Only events with a recorded impact enter the median; a country whose other
    events were all zero gets 0. The comparison is ``year <= year``, so an event
    can see others from its own year. That is a mild look-ahead, kept because it
    is how the published baseline was computed -- flagged here rather than
    silently changed, since correcting it would move the reported numbers.
    """
    events = (
        df.groupby(["GID_0", "DisNo.", "year"], observed=True)[target_name]
        .apply(lambda x: x.unique())
        .reset_index()
    )
    events["impact_percent"] = events[target_name].apply(
        lambda arr: [v for v in arr if v != 0])
    events = events.drop(columns=target_name)

    # Each event is paired with the recorded impacts of its country's other
    # events up to its own year, and the median taken over that pool. Pairing
    # within a country keeps the product small: the countries are independent,
    # so this never forms a pool across the whole record.
    sources = (events.explode("impact_percent")
               .dropna(subset=["impact_percent"])
               .rename(columns={"DisNo.": "source_event", "year": "source_year",
                                "impact_percent": "impact"}))
    pairs = events[["GID_0", "DisNo.", "year"]].merge(
        sources[["GID_0", "source_event", "source_year", "impact"]],
        on="GID_0", how="inner")
    pairs = pairs[(pairs["source_year"] <= pairs["year"])
                  & (pairs["source_event"] != pairs["DisNo."])]

    medians = (pairs.groupby(["GID_0", "DisNo."], observed=True)["impact"]
               .median().rename("median_pct").reset_index())
    out = events[["GID_0", "DisNo."]].merge(medians, on=["GID_0", "DisNo."],
                                            how="left")
    # An event with no earlier impact in its country has no pool to take a
    # median over, and predicts nothing.
    out["median_pct"] = pd.to_numeric(out["median_pct"], errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)


class _Baseline:
    """Shared plumbing: predictions are a percentage of each cell's population."""

    name = "baseline"

    def __init__(self, target_name=TARGET_NAME, wind_threshold=WIND_THRESHOLD):
        self.target_name = target_name
        self.wind_threshold = wind_threshold

    def _predict_percent(self, df_train, df_test):
        raise NotImplementedError

    def train_and_predict(self, df_train, df_test):
        df_result = df_test.copy()
        df_result["prediction_perc"] = (
            self._predict_percent(df_train, df_test).clip(lower=0, upper=100))
        return df_result


class WindspeedExposedModel(_Baseline):
    """Everyone in the cell is affected when the wind exceeds the threshold."""

    name = "windspeed-exposed"

    def _predict_percent(self, df_train, df_test):
        hit = (df_test["wind_speed"] > self.wind_threshold).astype(float)
        return 100.0 * hit


class HistoricalModel(_Baseline):
    """The country's median past impact %, applied regardless of this storm's wind."""

    name = "historical"

    def _median_for(self, df_train, df_test):
        # The median is computed over the training events only, which under
        # LOOCV is every event except the one being predicted.
        medians = country_median_past_impact(df_train, self.target_name)
        per_country = (medians.groupby("GID_0", observed=True)["median_pct"]
                       .median().rename("country_median"))
        out = df_test[["GID_0"]].merge(
            per_country, left_on="GID_0", right_index=True, how="left")
        return out["country_median"].fillna(0.0).to_numpy()

    def _predict_percent(self, df_train, df_test):
        return pd.Series(self._median_for(df_train, df_test), index=df_test.index)


class WindspeedHistoricalModel(HistoricalModel):
    """The historical median, applied only where the wind exceeds the threshold."""

    name = "windspeed-historical"

    def _predict_percent(self, df_train, df_test):
        median = pd.Series(self._median_for(df_train, df_test), index=df_test.index)
        hit = (df_test["wind_speed"] > self.wind_threshold).astype(float)
        return median * hit


# --------------------------------------------------------------------------
# Direct evaluation
# --------------------------------------------------------------------------
#: Written beside the per-event folders the fitted models produce.
BUNDLE_SUFFIX = ".parquet"


def predict_all(df, wind_threshold=WIND_THRESHOLD, target_name=TARGET_NAME):
    """Every baseline's grid-cell prediction, in one pass over the dataset.

    These are rules, not fitted models: none of them has a parameter estimated
    from a training split, so there is nothing for cross-validation to hold
    out. The one leave-one-out element -- a country's median must not include
    the event being predicted -- is a property of
    ``country_median_past_impact`` and needs no fold.

    Returns {model name: frame with `prediction_perc`}.
    """
    medians = country_median_past_impact(df, target_name)
    base = df.merge(medians, on=["GID_0", "DisNo."], how="left")
    base["median_pct"] = base["median_pct"].fillna(0.0)
    exposed = (base["wind_speed"] > wind_threshold).astype(float)

    percentages = {
        "historical": base["median_pct"],
        "windspeed-exposed": 100.0 * exposed,
        "windspeed-historical": base["median_pct"] * exposed,
    }

    out = {}
    for name, perc in percentages.items():
        frame = base.drop(columns=["median_pct"]).copy()
        frame["prediction_perc"] = perc.clip(lower=0, upper=100).to_numpy()
        out[name] = frame
    return out


def build_all_baselines(df=None, strategy="global", wind_threshold=WIND_THRESHOLD,
                        target_name=TARGET_NAME):
    """Score every baseline and write one bundle per model and level.

    The fitted models write a file per event because each event is a separate
    fold; a rule has no folds, so its predictions are kept whole. The file
    names keep the `<model>_<strategy>_<level>` form the evaluation reads, so
    the two sit side by side.
    """
    from src.evaluation.aggregation import aggregate_predictions_to_adm1
    from src.models.train import load_training_dataset, prepare_data

    if df is None:
        df = prepare_data(aggregate_to_adm1=False)

    written = []
    for name, grid in predict_all(df, wind_threshold, target_name).items():
        grid_path = OUTPUT_DIR / f"model_output/{name}_{strategy}_grid{BUNDLE_SUFFIX}"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(grid_path, index=False)

        adm1 = aggregate_predictions_to_adm1(grid, target_name=target_name)
        # The aggregation groups on the country-event, which drops the storm
        # identifier. Intervals are resampled over the physical cyclone, and a
        # cyclone striking several countries yields one record per country, so
        # the identifier is put back.
        if "sid" in grid.columns and "sid" not in adm1.columns:
            cyclones = grid[["DisNo.", "sid"]].drop_duplicates()
            adm1 = adm1.merge(cyclones, on="DisNo.", how="left")
        adm1_path = OUTPUT_DIR / f"model_output/{name}_{strategy}_adm1{BUNDLE_SUFFIX}"
        adm1.to_parquet(adm1_path, index=False)

        logger.info(f"{name}: {len(grid):,} grid rows -> {grid_path.name}; "
                    f"{len(adm1):,} ADM1 rows -> {adm1_path.name}")
        written += [grid_path, adm1_path]
    return written
