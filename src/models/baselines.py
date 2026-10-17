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
    country_events = (
        df.groupby(["GID_0", "DisNo.", "year"], observed=True)[target_name]
        .apply(lambda x: x.unique())
        .reset_index()
    )
    country_events["impact_percent"] = country_events[target_name].apply(
        lambda arr: [v for v in arr if v != 0])
    country_events = country_events.drop(columns=target_name)

    rows = []
    for _, row in country_events.iterrows():
        past = country_events[
            (country_events["GID_0"] == row["GID_0"])
            & (country_events["year"] <= row["year"])
            & (country_events["DisNo."] != row["DisNo."])
        ]
        impacts = [v for sub in past["impact_percent"] for v in sub]
        rows.append((row["GID_0"], row["DisNo."], np.median(impacts) if impacts else 0.0))

    return pd.DataFrame(rows, columns=["GID_0", "DisNo.", "median_pct"])


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
