"""Skill of the forecast-time predictions, read against how early they were made.

The study predicts every grid cell of an event once per ensemble member per
forecast cycle. This reduces that to something a figure can carry, scoring
precision and recall in bins of lead time.

The ensemble is summarised in a fixed order of operations, which differs
between the two quantities on purpose. The central estimate takes the median
across members at grid level and aggregates that field to ADM1. The band
aggregates each member to ADM1 first and takes the quartiles across members,
because a sum of per-cell quantiles is not a quantile of the sum.

Two models are compared on identical forecasts. The two-stage model's
prediction travels in the study's output; the windspeed-exposed baseline is a
rule over the wind field, so it is evaluated here rather than re-run -- every
cell above the threshold has its whole population affected.

The per-event files run to tens of millions of rows, so an event is read one
forecast cycle at a time and leaves only its ADM1 summary behind.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.aggregation import (IMPACT_THRESHOLD_HIGH,
                                        IMPACT_THRESHOLD_LOW, exceeds)
from src.models.baselines import WIND_THRESHOLD

logger = logging.getLogger(__name__)

TARGET = "perc_affected_pop_grid_region"
ADM1_KEYS = ["DisNo.", "GID_0", "GID_1"]

#: Columns the summary needs; the rest of the study's width is never read.
READ_COLUMNS = ["DisNo.", "GID_0", "GID_1", "id", "cycle", "ensemble_member",
                "lead_hours", "population", "wind_speed", "prediction_perc",
                TARGET]

#: The ensemble is summarised by its median, with the quartiles as a band.
#: The median is the forecast a user would act on; the band is the spread the
#: ensemble carried, not a confidence interval on the metric.
LOWER_Q, UPPER_Q = 0.25, 0.75

#: Lead-time bins, in hours before the storm reached the country. Cycles issued
#: further ahead than the last edge are dropped rather than piled into it.
LEAD_EDGES = tuple(range(0, 132, 12))

#: Support a bin needs before its scores are reported. The archive thins with
#: lead time, and the longest bins come down to a handful of storms: there,
#: precision reaches 1.0 on a few calls and recall falls to 0 because nothing
#: was called at all. Neither is a statement about skill, so a bin that cannot
#: support one is left unscored rather than drawn.
MIN_EVENTS = 20
MIN_POSITIVES = 20

THRESHOLDS = (IMPACT_THRESHOLD_LOW, IMPACT_THRESHOLD_HIGH)

#: Column stems: the two models, each as median and the two quartiles.
VARIANTS = ("median", "q25", "q75")
MODELS = ("xgb", "wind")


def _cycles(path):
    """The forecast cycles present in one event's file."""
    column = pq.read_table(path, columns=["cycle"]).column("cycle")
    return pd.Series(column.to_pandas()).dropna().unique()


def _member_quantities(frame):
    """People affected per cell per member, under each model.

    The baseline is a rule rather than a prediction: a cell above the wind
    threshold has its whole population affected, and the rule is applied to
    each member's own wind field before anything is collapsed.
    """
    frame = frame.copy()
    frame["prediction_ppl"] = frame["prediction_perc"].clip(0, 100) \
        * frame["population"] / 100
    frame["wind_ppl"] = np.where(frame["wind_speed"] > WIND_THRESHOLD,
                                 frame["population"], 0.0)
    return frame


def _central_from_members(frame):
    """The central estimate: a median across members per cell, then aggregated.

    For the baseline the median is taken over the wind the rule reads and the
    rule applied to that, rather than over the rule's own output: the median
    of a 0/1 indicator would only ever be 0 or 1.
    """
    keys = ADM1_KEYS + ["id"]
    cells = frame.groupby(keys, observed=True).agg(
        pred_ppl=("prediction_ppl", "median"),
        wind=("wind_speed", "median"),
        population=("population", "first"),
        actual_perc=(TARGET, "first"),
        members=("ensemble_member", "size")).reset_index()

    cells["wind_ppl"] = np.where(cells["wind"] > WIND_THRESHOLD,
                                 cells["population"], 0.0)
    cells["actual_ppl"] = cells["actual_perc"] * cells["population"] / 100

    units = cells.groupby(ADM1_KEYS, observed=True).agg(
        pred_ppl=("pred_ppl", "sum"), wind_ppl=("wind_ppl", "sum"),
        actual_ppl=("actual_ppl", "sum"), population=("population", "sum"),
        actual_perc=("actual_perc", "max"), cells=("id", "nunique"),
        members=("members", "max")).reset_index()
    units = units[units["population"] > 0]
    if units.empty:
        return units

    units["xgb_median"] = (100 * units["pred_ppl"]
                           / units["population"]).clip(0, 100)
    units["wind_median"] = (100 * units["wind_ppl"]
                            / units["population"]).clip(0, 100)
    units["reported"] = units["actual_perc"].clip(0, 100)
    return units.drop(columns=["pred_ppl", "wind_ppl"])


def _band_from_members(frame):
    """The spread: each member aggregated to ADM1, then the quartiles over them.

    Taken in this order rather than as a quantile per cell, because a sum of
    per-cell quantiles is not a quantile of the sum: it describes a field no
    member forecast, and understates a unit's total whenever members put their
    impact in different cells. What a member implies for a whole unit is the
    quantity whose spread this band is meant to show.
    """
    per_member = frame.groupby(ADM1_KEYS + ["ensemble_member"],
                               observed=True).agg(
        pred_ppl=("prediction_ppl", "sum"),
        wind_ppl=("wind_ppl", "sum"),
        population=("population", "sum")).reset_index()
    per_member = per_member[per_member["population"] > 0]
    if per_member.empty:
        return pd.DataFrame(columns=ADM1_KEYS)

    per_member["xgb"] = (100 * per_member["pred_ppl"]
                         / per_member["population"]).clip(0, 100)
    per_member["wind"] = (100 * per_member["wind_ppl"]
                          / per_member["population"]).clip(0, 100)

    band = (per_member.groupby(ADM1_KEYS, observed=True)[["xgb", "wind"]]
            .quantile([LOWER_Q, UPPER_Q]).unstack())
    band.columns = [f"{model}_{'q25' if q == LOWER_Q else 'q75'}"
                    for model, q in band.columns]
    return band.reset_index()


def event_units(path):
    """One event's ADM1 predictions, one block per forecast cycle.

    Read a cycle at a time: a single event can hold tens of millions of rows,
    which is the reason the study writes them per cycle in the first place.
    """
    out = []
    for cycle in _cycles(path):
        frame = pq.read_table(path, columns=READ_COLUMNS,
                              filters=[("cycle", "==", cycle)]).to_pandas()
        if frame.empty:
            continue
        lead = frame["lead_hours"].iloc[0]
        frame = _member_quantities(frame)
        units = _central_from_members(frame)
        if units.empty:
            continue
        units = units.merge(_band_from_members(frame), on=ADM1_KEYS, how="left")
        units["cycle"] = cycle
        units["lead_hours"] = lead
        out.append(units)
        del frame
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def build_study_table(events_dir, dest=None):
    """Every event's ADM1-by-cycle summary, as one small frame.

    The study's grid predictions are hundreds of millions of rows; what comes
    back here is a few tens of thousands, one per (unit, cycle).
    """
    files = sorted(Path(events_dir).glob("forecast_*.parquet"))
    frames = []
    for i, f in enumerate(files, 1):
        units = event_units(f)
        if not units.empty:
            frames.append(units)
        if i % 25 == 0 or i == len(files):
            logger.info(f"summarised {i}/{len(files)} events")
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dest, index=False)
        logger.info(f"{len(table):,} (unit, cycle) rows -> {dest}")
    return table


def _scores(reported, predicted, threshold):
    """Precision and recall of one threshold's positive class."""
    truth = exceeds(reported, threshold)
    call = exceeds(predicted, threshold)
    tp = int((truth & call).sum())
    fp = int((~truth & call).sum())
    fn = int((truth & ~call).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "positives": int(truth.sum()),
        "called": int(call.sum()),
    }


def metrics_by_lead(table, thresholds=THRESHOLDS, edges=LEAD_EDGES,
                    min_events=MIN_EVENTS, min_positives=MIN_POSITIVES):
    """Precision and recall per lead-time bin, per model, per threshold.

    `lead` is the mean lead of the cycles that fell in the bin rather than the
    bin's edge, so a point sits where its forecasts actually were.

    Bins without the support to be scored are dropped; see `MIN_EVENTS`.
    """
    table = table[(table["lead_hours"] >= edges[0])
                  & (table["lead_hours"] < edges[-1])].copy()
    table["lead_bin"] = pd.cut(table["lead_hours"], bins=list(edges),
                               right=False, labels=list(edges[:-1]))

    rows = []
    for (threshold, lead_bin), group in [
            ((t, b), g) for t in thresholds
            for b, g in table.groupby("lead_bin", observed=True)]:
        row = {"threshold": threshold, "lead_bin": int(lead_bin),
               "lead": group["lead_hours"].mean(), "units": len(group),
               "events": group["DisNo."].nunique(),
               "cycles": group["cycle"].nunique()}
        for model in MODELS:
            for q in VARIANTS:
                s = _scores(group["reported"], group[f"{model}_{q}"], threshold)
                row[f"precision_{model}_{q}"] = s["precision"]
                row[f"recall_{model}_{q}"] = s["recall"]
            row[f"positives_{model}"] = _scores(
                group["reported"], group[f"{model}_median"], threshold)["positives"]
        rows.append(row)

    scored = pd.DataFrame(rows)
    enough = ((scored["events"] >= min_events)
              & (scored["positives_xgb"] >= min_positives))
    if not enough.all():
        thin = scored.loc[~enough, ["threshold", "lead_bin", "events",
                                    "positives_xgb"]]
        logger.info(f"{len(thin)} lead bins left unscored for want of support:\n"
                    f"{thin.to_string(index=False)}")

    return (scored[enough]
            .sort_values(["threshold", "lead_bin"])
            .reset_index(drop=True))


def post_landfall(units, model_table, wind_table,
                  thresholds=THRESHOLDS):
    """The same scores from the observed wind field, for the same units.

    This is where the skill curve ends up once the storm has arrived and the
    hazard is known. It is restricted to the units the forecast study covers,
    so the comparison is against the same places rather than the whole record.
    """
    keep = set(map(tuple, units[ADM1_KEYS].drop_duplicates().to_numpy()))

    def restrict(df):
        mask = [tuple(r) in keep for r in df[ADM1_KEYS].to_numpy()]
        return df[mask]

    rows = []
    for name, df in (("xgb", restrict(model_table)),
                     ("wind", restrict(wind_table))):
        for threshold in thresholds:
            s = _scores(df["reported"], df["predicted"], threshold)
            rows.append({"model": name, "threshold": threshold,
                         "precision": s["precision"], "recall": s["recall"],
                         "units": len(df), "positives": s["positives"]})
    return pd.DataFrame(rows)
