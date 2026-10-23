"""Impact predicted from what was forecast, rather than from what happened.

The model is the same two-stage model the rest of the study uses, and it is
fitted the same way: for each storm, on every other cyclone. Only the hazard
changes. Each ensemble member's forecast wind field replaces the observed
wind over the same cells, so a prediction answers what the model would have
said at that forecast cycle, with the uncertainty the ensemble carried.

Everything other than wind is taken as observed. Exposure and terrain do not
depend on the forecast, and rainfall does: a forecast rainfall field would
have to come from the ensemble's gridded output, which this study does not
yet use, so the observed rainfall stands in. Predictions are therefore better
informed about rain than a forecaster would have been, and the wind is the
only quantity whose uncertainty is represented.
"""
import logging

import pandas as pd

from src.config import FEATURES
from src.dynamic_features.process_forecast_tracks import forecast_windfields
from src.models.train import prepare_data
from src.models.two_stage_xgb import TwoStageXGBoost

logger = logging.getLogger(__name__)

TARGET = "perc_affected_pop_grid_region"


def member_frames(base, fields):
    """The event's cells repeated per forecast, carrying that forecast's wind.

    `base` is the event as observed, one row per cell with every feature the
    model reads. Each (cycle, member) takes a copy of it with the wind field
    that member forecast, so the frames differ in the hazard alone.
    """
    if base.empty or fields.empty:
        return pd.DataFrame()

    wind = fields[["id", "cycle", "ensemble_member", "wind_speed", "lead_hours"]]
    base = base.drop(columns=["wind_speed"])
    merged = base.merge(wind, on="id", how="inner")
    if merged.empty:
        logger.warning("no grid cells in common between the event and its "
                       "forecasts; the two may be built on different grids")
    return merged


def predict_event(disno, sid, iso3, event_name, dataset=None, model=None,
                  provider=None, lead_hours=None, min_lead=0):
    """What the model would have predicted at each forecast cycle.

    The storm is withheld whole: training drops every record of its cyclone,
    not only the country-event being predicted, so a storm that struck several
    countries cannot inform its own forecast through one of the others.

    `min_lead` drops cycles issued after the storm arrived. They are collected
    and gridded like any other, since their wind decay confirms the right storm
    was followed, but they are not warnings and do not belong in a skill curve.
    """
    dataset = prepare_data() if dataset is None else dataset
    kwargs = {} if provider is None else {"provider": provider}
    fields = forecast_windfields(disno, sid, iso3, event_name,
                                 lead_hours=lead_hours, **kwargs)
    if fields.empty:
        return pd.DataFrame()
    if min_lead is not None:
        before = fields["cycle"].nunique()
        fields = fields[fields["lead_hours"] >= min_lead]
        dropped = before - fields["cycle"].nunique()
        if dropped:
            logger.info(f"{disno}: {dropped} cycles issued after arrival dropped")
    if fields.empty:
        return pd.DataFrame()

    base = dataset[dataset["DisNo."].astype(str) == disno]
    if base.empty:
        logger.warning(f"{disno}: not in the training dataset")
        return pd.DataFrame()

    test = member_frames(base, fields)
    if test.empty:
        return pd.DataFrame()

    train = dataset[dataset["sid"].astype(str) != str(sid)]
    logger.info(f"{disno}: fitting on {len(train):,} cells from "
                f"{train['sid'].nunique()} cyclones, predicting "
                f"{test['cycle'].nunique()} cycles x "
                f"{test['ensemble_member'].nunique()} members")

    model = TwoStageXGBoost(features=list(FEATURES)) if model is None else model
    predictions = model.train_and_predict(train, test)
    predictions["DisNo."] = disno
    predictions["sid"] = sid
    return predictions


def predict_all(events, dataset=None, provider=None, lead_hours=None,
                min_lead=0):
    """Forecast-time predictions for every event in `events`.

    `events` carries `DisNo.`, `sid`, `iso3` and the event's name. The dataset
    is read once and reused, the fitting being per storm rather than per
    forecast.
    """
    dataset = prepare_data() if dataset is None else dataset
    frames = []
    for _, row in events.iterrows():
        try:
            out = predict_event(row["DisNo."], row["sid"], row["iso3"],
                                row["Event Name"], dataset=dataset,
                                provider=provider, lead_hours=lead_hours,
                                min_lead=min_lead)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"{row['DisNo.']}: prediction failed ({e})")
            continue
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    logger.info(f"{out['DisNo.'].nunique()} events predicted, {len(out):,} rows")
    return out
