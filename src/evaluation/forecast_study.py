"""The historical forecast study: what the model would have said beforehand.

Each archived forecast cycle is run through the model as though it were the
present: the ensemble's wind fields stand in for the observed one, the storm
is withheld from training, and the resulting prediction is dated by how long
before the storm's arrival the cycle was issued. Skill read against that lead
time is what the study reports.

The events are taken from the dataset the model is evaluated on, so the study
covers the same storms as the rest of the paper rather than a list fixed by
hand. Which of them the study can actually use is decided by the archive:
centres differ widely in coverage, and none reaches before 2006.
"""
import logging

import pandas as pd

from src.collectors.forecast_collector import (DEFAULT_PROVIDER,
                                               collect_all_forecasts,
                                               event_windows, forecast_root)
from src.config import INPUT_DIR, OUTPUT_DIR
from src.models.forecast_model import predict_all, write_compiled

logger = logging.getLogger(__name__)


def study_events(disnos=None):
    """The events the study may cover, with what the forecast path needs.

    An event needs a storm identifier to be timed against its own track, and
    a reported window to know which cycles to look for.
    """
    dataset = pd.read_parquet(
        INPUT_DIR / "model_input_dataset" / "training_dataset.parquet",
        columns=["DisNo.", "sid", "GID_0"]).drop_duplicates()
    for column in dataset.columns:
        dataset[column] = dataset[column].astype(str)
    dataset = dataset.drop_duplicates("DisNo.").rename(columns={"GID_0": "iso3"})

    events = event_windows(disnos=disnos or set(dataset["DisNo."]))
    events = events.merge(dataset, on="DisNo.", how="inner")
    logger.info(f"{len(events)} events, {events['sid'].nunique()} cyclones, "
                f"{events['iso3'].nunique()} countries, "
                f"{events['start'].dt.year.min()}-{events['start'].dt.year.max()}")
    return events


def collect_study_forecasts(provider=DEFAULT_PROVIDER, disnos=None):
    """Download what the archive holds for the study's events.

    Returns the events that ended up with at least one cycle, which are the
    ones the study can go on to predict.
    """
    events = study_events(disnos)
    counts = collect_all_forecasts(events, provider=provider)
    covered = events[events["DisNo."].map(counts).fillna(0) > 0].copy()
    covered["cycles"] = covered["DisNo."].map(counts)

    out = forecast_root(provider) / "coverage.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_frame().to_csv(out)
    logger.info(f"{len(covered)} of {len(events)} events have forecasts "
                f"-> {out}")
    return covered


def run_forecast_study(provider=DEFAULT_PROVIDER, lead_hours=120.0,
                       events=None, min_lead=0):
    """Predict each collected cycle, and write the predictions out.

    The storm being predicted is withheld from training whole, so a storm that
    struck several countries cannot reach its own forecast through one of the
    others.
    """
    if events is None:
        events = study_events()
        counts = _collected_counts(provider)
        events = events[events["DisNo."].map(counts).fillna(0) > 0]
    if events.empty:
        logger.warning("no events with collected forecasts; nothing to predict")
        return pd.DataFrame()

    out_dir = OUTPUT_DIR / "forecast_study" / provider
    events_dir = out_dir / "events"
    n = predict_all(events, events_dir, provider=provider,
                    lead_hours=lead_hours, min_lead=min_lead)
    if not n:
        logger.warning("no forecast predictions produced")
        return pd.DataFrame()

    # Compiled once at the end, from what reached disk, so an interrupted run
    # keeps the events it finished.
    path = out_dir / "grid_predictions.parquet"
    rows, n_events = write_compiled(events_dir, path)
    logger.info(f"{n_events} events, {rows:,} grid predictions -> {path}")
    return path


def _collected_counts(provider):
    """How many cycles each event has on disk, counted from the files."""
    root = forecast_root(provider)
    if not root.exists():
        return pd.Series(dtype=int)
    return pd.Series({
        folder.name: len(list(folder.glob("*.xml")) + list(folder.glob("*.xml.gz")))
        for folder in root.iterdir() if folder.is_dir()
    })
