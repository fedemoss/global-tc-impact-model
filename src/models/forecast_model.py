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
from pathlib import Path

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
                  provider=None, lead_hours=None, min_lead=0, out_path=None):
    """What the model would have predicted at each forecast cycle.

    The storm is withheld whole: training drops every record of its cyclone,
    not only the country-event being predicted, so a storm that struck several
    countries cannot inform its own forecast through one of the others.

    `min_lead` drops cycles issued after the storm arrived. They are collected
    and gridded like any other, since their wind decay confirms the right storm
    was followed, but they are not warnings and do not belong in a skill curve.

    Returns the predictions, or None when the event was large enough to be
    written to `out_path` a cycle at a time instead of returned whole.
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

    train = dataset[dataset["sid"].astype(str) != str(sid)]
    cells = len(base)
    cycles = sorted(fields["cycle"].unique())
    members = fields["ensemble_member"].nunique()
    logger.info(f"{disno}: fitting on {len(train):,} cells from "
                f"{train['sid'].nunique()} cyclones, predicting "
                f"{len(cycles)} cycles x {members} members over {cells:,} cells")

    model = TwoStageXGBoost(features=list(FEATURES)) if model is None else model
    # Every cycle of this event withholds the same cyclone, so both stages are
    # fitted once here and only the prediction repeats per cycle.
    model.fit(train)

    if cells * members * len(cycles) <= MAX_TEST_ROWS:
        predictions = model.predict(member_frames(base, fields))
        predictions["DisNo."] = disno
        predictions["sid"] = sid
        return predictions

    _write_by_cycle(model, base, fields, cycles, out_path, disno, sid,
                    cells * members)
    return None


#: Above this many rows the event is predicted and written a cycle at a time.
#: A country's cell count varies by three orders of magnitude, so an event
#: over a large country carries a test frame several times the training
#: frame's size, which is what exhausts memory rather than the fitting.
MAX_TEST_ROWS = 2_000_000


def _write_by_cycle(model, base, fields, cycles, out_path, disno, sid,
                    rows_per_cycle):
    """Predict an event cycle by cycle, writing each one before the next.

    Both the frames and the predictions are per cycle: an event over a large
    country spans tens of millions of rows across its cycles, so neither
    building them all up front nor holding the results to concatenate at the
    end fits in memory. Each cycle is appended as a row group and released.
    """
    import gc

    import pyarrow as pa
    import pyarrow.parquet as pq

    if out_path is None:
        raise ValueError(f"{disno}: too large to predict in one frame "
                         f"({rows_per_cycle * len(cycles):,} rows) and no "
                         f"out_path was given to stream it to")

    logger.info(f"{rows_per_cycle * len(cycles):,} rows over {len(cycles)} "
                f"cycles; each is predicted and written on its own")

    writer, schema, written = None, None, 0
    try:
        for cycle in cycles:
            chunk = member_frames(base, fields[fields["cycle"] == cycle])
            if chunk.empty:
                continue
            part = model.predict(chunk)
            part["DisNo."] = disno
            part["sid"] = sid

            table = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            else:
                # A cycle's categories can differ from the first cycle's, which
                # would otherwise make its dictionary columns a different type.
                table = table.cast(schema)
            writer.write_table(table)
            written += len(part)
            del chunk, part, table
            gc.collect()
    finally:
        if writer is not None:
            writer.close()
    return written


def predict_all(events, out_dir, dataset=None, provider=None, lead_hours=None,
                min_lead=0, recompute=False):
    """Forecast-time predictions for every event in `events`.

    `events` carries `DisNo.`, `sid`, `iso3` and the event's name. The dataset
    is read once and reused, the fitting being per storm rather than per
    forecast.

    Each event is written as it finishes rather than held until the end.
    Every event contributes tens of thousands of grid predictions, so keeping
    them all alongside the training frame costs more memory than the fitting
    itself; writing as we go also makes the run resumable.

    An event is written through a temporary name and moved into place once it
    is complete, so a run interrupted mid-event leaves nothing that the resume
    check would mistake for a finished one.
    """
    import gc
    import os

    dataset = prepare_data() if dataset is None else dataset
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done, failed = 0, []
    for _, row in events.iterrows():
        disno = row["DisNo."]
        target = out_dir / f"forecast_{disno}.parquet"
        if target.exists() and not recompute:
            done += 1
            continue
        partial = target.with_suffix(".parquet.partial")
        try:
            out = predict_event(disno, row["sid"], row["iso3"],
                                row["Event Name"], dataset=dataset,
                                provider=provider, lead_hours=lead_hours,
                                min_lead=min_lead, out_path=partial)
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f"{disno}: prediction failed ({e})")
            partial.unlink(missing_ok=True)
            failed.append(disno)
            continue
        if out is None:
            # Written cycle by cycle; nothing was returned to write here.
            if partial.exists():
                os.replace(partial, target)
                done += 1
        elif not out.empty:
            out.to_parquet(partial, index=False)
            os.replace(partial, target)
            done += 1
        del out
        gc.collect()

    logger.info(f"{done} events predicted into {out_dir}"
                + (f"; {len(failed)} failed" if failed else ""))
    return done


def compile_predictions(out_dir):
    """The per-event predictions as one frame, read back from disk.

    Suitable only where the events are few or small: the whole study runs to
    hundreds of millions of rows, which is what `write_compiled` is for.
    """
    files = sorted(Path(out_dir).glob("forecast_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _as_plain_strings(schema):
    """The same schema with dictionary columns widened to plain strings.

    Each event is written with its own categories -- its country's units, its
    own cycles -- so two events' dictionary columns are different Arrow types
    and cannot share one file's schema. Decoding them makes the events
    concatenable at the cost of a larger file.
    """
    import pyarrow as pa

    fields = [f.with_type(pa.string()) if pa.types.is_dictionary(f.type) else f
              for f in schema]
    return pa.schema(fields)


def write_compiled(out_dir, dest, batch_rows=500_000):
    """Join the per-event predictions into one parquet without loading them.

    Returns the number of rows and events written. The events are streamed a
    batch at a time: compiling by concatenation needs the whole study resident
    at once, which is far larger than the memory that predicted it.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(out_dir).glob("forecast_*.parquet"))
    if not files:
        return 0, 0

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".parquet.partial")

    schema = _as_plain_strings(pq.ParquetFile(files[0]).schema_arrow)
    rows, writer = 0, None
    try:
        writer = pq.ParquetWriter(partial, schema)
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_rows):
                table = pa.Table.from_batches([batch]).cast(schema)
                writer.write_table(table)
                rows += table.num_rows
                del table
    finally:
        if writer is not None:
            writer.close()
    os.replace(partial, dest)
    return rows, len(files)
