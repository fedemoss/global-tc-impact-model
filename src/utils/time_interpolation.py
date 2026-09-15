"""Interpolation of the time-dependent exposure layers to the year of an event.

WorldPop and GHS-SMOD are only published for a handful of anchor years. Using a
single year for every event - WorldPop 2020 and SMOD P2025, as the pipeline did
originally - overstates the exposure of early events and understates late ones,
and the size of that bias is strongly country dependent: between 2000 and 2020
Madagascar's population grew by a factor of 1.8 while Japan's was flat.

These helpers turn the anchors into a per-event-year estimate by piecewise
linear interpolation between the two surrounding anchors.
"""
import numpy as np
import pandas as pd


def interpolation_weights(year, anchor_years, extrapolate=True):
    """Index and fraction of the anchor segment that `year` falls in.

    Returns `(i, frac)` such that the estimate is
    `values[i] + frac * (values[i + 1] - values[i])`. Outside the anchor range
    the nearest segment is continued (`frac` < 0 or > 1) when `extrapolate` is
    True, and clamped to the endpoint otherwise.
    """
    anchors = np.asarray(anchor_years, dtype=float)
    if len(anchors) < 2:
        raise ValueError("at least two anchor years are needed")
    if not np.all(np.diff(anchors) > 0):
        raise ValueError("anchor years must be strictly increasing")

    i = int(np.clip(np.searchsorted(anchors, year, side="right") - 1, 0, len(anchors) - 2))
    frac = (year - anchors[i]) / (anchors[i + 1] - anchors[i])
    if not extrapolate:
        frac = float(np.clip(frac, 0.0, 1.0))
    return i, float(frac)


def interpolate_columns(df, column_template, anchor_years, year, extrapolate=True):
    """Piecewise-linear estimate at `year` of a feature stored one column per anchor.

    `column_template` is formatted with each anchor year, e.g. "population_{}".
    """
    i, frac = interpolation_weights(year, anchor_years, extrapolate=extrapolate)
    low = df[column_template.format(anchor_years[i])].astype("float64")
    high = df[column_template.format(anchor_years[i + 1])].astype("float64")
    return low + frac * (high - low)


def build_yearly_table(
    wide,
    years,
    features,
    anchor_years,
    id_columns=("id", "iso3"),
    extrapolate=True,
    clip=None,
    dtype="float64",
):
    """Long table of interpolated features, one block of rows per year.

    `wide` holds one row per grid cell and one column per (feature, anchor year),
    named "{feature}_{anchor}". The result carries the id columns, a `year`
    column and one column per feature, restricted to the years actually asked
    for so the table stays small.
    """
    id_columns = list(id_columns)
    blocks = []
    for year in sorted({int(y) for y in years}):
        block = wide[id_columns].copy()
        block["year"] = year
        for feature in features:
            estimate = interpolate_columns(
                wide, feature + "_{}", anchor_years, year, extrapolate=extrapolate
            )
            if clip is not None:
                estimate = estimate.clip(lower=clip[0], upper=clip[1])
            block[feature] = estimate.astype(dtype)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def event_year_from_disno(disno):
    """Year of an EM-DAT disaster number ("2010-0123-MEX" -> 2010)."""
    return pd.Series(disno).astype(str).str[:4].astype("Int64")
