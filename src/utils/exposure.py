"""Loading the time-dependent exposure layers at the year of an event.

Population (WorldPop) and degree of urbanisation (GHS-SMOD) are aggregated one
file per anchor year/epoch by the static_features processors. These loaders turn
those anchors into a value for the year each event actually happened.

Both the model features and the impact target depend on this: the target
`perc_affected_pop_grid_region` divides EM-DAT's "Total Affected" by the
population of the affected admin units, so freezing that denominator at 2020
inflates the denominator - and deflates the reported percentage - for every
event before 2020.
"""
import logging

import pandas as pd

from src.config import (
    POP_ANCHOR_YEARS, SMOD_EPOCH_YEARS, population_grid_path, urbanization_grid_path,
)
from src.utils.time_interpolation import build_yearly_table

# Interpolated from the SMOD anchors; `urban` is rebuilt from its two halves so
# it stays exactly consistent with them.
URBAN_FEATURES = ["rural", "water", "urban_cluster", "urban_centre"]


def _load_anchor_table(iso3, paths_by_year, value_columns):
    """One row per grid cell, one column per (feature, anchor year)."""
    wide = None
    for year, path in paths_by_year.items():
        if not path.exists():
            logging.warning(
                f"{iso3}: missing exposure anchor {path.name}; "
                f"cannot build time-varying exposure"
            )
            return None
        df = pd.read_csv(path)
        df["iso3"] = iso3
        present = [c for c in value_columns if c in df.columns]
        df = df[["id", "iso3"] + present].rename(columns={c: f"{c}_{year}" for c in present})
        wide = df if wide is None else wide.merge(df.drop(columns="iso3"), on="id", how="inner")
    return wide


def _clean_years(years):
    return sorted({int(y) for y in years if pd.notna(y)})


def load_population_by_year(iso3, years):
    """Population per grid cell for each requested year.

    Years past the last WorldPop anchor (2020) continue the 2015->2020 slope;
    estimates are clipped at zero.
    """
    years = _clean_years(years)
    if not years:
        return None
    wide = _load_anchor_table(
        iso3, {y: population_grid_path(iso3, y) for y in POP_ANCHOR_YEARS}, ["population"]
    )
    if wide is None:
        return None
    return build_yearly_table(
        wide, years, ["population"], POP_ANCHOR_YEARS, extrapolate=True, clip=(0, None)
    )


def load_urbanization_by_year(iso3, years):
    """Degree-of-urbanisation shares per grid cell for each requested year.

    The SMOD anchors run to P2025, so post-2020 events are interpolated rather
    than extrapolated and the fractions stay inside [0, 1].
    """
    years = _clean_years(years)
    if not years:
        return None
    anchor_years = sorted(SMOD_EPOCH_YEARS.values())
    epoch_by_year = {year: epoch for epoch, year in SMOD_EPOCH_YEARS.items()}
    wide = _load_anchor_table(
        iso3,
        {y: urbanization_grid_path(iso3, epoch_by_year[y]) for y in anchor_years},
        URBAN_FEATURES,
    )
    if wide is None:
        return None
    out = build_yearly_table(
        wide, years, URBAN_FEATURES, anchor_years, extrapolate=False, clip=(0.0, 1.0)
    )
    out["urban"] = out["urban_cluster"] + out["urban_centre"]
    return out


def load_time_varying_features(iso3, years):
    """Population and degree of urbanisation, one row per (grid cell, year)."""
    population = load_population_by_year(iso3, years)
    urbanization = load_urbanization_by_year(iso3, years)
    if population is None or urbanization is None:
        return None
    return population.merge(urbanization, on=["id", "iso3", "year"], how="inner")
