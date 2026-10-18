import gc
import logging
import os
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import INPUT_DIR, OUTPUT_DIR, resolve_iso3_list
from src.dynamic_features.process_wind_features import basin_lookup
from src.utils.exposure import load_time_varying_features
from src.utils.time_interpolation import event_year_from_disno

def load_grid_gid_mapping(iso3, grid_global=None, adm_global=None):
    """
    Spatially joins grid cell centroids to their GID_1/GID_2 administrative unit.
    Mirrors the enrichment done in process_shdi.py::process_all_shdi, since the
    grid produced by grid_cells.py only carries GID_0 (country-level).

    `grid_global` and `adm_global` let the caller read the two GeoPackages once
    and reuse them across countries; both are read here only when not supplied.
    """
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    adm2_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    if not grid_path.exists() or not adm2_path.exists():
        return pd.DataFrame(columns=["id", "iso3", "GID_1", "GID_2"])

    grid = grid_global if grid_global is not None else gpd.read_file(grid_path)
    grid = grid[grid.iso3 == iso3].copy()
    if grid.empty:
        return pd.DataFrame(columns=["id", "iso3", "GID_1", "GID_2"])

    def national_fallback():
        # Small countries (e.g. ATG) where the 0.1 deg grid is too coarse relative
        # to admin-1 unit size have no usable GID_1 breakdown — evaluate the whole
        # country as a single GID_0 unit instead.
        df_grid_map = grid[["id", "iso3"]].copy()
        df_grid_map["GID_1"] = iso3
        df_grid_map["GID_2"] = None
        return df_grid_map

    adm_country = adm_global if adm_global is not None else gpd.read_file(adm2_path)
    adm_country = adm_country[adm_country.GID_0 == iso3]
    if adm_country.empty:
        return national_fallback()

    grid["centroid"] = grid.geometry.centroid
    enriched = gpd.sjoin(
        grid.set_geometry("centroid"),
        adm_country[["GID_1", "GID_2", "geometry"]],
        how="left",
        predicate="within"
    )
    df_grid_map = enriched[["id", "iso3", "GID_1", "GID_2"]].drop_duplicates(subset="id")

    # If most grid cells fail to fall within any admin-1 polygon, the country's
    # GID_1 breakdown isn't practically usable — fall back to national level
    # rather than keeping a mostly-unmatched, patchy subnational mapping.
    match_rate = df_grid_map["GID_1"].notna().mean()
    if match_rate < 0.5:
        return national_fallback()

    df_grid_map[["GID_1", "GID_2"]] = df_grid_map[["GID_1", "GID_2"]].astype(object)
    return df_grid_map

logger = logging.getLogger(__name__)

# Global CSV layers are country-agnostic but are consumed inside the per-country
# loop, so reading them per country costs hundreds of megabytes of I/O each
# time. Small layers are cached whole; large ones are split by country on first
# use, since holding several tens of millions of rows at once does not fit
# alongside the impact table and the two GeoPackages.
_GLOBAL_LAYER_CACHE = {}
_SPLIT_CHUNK_ROWS = 1_000_000


def _split_dir(path):
    return Path(path).parent / "_by_iso3" / Path(path).stem


def _ensure_split_by_iso3(path, key="iso3", rename=None):
    """Split a global CSV into one CSV per country, once, streaming by chunk."""
    out_dir = _split_dir(path)
    marker = out_dir / "_complete"
    # The split is a cache of `path` and is rebuilt whenever the source is
    # newer, so that a re-run of an upstream stage is picked up here.
    if marker.exists():
        if marker.stat().st_mtime >= Path(path).stat().st_mtime:
            return out_dir
        logger.info(f"{Path(path).name} is newer than its country split; re-splitting")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    for chunk in pd.read_csv(path, chunksize=_SPLIT_CHUNK_ROWS):
        if rename:
            chunk = chunk.rename(columns=rename)
        if key not in chunk.columns:
            shutil.rmtree(out_dir)
            raise KeyError(f"{Path(path).name} has no '{key}' column to split on")
        for iso, group in chunk.groupby(key, observed=True):
            dest = out_dir / f"{iso}.csv"
            group.to_csv(dest, mode="a", header=not dest.exists(), index=False)
        rows += len(chunk)
    marker.touch()
    logger.info(f"Split {Path(path).name} by country: {rows:,} rows")
    return out_dir


def _load_country_layer(path, iso3, key="iso3", rename=None, usecols=None):
    """Return one country's rows from a global CSV, without holding the whole."""
    out_dir = _ensure_split_by_iso3(path, key=key, rename=rename)
    dest = out_dir / f"{iso3}.csv"
    if not dest.exists():
        return None
    if usecols is not None:
        return pd.read_csv(dest, usecols=lambda c: c in set(usecols))
    return pd.read_csv(dest)


def _load_global_layer(path, rename=None):
    """Read a small global CSV once and reuse it for every country."""
    key = str(path)
    if key not in _GLOBAL_LAYER_CACHE:
        df = pd.read_csv(path)
        if rename:
            df = df.rename(columns=rename)
        _GLOBAL_LAYER_CACHE[key] = df
        logger.info(f"Cached global layer {Path(path).name}: {len(df):,} rows")
    return _GLOBAL_LAYER_CACHE[key]

def load_static_features(iso3):
    """
    Loads and merges all static spatial features for a specific country.

    """
    df_static = pd.DataFrame()

    # 1. Gridded Static Layers
    # The SRTM dataset contains elevation, slope, ruggedness, and coastal features.
    # Population and degree of urbanisation are NOT here: both change over the
    # 2000-2022 record and are loaded per event year by
    # load_time_varying_features() instead.
    layers = {
        "SRTM": f"srtm_grid_data_{iso3}.csv",
        "FloodRisk": f"flood_risk_{iso3}.csv",
        "SHDI": f"shdi_grid_{iso3}.csv"
    }

    for folder, file in layers.items():
        path = OUTPUT_DIR / folder / "grid_data" / file
        if path.exists():
            layer_df = pd.read_csv(path)

            if df_static.empty:
                df_static = layer_df
            else:
                # Merge on the unique grid identifier (and iso3 when the layer has it —
                # e.g. JRC's output omits iso3, but "id" alone is already globally unique)
                merge_keys = ["id", "iso3"] if "iso3" in layer_df.columns else ["id"]
                # Drop columns already present (e.g. SHDI's own GID_1/GID_2 helper
                # columns) to avoid pandas _x/_y suffixing on the merge below.
                overlap = [c for c in layer_df.columns if c in df_static.columns and c not in merge_keys]
                df_static = df_static.merge(layer_df.drop(columns=overlap), on=merge_keys, how="left")

    if df_static.empty:
        return pd.DataFrame()

    # 2. Global Static Layers
    # Landslide Risk
    landslide_path = OUTPUT_DIR / "LandSlides" / "global_grid_landslide_risk.csv"
    if landslide_path.exists():
        df_land = _load_global_layer(landslide_path)
        # Filter global data to the specific country during merge
        df_static = df_static.merge(
            df_land[df_land.iso3 == iso3], 
            on=["id", "iso3"], 
            how="left"
        )

    # Storm Surge Risk
    surge_path = OUTPUT_DIR / "StormSurges" / "grid_data" / "global_grid_storm_surges_risk.csv"
    if surge_path.exists():
        df_surge = _load_global_layer(surge_path, rename={"GID_0": "iso3"})
        df_static = df_static.merge(
            df_surge[df_surge.iso3 == iso3], 
            on=["id", "iso3"], 
            how="left"
        )

    # GID_1/GID_2 are internal helper columns from SHDI's own admin mapping — the
    # authoritative ones already live on df_dyn via load_grid_gid_mapping.
    return df_static.drop(columns=["GID_1", "GID_2"], errors="ignore")


def build_country_dataset(iso3, df_impact_country,
                          grid_global=None, adm_global=None):
    """
    Core Logic: Maps grid-level hazard intensity to sub-national impact targets.

    `df_impact_country` holds only this country's rows of the grid-level impact
    table, and supplies both the target and the DisNo. event identifier.
    `grid_global` and `adm_global` are passed through to load_grid_gid_mapping
    so the GeoPackages are read once per run.
    """
    # 1. Load Wind Exposure (Grid level: 'id')
    wind_path = OUTPUT_DIR / "IBTRACS" / "standard" / f"windfield_data_{iso3}.csv"
    if not wind_path.exists(): return None
    df_wind = pd.read_csv(wind_path).rename(columns={"grid_point_id": "id", "GID_0": "iso3"})
    # The wind field is a property of the cyclone track: one value per
    # (sid, cell). The collector iterates the EM-DAT metadata, which holds a
    # row per country-event, so a cyclone with two records in one country is
    # written once per record. Collapsing them here keeps the impact join from
    # pairing each copy with each record.
    n_dup_wind = int(df_wind.duplicated(subset=["sid", "id"]).sum())
    if n_dup_wind:
        logging.warning(
            f"{iso3}: wind field has {n_dup_wind:,} duplicate (sid, id) rows of "
            f"{len(df_wind):,}; keeping one per cyclone-cell"
        )
        df_wind = df_wind.drop_duplicates(subset=["sid", "id"])

    # 2. Load Grid-to-GID Mapping (Crucial for sub-national alignment)
    df_grid_map = load_grid_gid_mapping(iso3, grid_global=grid_global, adm_global=adm_global)
    if df_grid_map.empty: return None

    # 3. Merge Wind with Grid Mapping
    df_dyn = df_wind.merge(df_grid_map, on=["id", "iso3"], how="inner")

    # 4. Merge with Impact Targets (Inner Join on sid AND GID)
    # This ensures a grid cell only gets a 'Total Affected' value if its parent GID was affected
    if df_impact_country is None or df_impact_country.empty:
        return None
    df_impact_country = df_impact_country.copy()
    missing = {"DisNo.", "impact_capped"} - set(df_impact_country.columns)
    if missing:
        raise KeyError(
            f"{iso3}: impact table is missing {sorted(missing)}. It predates the "
            f"current schema -- re-run the dynamic stage to regenerate "
            f"EMDAT/impact_data.csv."
        )
    # A DisNo. ends in the ISO3 of the country the event was recorded for, so
    # it is only ever attributed to that country's cells. An event reaching a
    # different country means its administrative units were geolocated outside
    # it, which the geometry step can do for a territory spanning the
    # antimeridian, where a centroid taken in plain longitude falls far from
    # the units it summarises.
    foreign = df_impact_country["DisNo."].astype(str).str[-3:] != iso3
    if foreign.any():
        logging.warning(
            f"{iso3}: dropping {int(foreign.sum()):,} impact rows belonging to "
            f"{sorted(set(df_impact_country.loc[foreign, 'DisNo.']))[:5]}"
        )
        df_impact_country = df_impact_country[~foreign]
        if df_impact_country.empty:
            return None

    df_impact_country[["GID_1", "GID_2"]] = df_impact_country[["GID_1", "GID_2"]].astype(object)

    # The impact table holds one row per (country-event, cell). The key is
    # (DisNo., id), not (sid, id): EM-DAT sometimes records two disasters for
    # one cyclone in one country, with different affected counts, and those are
    # distinct events rather than duplicates. Anything repeating (DisNo., id)
    # would multiply the join below instead of failing.
    n_dup = int(df_impact_country.duplicated(subset=["DisNo.", "id"]).sum())
    if n_dup:
        logging.warning(
            f"{iso3}: impact table has {n_dup:,} duplicate (DisNo., id) rows of "
            f"{len(df_impact_country):,}; collapsing to one row per event-cell"
        )
        df_impact_country = df_impact_country.drop_duplicates(subset=["DisNo.", "id"])
    is_national_level = (df_grid_map["GID_1"] == iso3).all()
    if is_national_level:
        # GID_1 doesn't resolve subnationally for this country (see load_grid_gid_mapping) —
        # every grid cell shares the same national impact figures for a given event.
        impact_by_event = df_impact_country.groupby("sid").agg({
            "DisNo.": "first", "GID_0": "first", "level": "first",
            "Total Affected": "max", "perc_affected_pop_grid_region": "max",
            "impact_capped": "max"
        }).reset_index()
        df_dyn = df_dyn.merge(impact_by_event, on="sid", how="inner")
    else:
        # DisNo. is taken from the impact table, which holds only this country's
        # rows and so names the country-event the cell actually belongs to. The
        # IBTrACS metadata is global and carries one DisNo. per country a
        # cyclone hit, so joining it on sid alone attaches every affected
        # country's DisNo. to every cell and multiplies the row count by the
        # number of countries the cyclone crossed.
        #
        # Join on the grid cell itself: the impact table carries a row per
        # (event, cell), so the result is one row per event-cell. Joining on
        # (sid, GID_1, GID_2) would be many-to-many, since hundreds of cells
        # share an administrative unit on each side.
        df_dyn = df_dyn.merge(
            df_impact_country[["sid", "id", "DisNo.", "GID_0", "level",
                               "Total Affected", "perc_affected_pop_grid_region",
                               "impact_capped"]],
            on=["sid", "id"],
            how="inner"
        )

    # Each output row is one country-event observed at one grid cell. The join
    # is on sid, so a cyclone with two EM-DAT records yields two rows per cell,
    # one per event; a repeated (DisNo., id) instead means a key was not unique
    # and the join has multiplied the row count.
    n_dup_out = int(df_dyn.duplicated(subset=["DisNo.", "id"]).sum())
    if n_dup_out:
        logging.error(
            f"{iso3}: impact join produced {n_dup_out:,} duplicate (DisNo., id) "
            f"rows of {len(df_dyn):,}; the join keys are not unique"
        )

    if df_dyn.empty: return None

    # 5. Merge Rainfall
    rain_path = OUTPUT_DIR / "PPS" / f"rainfall_data_{iso3}.csv"
    if rain_path.exists():
        # Missing rainfall is left missing here; every fill happens in one
        # place in compile_global_dataset, after the gaps have been counted.
        df_dyn = df_dyn.merge(pd.read_csv(rain_path), on=["id", "iso3", "sid"], how="left")

    # 6. Merge Historical Frequency
    hist_path = OUTPUT_DIR / "features" / "historical_events_feature.csv"
    if hist_path.exists():
        df_hist = _load_country_layer(hist_path, iso3)
        if df_hist is None:
            df_hist = pd.DataFrame(columns=["id", "iso3", "DisNo.", "N_events_5_years"])
        df_dyn = df_dyn.merge(df_hist, on=["id", "iso3", "DisNo."], how="left")
    else:
        # Silently skipping this merge is how N_events_5_years goes missing from
        # the training set without anything failing.
        logger.warning(
            f"{hist_path} not found -- N_events_5_years will be absent for {iso3}. "
            f"Run generate_all_historical_features() with no iso3_filter."
        )

    # 7. Merge Static Features
    df_static = load_static_features(iso3)
    if not df_static.empty:
        df_final = df_dyn.merge(df_static, on=["id", "iso3"], how="left")
    else:
        df_final = df_dyn

    # 8. Merge Time-Varying Exposure (population, degree of urbanisation)
    # Keyed on the event year, so a 2000 storm is scored against the population
    # and urban extent of 2000 rather than of 2020/2025.
    df_final["year"] = event_year_from_disno(df_final["DisNo."])
    df_time = load_time_varying_features(iso3, df_final["year"].dropna().unique())
    if df_time is None:
        logging.error(f"{iso3}: no time-varying exposure available, country skipped")
        return None
    n_before = len(df_final)
    df_final = df_final.merge(df_time, on=["id", "iso3", "year"], how="left")
    if len(df_final) != n_before:
        logging.error(f"{iso3}: exposure merge changed row count {n_before} -> {len(df_final)}")

    # 9. Attach the cyclone's basin, the unit the basin-constrained evaluation
    # restricts training to.
    df_basin = basin_lookup()
    if not df_basin.empty:
        n_before = len(df_final)
        df_final = df_final.merge(df_basin, on="sid", how="left")
        if len(df_final) != n_before:
            logging.error(f"{iso3}: basin merge changed row count {n_before} -> {len(df_final)}")
        unmapped = df_final["cyclone_basin"].isna()
        if unmapped.any():
            logging.warning(
                f"{iso3}: {int(unmapped.sum()):,} rows have no basin "
                f"({sorted(set(df_final.loc[unmapped, 'sid']))[:3]})"
            )

    return df_final

def _write_nan_report(nan_counts, row_counts, out_path, filled_with_zero):
    """Record how many values were missing per feature before any filling.

    Writes two files beside the dataset: a per-feature summary and a
    per-country breakdown, so a feature that is fine globally but absent for
    one country is still visible.
    """
    per_country = pd.DataFrame(nan_counts).fillna(0).astype("int64")  # feature x iso3
    total_rows = int(sum(row_counts.values()))
    summary = pd.DataFrame({
        "nan_rows": per_country.sum(axis=1),
        "total_rows": total_rows,
    })
    summary["pct_nan"] = 100 * summary["nan_rows"] / max(total_rows, 1)
    summary["filled_with_zero"] = summary.index.isin(filled_with_zero)
    countries_affected = (per_country > 0).sum(axis=1)
    summary["countries_affected"] = countries_affected
    summary = summary.sort_values("nan_rows", ascending=False)

    summary.to_csv(out_path / "nan_report.csv")
    per_country.T.to_csv(out_path / "nan_report_by_country.csv")

    worst = summary[summary["nan_rows"] > 0]
    if worst.empty:
        logger.info(f"No missing values in any feature across {total_rows:,} rows")
        return
    logger.warning(
        f"Missing values before filling, across {total_rows:,} rows "
        f"({len(worst)} features affected) -> {out_path / 'nan_report.csv'}"
    )
    for feat, r in worst.head(25).iterrows():
        flag = " [filled with 0]" if r["filled_with_zero"] else ""
        logger.warning(
            f"    {feat:<28} {int(r['nan_rows']):>12,} rows "
            f"({r['pct_nan']:6.2f}%)  in {int(r['countries_affected'])} countries{flag}"
        )


def compile_global_dataset():
    """Compiles the master global training dataset."""
    meta_dir = OUTPUT_DIR / "IBTRACS" / "standard"
    meta_files = list(meta_dir.glob("metadata_*.csv"))
    # Presence of the IBTrACS metadata is the signal that the collect stage ran.
    # DisNo. itself comes from the impact table, which carries it per country.
    if not meta_files: return

    impact_path = OUTPUT_DIR / "EMDAT" / "impact_data.csv"
    if not impact_path.exists(): return

    # The grid-level impact table carries a row per (event, cell) for the whole
    # globe. It is split by country on first use rather than held in memory,
    # since the per-country slice is all any single merge below consumes.
    _ensure_split_by_iso3(impact_path)
    # Columns the per-country merges consume. `id` is the grid cell and is the
    # key the impact join uses; the rest are carried through to the output.
    impact_cols = ["sid", "id", "DisNo.", "GID_0", "GID_1", "GID_2", "iso3",
                   "level", "Total Affected", "perc_affected_pop_grid_region",
                   "impact_capped"]

    # Read the two GeoPackages once and reuse them for every country, rather
    # than re-reading the global grid inside load_grid_gid_mapping each time.
    grid_path = INPUT_DIR / "GRID" / "merged" / "global_grid_land_overlap.gpkg"
    adm2_path = INPUT_DIR / "SHP" / "global_shapefile_GID_adm2.gpkg"
    grid_global = gpd.read_file(grid_path) if grid_path.exists() else None
    adm_global = gpd.read_file(adm2_path) if adm2_path.exists() else None

    # Every zero-fill happens here, after the gaps have been counted into
    # nan_report.csv, so the extent of each fill is recorded alongside the
    # dataset.
    fill_0 = ["mean_elev", "mean_slope", "mean_rug", "urban", "rural", "water",
              "storm_tide_rp_0010", "landslide_risk_sum", "flood_risk", "population",
              "coast_length_meters",
              # a storm with no IMERG coverage, and a cell with no prior events
              "rainfall_max_24h", "N_events_5_years"]

    out_path = INPUT_DIR / "model_input_dataset"
    out_path.mkdir(parents=True, exist_ok=True)
    parts_dir = out_path / "_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    # Countries are written to per-country Parquet parts rather than held in a
    # list for one concatenation. Only a minority of countries have wind data,
    # but their combined frame is still far larger than the per-country peak,
    # and concatenating requires a second copy of all of it at once.
    iso3_list = resolve_iso3_list()
    columns = set()
    shdi_chunks = []
    written = 0
    nan_counts, row_counts = {}, {}
    for n, iso3 in enumerate(iso3_list, 1):
        df_impact_country = _load_country_layer(impact_path, iso3, usecols=impact_cols)
        df_country = build_country_dataset(
            iso3, df_impact_country,
            grid_global=grid_global, adm_global=adm_global)
        if df_country is None or df_country.empty:
            continue
        # Count missing values before any filling. Once a gap has been filled it
        # is indistinguishable from a real zero, and zero is a substantive claim
        # for several of these features: no elevation reads as sea level, no
        # population as an empty cell.
        nan_counts[iso3] = df_country.isna().sum()
        row_counts[iso3] = len(df_country)

        for col in fill_0:
            if col in df_country.columns:
                df_country[col] = df_country[col].fillna(0)
        if "shdi" in df_country.columns:
            shdi_chunks.append(df_country["shdi"].dropna())
        columns.update(df_country.columns)
        df_country.to_parquet(parts_dir / f"{iso3}.parquet", index=False)
        written += 1
        logger.info(f"[{n}/{len(iso3_list)}] {iso3}: {len(df_country):,} rows")
        del df_country

    if nan_counts:
        _write_nan_report(nan_counts, row_counts, out_path, fill_0)

    if not written:
        shutil.rmtree(parts_dir)
        return

    # The SHDI median is the one statistic that needs every country, so it is
    # computed from the collected column alone rather than the full frames.
    shdi_median = pd.concat(shdi_chunks).median() if shdi_chunks else None
    del shdi_chunks, grid_global, adm_global
    gc.collect()

    columns = sorted(columns)
    parts = sorted(parts_dir.glob("*.parquet"))

    # ParquetWriter fixes its schema from the first table it is given, but the
    # per-country parts disagree on dtype: a column that is entirely missing for
    # one country arrives as null or float, and as string or int for another.
    # Unifying the part schemas first, then binding every table to the result,
    # keeps the write consistent without loading the parts twice.
    unified = pa.unify_schemas(
        [pq.read_schema(part) for part in parts], promote_options="permissive")
    unified = pa.schema([unified.field(name) for name in columns
                         if name in unified.names])

    final_path = out_path / "training_dataset.parquet"
    writer = None
    total_rows = 0
    try:
        for part in parts:
            df_part = pd.read_parquet(part).reindex(columns=columns)
            if shdi_median is not None and "shdi" in df_part.columns:
                df_part["shdi"] = df_part["shdi"].fillna(shdi_median)
            table = pa.Table.from_pandas(
                df_part, schema=unified, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(final_path, unified)
            writer.write_table(table)
            total_rows += len(df_part)
            del df_part, table
    finally:
        if writer is not None:
            writer.close()

    shutil.rmtree(parts_dir)
    logger.info(f"Master training dataset saved. Total rows: {total_rows:,}")


if __name__ == "__main__":
    from src.utils.logging_setup import configure_logging
    configure_logging()
    compile_global_dataset()