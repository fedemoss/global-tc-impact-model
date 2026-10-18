"""Regression tests for the two defects found in the global training dataset.

1. `DisNo.` was taken from the global IBTrACS metadata and joined on `sid`
   alone, so every country a cyclone crossed contributed a row to every grid
   cell of every other country it crossed.
2. `perc_affected_pop_grid_region` had no upper bound, so a reported affected
   count larger than the population it was attributed to produced shares of
   several thousand percent.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _impact_slice(iso3, disno, n_cells=3, total_affected=1000.0, perc=10.0):
    """One country's slice of the grid-level impact table for a single cyclone."""
    return pd.DataFrame({
        "sid": ["2013306N07162"] * n_cells,
        "id": [f"{iso3}_{i}" for i in range(n_cells)],
        "DisNo.": [disno] * n_cells,
        "GID_0": [iso3] * n_cells,
        "GID_1": [f"{iso3}.1_1"] * n_cells,
        "GID_2": [f"{iso3}.1.1_1"] * n_cells,
        "iso3": [iso3] * n_cells,
        "level": ["ADM1"] * n_cells,
        "Total Affected": [total_affected] * n_cells,
        "perc_affected_pop_grid_region": [perc] * n_cells,
        "impact_capped": [False] * n_cells,
    })


class TestDisNoAttribution:
    """A cyclone crossing several countries must not fan out across them."""

    def test_impact_slice_is_the_only_source_of_disno(self):
        from src import dataset_builder

        src = Path(dataset_builder.__file__).read_text()
        assert 'df_meta[["sid", "DisNo."]], on="sid"' not in src, (
            "DisNo. is being joined from the global metadata on sid alone, "
            "which re-introduces the cross-country fan-out"
        )

    def test_build_country_dataset_takes_no_global_metadata(self):
        import inspect

        from src.dataset_builder import build_country_dataset

        params = list(inspect.signature(build_country_dataset).parameters)
        assert "df_meta" not in params, (
            "build_country_dataset still accepts the global metadata frame"
        )

    def test_join_on_sid_and_id_preserves_row_count(self):
        """The merge the builder performs, in isolation."""
        # Three grid cells in Haiti, hit by a cyclone that also hit four
        # other countries.
        df_dyn = pd.DataFrame({
            "sid": ["2013306N07162"] * 3,
            "id": [f"HTI_{i}" for i in range(3)],
            "iso3": ["HTI"] * 3,
            "wind_speed": [40.0, 55.0, 61.0],
        })
        hti = _impact_slice("HTI", "2013-0433-HTI")

        merged = df_dyn.merge(
            hti[["sid", "id", "DisNo.", "GID_0", "level", "Total Affected",
                 "perc_affected_pop_grid_region", "impact_capped"]],
            on=["sid", "id"], how="inner")

        assert len(merged) == len(df_dyn)
        assert set(merged["DisNo."]) == {"2013-0433-HTI"}
        assert (merged["DisNo."].str[-3:] == merged["GID_0"]).all(), (
            "every cell must carry the DisNo. of its own country"
        )

    def test_global_metadata_join_would_multiply_rows(self):
        """Characterise the old behaviour, so the fix has something to beat."""
        df_dyn = pd.DataFrame({
            "sid": ["2013306N07162"] * 3,
            "id": [f"HTI_{i}" for i in range(3)],
        })
        # Haiyan appears once per country hit in the global metadata.
        df_meta = pd.DataFrame({
            "sid": ["2013306N07162"] * 5,
            "DisNo.": ["2013-0433-PHL", "2013-0433-CHN", "2013-0433-VNM",
                       "2013-0433-HTI", "2013-0433-PLW"],
        })
        old = df_dyn.merge(df_meta, on="sid", how="left")
        assert len(old) == 15  # 3 cells x 5 countries
        assert old["DisNo."].nunique() == 5


class TestTwoEventsPerCyclone:
    """EM-DAT records two disasters for one cyclone in one country (observed
    for PHL 2006209N13130 and 2007257N16134, with different affected counts).
    Both are real events and both must survive."""

    def test_the_uniqueness_key_is_disno_not_sid(self):
        a = _impact_slice("PHL", "2006-0410-PHL", total_affected=15_000, perc=46.48)
        b = _impact_slice("PHL", "2006-0415-PHL", total_affected=476_027, perc=100.0)
        impact = pd.concat([a, b], ignore_index=True)

        assert impact.duplicated(subset=["sid", "id"]).any(), (
            "(sid, id) repeats across the two events -- it is not the key"
        )
        assert not impact.duplicated(subset=["DisNo.", "id"]).any(), (
            "(DisNo., id) is the key and must be unique"
        )

    def test_both_events_survive_the_join(self):
        """Deduplicating on (sid, id) would discard the larger disaster."""
        df_dyn = pd.DataFrame({
            "sid": ["2013306N07162"] * 3,
            "id": [f"PHL_{i}" for i in range(3)],
            "wind_speed": [40.0, 55.0, 61.0],
        })
        impact = pd.concat([
            _impact_slice("PHL", "2006-0410-PHL", total_affected=15_000, perc=46.48),
            _impact_slice("PHL", "2006-0415-PHL", total_affected=476_027, perc=100.0),
        ], ignore_index=True)

        out = df_dyn.merge(
            impact[["sid", "id", "DisNo.", "GID_0", "level", "Total Affected",
                    "perc_affected_pop_grid_region", "impact_capped"]],
            on=["sid", "id"], how="inner")

        assert out["DisNo."].nunique() == 2
        assert len(out) == 6  # 3 cells x 2 events
        assert not out.duplicated(subset=["DisNo.", "id"]).any()
        assert sorted(out["Total Affected"].unique()) == [15_000.0, 476_027.0]

    def test_guard_collapses_only_true_repeats(self):
        """A genuinely repeated (DisNo., id) is the thing worth collapsing."""
        slice_ = _impact_slice("PHL", "2006-0410-PHL")
        impact = pd.concat([slice_, slice_], ignore_index=True)

        n_dup = int(impact.duplicated(subset=["DisNo.", "id"]).sum())
        assert n_dup == 3
        assert len(impact.drop_duplicates(subset=["DisNo.", "id"])) == 3


class TestWindFieldDeduplication:
    """The collector iterates EM-DAT records, so a cyclone with two records in
    one country has its wind field written twice (observed: PHL, 7,548
    byte-identical duplicate rows for 2006209N13130 and 2007257N16134)."""

    def test_duplicate_wind_rows_square_the_impact_join(self):
        wind = pd.DataFrame({
            "sid": ["s1"] * 6,
            "id": ["PHL_0", "PHL_1", "PHL_2"] * 2,   # the cyclone written twice
            "wind_speed": [40.0, 55.0, 61.0] * 2,
        })
        impact = pd.concat([
            _impact_slice("PHL", "2006-0410-PHL"),
            _impact_slice("PHL", "2006-0415-PHL"),
        ], ignore_index=True).assign(sid="s1")

        joined = wind.merge(
            impact[["sid", "id", "DisNo."]], on=["sid", "id"], how="inner")
        assert len(joined) == 12          # 3 cells x 2 records x 2 copies
        assert int(joined.duplicated(subset=["DisNo.", "id"]).sum()) == 6

        deduped = wind.drop_duplicates(subset=["sid", "id"]).merge(
            impact[["sid", "id", "DisNo."]], on=["sid", "id"], how="inner")
        assert len(deduped) == 6          # 3 cells x 2 records
        assert not deduped.duplicated(subset=["DisNo.", "id"]).any()
        assert deduped["DisNo."].nunique() == 2

    def test_dedup_keeps_the_wind_values(self):
        wind = pd.DataFrame({
            "sid": ["s1"] * 4, "id": ["c0", "c1"] * 2,
            "wind_speed": [40.0, 55.0] * 2,
        })
        out = wind.drop_duplicates(subset=["sid", "id"])
        assert out["wind_speed"].tolist() == [40.0, 55.0]


class TestImpactCap:
    @pytest.mark.parametrize("affected,pop,expected,flagged", [
        (1_000, 10_000, 10.0, False),     # ordinary
        (10_000, 10_000, 100.0, False),   # exactly the population
        (354_000, 10_000, 100.0, True),   # Haiyan-style over-attribution
    ])
    def test_cap_and_flag(self, affected, pop, expected, flagged):
        df = pd.DataFrame({"id": ["c0"], "Total Affected": [float(affected)]})
        perc = (100 * df["Total Affected"]) / pop
        df["impact_capped"] = perc > 100
        df["perc_affected_pop_grid_region"] = perc.clip(upper=100)

        assert df["perc_affected_pop_grid_region"].iloc[0] == pytest.approx(expected)
        assert bool(df["impact_capped"].iloc[0]) is flagged

    def test_process_emdat_caps_the_target(self):
        from src.dynamic_features import process_emdat

        src = Path(process_emdat.__file__).read_text()
        assert "perc.clip(upper=100)" in src
        assert '"impact_capped"' in src

    def test_unaffected_cells_are_not_flagged(self):
        """Cells outside the affected units get 0% and no flag."""
        df_merged = pd.DataFrame({"id": ["c0", "c1", "c2"]})
        df_affected = pd.DataFrame({
            "id": ["c0"],
            "perc_affected_pop_grid_region": [100.0],
            "impact_capped": [True],
        })
        out = df_merged.merge(
            df_affected[["id", "perc_affected_pop_grid_region", "impact_capped"]]
            .drop_duplicates(subset=["id"]),
            on="id", how="left",
        ).fillna({"perc_affected_pop_grid_region": 0})
        out["impact_capped"] = out["impact_capped"].eq(True)

        assert out["perc_affected_pop_grid_region"].tolist() == [100.0, 0.0, 0.0]
        assert out["impact_capped"].tolist() == [True, False, False]
        assert out["impact_capped"].dtype == bool

    def test_flag_survives_the_csv_round_trip(self, tmp_path):
        """The impact table is streamed to CSV and read back by the builder."""
        path = tmp_path / "impact_data.csv"
        pd.DataFrame({
            "sid": ["s1", "s1"], "id": ["c0", "c1"],
            "perc_affected_pop_grid_region": [100.0, 12.5],
            "impact_capped": [True, False],
        }).to_csv(path, index=False)

        back = pd.read_csv(path)
        assert back["impact_capped"].dtype == bool
        assert back["impact_capped"].tolist() == [True, False]
        # the national-level branch aggregates the flag per cyclone
        agg = back.groupby("sid").agg({"impact_capped": "max"})
        assert bool(agg["impact_capped"].iloc[0]) is True


class TestSplitCacheInvalidation:
    def test_split_is_rebuilt_when_source_is_newer(self, tmp_path):
        import os

        from src.dataset_builder import _ensure_split_by_iso3, _split_dir

        path = tmp_path / "impact_data.csv"
        pd.DataFrame({"iso3": ["ATG"], "v": [1]}).to_csv(path, index=False)
        out = _ensure_split_by_iso3(path)
        assert pd.read_csv(out / "ATG.csv")["v"].tolist() == [1]

        # Regenerate the source, as a re-run of the dynamic stage would.
        pd.DataFrame({"iso3": ["ATG"], "v": [2]}).to_csv(path, index=False)
        os.utime(path, (_now := 2_000_000_000, _now))

        out = _ensure_split_by_iso3(path)
        assert pd.read_csv(out / "ATG.csv")["v"].tolist() == [2], (
            "stale country split was reused after the source changed"
        )

    def test_split_is_reused_when_unchanged(self, tmp_path):
        from src.dataset_builder import _ensure_split_by_iso3

        path = tmp_path / "impact_data.csv"
        pd.DataFrame({"iso3": ["ATG"], "v": [1]}).to_csv(path, index=False)
        out = _ensure_split_by_iso3(path)
        marker_mtime = (out / "_complete").stat().st_mtime

        out = _ensure_split_by_iso3(path)
        assert (out / "_complete").stat().st_mtime == marker_mtime
