"""The SID corrections must be well formed and actually applied."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamic_features.process_emdat import (  # noqa: E402
    SID_CORRECTIONS,
    UNUSABLE_SIDS,
    add_missing_sid,
)

SID_RE = re.compile(r"^\d{4}\d{3}[NS]\d{2}\d{3}$")


class TestCorrectionTable:
    @pytest.mark.parametrize("bad,good", sorted(SID_CORRECTIONS.items()))
    def test_both_ids_are_well_formed(self, bad, good):
        assert SID_RE.match(bad), f"{bad} is not a storm id"
        assert SID_RE.match(good), f"{good} is not a storm id"

    @pytest.mark.parametrize("bad,good", sorted(SID_CORRECTIONS.items()))
    def test_correction_changes_the_id(self, bad, good):
        assert bad != good, "an identity mapping corrects nothing"

    @pytest.mark.parametrize("bad,good", sorted(SID_CORRECTIONS.items()))
    def test_season_is_preserved(self, bad, good):
        """A correction nudges the day or position, never the year."""
        assert bad[:4] == good[:4]

    @pytest.mark.parametrize("bad,good", sorted(SID_CORRECTIONS.items()))
    def test_hemisphere_is_preserved(self, bad, good):
        assert bad[7] == good[7]

    def test_no_chained_corrections(self):
        """A target must not itself need correcting."""
        assert not (set(SID_CORRECTIONS.values()) & set(SID_CORRECTIONS))

    def test_unusable_ids_are_not_also_corrected(self):
        assert not (UNUSABLE_SIDS & set(SID_CORRECTIONS))


class TestApplication:
    def _frame(self, sids):
        return pd.DataFrame({
            "sid": sids,
            "DisNo.": [f"2021-000{i}-AUS" for i in range(len(sids))],
            "Event Name": ["X"] * len(sids),
            "Start Year": [2021] * len(sids),
            "GID_0": ["AUS"] * len(sids),
        })

    def test_broken_ids_are_replaced(self):
        bad = sorted(SID_CORRECTIONS)[0]
        out = add_missing_sid(self._frame([bad, "2015045S12145"]))
        assert bad not in set(out["sid"])
        assert SID_CORRECTIONS[bad] in set(out["sid"])

    def test_unusable_ids_are_dropped(self):
        unusable = sorted(UNUSABLE_SIDS)[0]
        out = add_missing_sid(self._frame([unusable, "2020100N10100"]))
        assert unusable not in set(out["sid"])
        assert "2020100N10100" in set(out["sid"])

    def test_untouched_ids_survive(self):
        out = add_missing_sid(self._frame(["2020100N10100", "2019200S15150"]))
        assert set(out["sid"]) == {"2020100N10100", "2019200S15150"}

    def test_rows_without_a_sid_are_dropped(self):
        out = add_missing_sid(self._frame(["2020100N10100", None]))
        assert len(out) == 1
