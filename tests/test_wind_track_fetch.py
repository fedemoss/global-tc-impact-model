"""A transient IBTrACS read failure must not be recorded as a missing storm.

Six cyclones were lost from the global dataset this way. CLIMADA downloads
IBTrACS.ALL.v04r01.nc on the first track request; the wind stage runs a process
pool, so several workers issued that request at once and the ones that arrived
while the file was still being written failed with "No such file or directory"
or "NetCDF: HDF error". The broad `except Exception` recorded each as a storm
that does not exist, and because the stage skips countries whose output already
exists, re-running never recovered them.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ABSENT = "The following given IDs are not in IBTrACS: 2013018S14138."
TRANSIENT = [
    FileNotFoundError(2, "No such file or directory: 'IBTrACS.ALL.v04r01.nc'"),
    OSError("NetCDF: HDF error"),
    RuntimeError("NetCDF: HDF error"),
]


@pytest.fixture
def fetch(monkeypatch):
    from src.dynamic_features import process_wind_features as pwf

    monkeypatch.setattr(pwf.time, "sleep", lambda *_: None)
    return pwf


class TestAbsentIsFinal:
    def test_absent_storm_is_not_retried(self, fetch):
        calls = []

        def fake(storm_id):
            calls.append(storm_id)
            raise ValueError(ABSENT)

        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(fake)
        t, reason = fetch._fetch_track("2013018S14138")

        assert t is None
        assert "absent from IBTrACS" in reason
        assert len(calls) == 1, "a storm that is not in the archive is final"


class TestTransientIsRetried:
    @pytest.mark.parametrize("exc", TRANSIENT)
    def test_transient_error_is_retried_then_succeeds(self, fetch, exc):
        calls = []

        def fake(storm_id):
            calls.append(storm_id)
            if len(calls) < 3:
                raise exc
            return "TRACKS"

        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(fake)
        t, reason = fetch._fetch_track("2008245N17323", attempts=3, backoff=0)

        assert t == "TRACKS", "the storm exists and must be recovered"
        assert reason is None
        assert len(calls) == 3

    @pytest.mark.parametrize("exc", TRANSIENT)
    def test_persistent_transient_error_is_not_called_absent(self, fetch, exc):
        def fake(storm_id):
            raise exc

        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(fake)
        t, reason = fetch._fetch_track("2008245N17323", attempts=2, backoff=0)

        assert t is None
        assert "unreadable" in reason
        assert "absent from IBTrACS" not in reason, (
            "an unreadable archive must never be reported as a missing storm"
        )


class TestPrefetch:
    def test_prefetch_tolerates_an_unknown_sample_id(self, fetch):
        """The archive is readable; the sample storm just is not in it."""
        def fake(storm_id):
            raise ValueError(ABSENT)

        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(fake)
        assert fetch.ensure_ibtracs_available("2013018S14138") is True

    def test_prefetch_reports_failure_when_archive_unreachable(self, fetch):
        def fake(storm_id):
            raise OSError("NetCDF: HDF error")

        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(fake)
        assert fetch.ensure_ibtracs_available("x", attempts=2, backoff=0) is False

    def test_prefetch_succeeds_normally(self, fetch):
        fetch.TCTracks.from_ibtracs_netcdf = staticmethod(lambda storm_id: "TRACKS")
        assert fetch.ensure_ibtracs_available("2008245N17323") is True
