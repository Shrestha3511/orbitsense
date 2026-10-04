import numpy as np

import live_tracker_v8 as tracker
from live_tracker_v8 import (
    MAX_RISK_SLOTS,
    build_dynamic_figure_patch,
    camera_for_target,
    no_update,
    propagate_pair_series,
    search_satellite_lock,
)


class FakeSat:
    def __init__(self, offset, error_indices=None):
        self.offset = offset
        self._calls = 0
        self.error_indices = set(error_indices or [])

    def sgp4(self, jd, fr):
        call = self._calls
        self._calls += 1
        if call in self.error_indices:
            return 1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        pos = (jd + self.offset, fr + self.offset, float(self.offset))
        vel = (self.offset, 0.0, 0.0)
        return 0, pos, vel


def test_propagate_pair_series_uses_both_satellites():
    sat_i = FakeSat(offset=10.0)
    sat_j = FakeSat(offset=20.0)
    jdfr = [(1.0, 0.1), (2.0, 0.2)]

    pa, pb, va, vb = propagate_pair_series(sat_i, sat_j, jdfr)

    assert pa.shape == pb.shape == (2, 3)
    assert np.all(pa != pb)
    assert np.any(va != vb)


def test_propagate_pair_series_raises_when_all_samples_invalid():
    sat_i = FakeSat(offset=1.0, error_indices={0, 1})
    sat_j = FakeSat(offset=2.0, error_indices={0, 1})
    jdfr = [(1.0, 0.1), (2.0, 0.2)]

    try:
        propagate_pair_series(sat_i, sat_j, jdfr)
    except ValueError as exc:
        assert "No valid pair propagation" in str(exc)
    else:
        raise AssertionError("Expected ValueError when all SGP4 samples are invalid")


def test_search_satellite_lock_handles_clear_blank_and_no_match(monkeypatch):
    monkeypatch.setattr(tracker, "SATS", [])
    monkeypatch.setattr(tracker, "NAMES", [])
    monkeypatch.setattr(tracker, "NAMES_LOWER", [])

    cleared = search_satellite_lock("clear-focus-btn", "ISS")
    assert cleared == (None, None, "", "")

    blank = search_satellite_lock("search-btn", "   ")
    assert blank[0] is None
    assert blank[1] is None
    assert "Enter a satellite name or NORAD ID." in blank[2]

    no_match = search_satellite_lock("search-btn", "xyz")
    assert no_match[0] is no_update
    assert no_match[1] is no_update
    assert 'No match: "xyz"' == no_match[2]


def test_search_satellite_lock_supports_norad_case_insensitive_and_repeat(monkeypatch):
    class SearchSat:
        def __init__(self, satnum):
            self.satnum = satnum

    monkeypatch.setattr(tracker, "SATS", [SearchSat(25544), SearchSat(43013)])
    monkeypatch.setattr(tracker, "NAMES", ["ISS (ZARYA)", "TEST SAT"])
    monkeypatch.setattr(tracker, "NAMES_LOWER", ["iss (zarya)", "test sat"])
    monkeypatch.setattr(
        tracker,
        "satellite_telemetry",
        lambda idx, jd, fr: {"pos": np.array([idx + 1.0, 2.0, 3.0])},
    )

    norad = search_satellite_lock("search-btn", "25544")
    name_partial = search_satellite_lock("search-btn", "iss")
    repeat = search_satellite_lock("search-input", "ISS")

    assert norad[0] == 0 and norad[1] == [1.0, 2.0, 3.0]
    assert "LOCKED: ISS (ZARYA)" in norad[2]
    assert name_partial[0] == 0
    assert repeat[0] == 0


def test_build_dynamic_patch_updates_dynamic_traces_only():
    current_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    valid = np.array([True, True])
    risk_slots = [dict(x=[], y=[], z=[], color="#f00", text=[], name="") for _ in range(MAX_RISK_SLOTS)]

    patch = build_dynamic_figure_patch(
        current_pos=current_pos,
        valid=valid,
        risk_slots=risk_slots,
        locked_idx=1,
        lock_xyz=([4.0], [5.0], [6.0]),
        lock_text=["lock"],
        orbit_xyz=([1.0], [1.0], [1.0]),
        target_pos=np.array([1000.0, 2000.0, 3000.0]),
        reset_view=False,
    ).to_plotly_json()

    locations = [op["location"] for op in patch["operations"]]
    assert all(loc[0] != "data" or loc[1] >= 3 for loc in locations)
    assert ["layout", "scene", "camera"] in locations


def test_camera_for_target_rejects_invalid_and_returns_center():
    assert camera_for_target([np.nan, 0.0, 0.0]) is None
    cam = camera_for_target([1800.0, -1800.0, 0.0])
    assert cam is not None
    assert "center" in cam and "eye" in cam


def test_build_dynamic_patch_reset_view_restores_global_autorange():
    patch = build_dynamic_figure_patch(
        current_pos=np.array([[1.0, 2.0, 3.0]]),
        valid=np.array([True]),
        risk_slots=[dict(x=[], y=[], z=[], color="#f00", text=[], name="") for _ in range(MAX_RISK_SLOTS)],
        locked_idx=None,
        lock_xyz=([], [], []),
        lock_text=[],
        orbit_xyz=([], [], []),
        target_pos=None,
        reset_view=True,
    ).to_plotly_json()

    locations = [op["location"] for op in patch["operations"]]
    assert ["layout", "scene", "xaxis", "autorange"] in locations
    assert ["layout", "scene", "camera"] in locations
