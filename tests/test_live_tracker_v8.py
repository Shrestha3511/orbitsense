import numpy as np

from live_tracker_v8 import propagate_pair_series


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
