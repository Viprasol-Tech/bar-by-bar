"""Tests for Bar and PointInTimeSeries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bar_by_bar.data import Bar, PointInTimeSeries

from .conftest import make_bar, series_from_closes


def test_bar_valid_construction() -> None:
    bar = Bar(timestamp=1, open=10.0, high=12.0, low=9.0, close=11.0, volume=5.0)
    assert bar.close == 11.0
    assert bar.volume == 5.0


def test_bar_is_frozen() -> None:
    bar = make_bar(1, 10.0)
    with pytest.raises(ValidationError):
        bar.close = 99.0  # type: ignore[misc]


def test_bar_rejects_high_below_low() -> None:
    with pytest.raises(ValidationError):
        Bar(timestamp=1, open=10.0, high=8.0, low=9.0, close=9.0)


def test_bar_rejects_close_outside_range() -> None:
    with pytest.raises(ValidationError):
        Bar(timestamp=1, open=10.0, high=12.0, low=9.0, close=20.0)


def test_bar_rejects_non_positive_price() -> None:
    with pytest.raises(ValidationError):
        Bar(timestamp=1, open=0.0, high=1.0, low=0.0, close=0.5)


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError):
        Bar(timestamp=1, open=10.0, high=10.0, low=10.0, close=10.0, volume=-1.0)


def test_bar_typical_price() -> None:
    bar = Bar(timestamp=1, open=10.0, high=12.0, low=9.0, close=12.0)
    assert bar.typical_price == pytest.approx((12.0 + 9.0 + 12.0) / 3.0)


def test_bar_is_up() -> None:
    assert make_bar(1, 10.0).is_up is False
    assert Bar(timestamp=1, open=9.0, high=11.0, low=9.0, close=11.0).is_up is True


def test_series_length_matches_bars() -> None:
    s = series_from_closes([1.0, 2.0, 3.0])
    assert len(s) == 3
    assert len(s.bars) == 3


def test_series_requires_at_least_one_bar() -> None:
    with pytest.raises(ValueError):
        PointInTimeSeries([])


def test_series_rejects_non_increasing_timestamps() -> None:
    bars = [make_bar(5, 1.0), make_bar(5, 2.0)]
    with pytest.raises(ValueError):
        PointInTimeSeries(bars)


def test_series_rejects_decreasing_timestamps() -> None:
    bars = [make_bar(5, 1.0), make_bar(4, 2.0)]
    with pytest.raises(ValueError):
        PointInTimeSeries(bars)


def test_view_at_returns_correct_boundary() -> None:
    s = series_from_closes([1.0, 2.0, 3.0, 4.0])
    view = s.view_at(2)
    assert view.t == 2
    assert len(view) == 3


def test_view_at_out_of_range_raises() -> None:
    s = series_from_closes([1.0, 2.0])
    with pytest.raises(IndexError):
        s.view_at(5)
    with pytest.raises(IndexError):
        s.view_at(-1)


def test_walk_yields_increasing_views() -> None:
    s = series_from_closes([1.0, 2.0, 3.0])
    ts = [view.t for view in s.walk()]
    assert ts == [0, 1, 2]


def test_iter_is_walk() -> None:
    s = series_from_closes([1.0, 2.0])
    assert [v.t for v in s] == [0, 1]
