"""Tests for the look-ahead guard and MarketView."""

from __future__ import annotations

import pytest

from bar_by_bar.guard import LookaheadError, MarketView

from .conftest import series_from_closes


def test_lookahead_error_is_index_error() -> None:
    assert issubclass(LookaheadError, IndexError)


def test_view_exposes_exactly_bars_up_to_t() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0, 14.0])
    view = s.view_at(2)
    assert len(view) == 3
    assert [b.close for b in view] == [10.0, 11.0, 12.0]


def test_current_is_bar_t() -> None:
    s = series_from_closes([10.0, 11.0, 12.0])
    view = s.view_at(1)
    assert view.current.close == 11.0


def test_indexing_future_bar_raises_lookahead() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(1)
    with pytest.raises(LookaheadError):
        _ = view[2]


def test_indexing_far_future_raises_lookahead() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(0)
    with pytest.raises(LookaheadError):
        _ = view[view.t + 1]


def test_indexing_current_and_past_is_allowed() -> None:
    s = series_from_closes([10.0, 11.0, 12.0])
    view = s.view_at(2)
    assert view[0].close == 10.0
    assert view[1].close == 11.0
    assert view[2].close == 12.0


def test_negative_index_resolves_within_window() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(2)
    assert view[-1].close == 12.0
    assert view[-3].close == 10.0


def test_negative_index_before_start_raises() -> None:
    s = series_from_closes([10.0, 11.0, 12.0])
    view = s.view_at(1)
    with pytest.raises(IndexError):
        _ = view[-5]


def test_slice_within_window_ok() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0, 14.0])
    view = s.view_at(3)
    closes = [b.close for b in view[0:3]]
    assert closes == [10.0, 11.0, 12.0]


def test_slice_past_boundary_raises_lookahead() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0, 14.0])
    view = s.view_at(2)
    with pytest.raises(LookaheadError):
        _ = view[0 : view.t + 3]


def test_open_ended_slice_clamps_to_boundary() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0, 14.0])
    view = s.view_at(2)
    closes = [b.close for b in view[:]]
    assert closes == [10.0, 11.0, 12.0]


def test_iteration_stops_at_t() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(1)
    assert [b.close for b in view] == [10.0, 11.0]


def test_view_is_read_only_setattr() -> None:
    s = series_from_closes([10.0, 11.0])
    view = s.view_at(1)
    with pytest.raises(AttributeError):
        view.t = 0  # type: ignore[misc]


def test_view_is_read_only_delattr() -> None:
    s = series_from_closes([10.0, 11.0])
    view = s.view_at(1)
    with pytest.raises(AttributeError):
        del view.t  # type: ignore[misc]


def test_last_returns_clamped_window() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(3)
    assert [b.close for b in view.last(2)] == [12.0, 13.0]
    # asking for more than exist clamps rather than peeking
    assert len(view.last(99)) == 4


def test_last_requires_positive_n() -> None:
    s = series_from_closes([10.0, 11.0])
    view = s.view_at(1)
    with pytest.raises(ValueError):
        view.last(0)


def test_closes_helper() -> None:
    s = series_from_closes([10.0, 11.0, 12.0, 13.0])
    view = s.view_at(3)
    assert view.closes() == (10.0, 11.0, 12.0, 13.0)
    assert view.closes(2) == (12.0, 13.0)


def test_has_helper() -> None:
    s = series_from_closes([10.0, 11.0, 12.0])
    view = s.view_at(1)
    assert view.has(2) is True
    assert view.has(3) is False


def test_construct_view_with_bad_t_raises() -> None:
    s = series_from_closes([10.0, 11.0])
    with pytest.raises(IndexError):
        MarketView(s.bars, 5)
