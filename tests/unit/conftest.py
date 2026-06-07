"""Shared fixtures and helpers for the unit tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from bar_by_bar.data import Bar, PointInTimeSeries


def make_bar(ts: int, close: float, *, vol: float = 100.0) -> Bar:
    """A flat OHLC bar at ``close`` (open=high=low=close) for easy reasoning."""
    return Bar(timestamp=ts, open=close, high=close, low=close, close=close, volume=vol)


def series_from_closes(closes: Sequence[float], *, start_ts: int = 1000) -> PointInTimeSeries:
    """Build a PointInTimeSeries from a list of close prices."""
    bars = [make_bar(start_ts + i, c) for i, c in enumerate(closes)]
    return PointInTimeSeries(bars)


@pytest.fixture
def rising_series() -> PointInTimeSeries:
    return series_from_closes([100.0 + i for i in range(30)])


@pytest.fixture
def falling_series() -> PointInTimeSeries:
    return series_from_closes([130.0 - i for i in range(30)])
