"""Market data primitives: the OHLCV :class:`Bar` and the :class:`PointInTimeSeries`.

The :class:`PointInTimeSeries` is the heart of the harness. Iterating it yields a
frozen, read-only :class:`MarketView` for each bar index ``t``. The view exposes
*only* the bars up to and including ``t`` -- any attempt to look forward raises a
:class:`LookaheadError` (see :mod:`bar_by_bar.guard`). This makes look-ahead bias
a hard error rather than a silent, results-corrupting mistake.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from bar_by_bar.guard import MarketView


class Bar(BaseModel):
    """A single OHLCV candle.

    The bar is immutable (frozen) so a view handed to an agent can never be
    mutated underneath the harness.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: int
    """Unix epoch seconds (or any monotonically increasing integer index)."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @field_validator("open", "high", "low", "close")
    @classmethod
    def _positive_price(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("prices must be finite")
        if value <= 0:
            raise ValueError("prices must be strictly positive")
        return value

    @field_validator("volume")
    @classmethod
    def _non_negative_volume(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("volume must be a finite, non-negative number")
        return value

    @model_validator(mode="after")
    def _check_ohlc(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must lie within [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must lie within [low, high]")
        return self

    @property
    def typical_price(self) -> float:
        """The (high + low + close) / 3 typical price."""
        return (self.high + self.low + self.close) / 3.0

    @property
    def is_up(self) -> bool:
        """Whether the bar closed above its open."""
        return self.close > self.open


class PointInTimeSeries:
    """An ordered series of bars that can be replayed point-in-time.

    Construction validates that timestamps are strictly increasing so that the
    notion of "the future" is well defined.

    Use :meth:`walk` (or iterate the object directly) to step bar by bar; each
    step yields a frozen :class:`MarketView` exposing only bars ``[0 .. t]``.
    """

    def __init__(self, bars: Sequence[Bar]) -> None:
        if len(bars) == 0:
            raise ValueError("PointInTimeSeries requires at least one bar")
        ordered = tuple(bars)
        for prev, cur in itertools.pairwise(ordered):
            if cur.timestamp <= prev.timestamp:
                raise ValueError("bar timestamps must be strictly increasing")
        self._bars: tuple[Bar, ...] = ordered

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def bars(self) -> tuple[Bar, ...]:
        """The full, underlying bar tuple. Used by the engine, not by agents."""
        return self._bars

    def view_at(self, t: int) -> MarketView:
        """Return a frozen view exposing bars ``[0 .. t]`` (inclusive)."""
        if t < 0 or t >= len(self._bars):
            raise IndexError(f"t={t} out of range for series of length {len(self._bars)}")
        return MarketView(self._bars, t)

    def walk(self) -> Iterator[MarketView]:
        """Yield a :class:`MarketView` for each bar index ``t`` in order."""
        for t in range(len(self._bars)):
            yield MarketView(self._bars, t)

    def __iter__(self) -> Iterator[MarketView]:
        return self.walk()
