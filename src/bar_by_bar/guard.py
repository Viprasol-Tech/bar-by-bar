"""The look-ahead guard: :class:`LookaheadError` and the frozen :class:`MarketView`.

A :class:`MarketView` is the *only* window an agent ever sees. It wraps the full
bar tuple together with a frozen "now" index ``t`` and refuses to reveal anything
beyond ``t``. Every access path -- indexing, slicing, ``last``, ``window``,
iteration, even negative indices -- is bounded to ``[0 .. t]``. Reaching past
``t`` raises :class:`LookaheadError`.

The view is genuinely read-only: it stores a tuple of frozen pydantic ``Bar``
models, exposes no setters, and ``__setattr__`` is locked after construction.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bar_by_bar.data import Bar


class LookaheadError(IndexError):
    """Raised when code tries to read a bar at or beyond the future boundary.

    Subclasses :class:`IndexError` so ordinary ``except IndexError`` handlers
    still catch out-of-range access, while callers that care about look-ahead
    specifically can catch :class:`LookaheadError`.
    """


class MarketView:
    """A frozen, read-only window over bars ``[0 .. t]``.

    ``t`` is the index of the *current* bar -- the most recent bar the agent is
    allowed to see. The current bar is included (decisions are made on the close
    of bar ``t``). Indices ``> t`` are the future and are forbidden.
    """

    __slots__ = ("_bars", "_frozen", "_t")

    _bars: tuple[Bar, ...]
    _t: int
    _frozen: bool

    def __init__(self, bars: tuple[Bar, ...], t: int) -> None:
        if t < 0 or t >= len(bars):
            raise IndexError(f"t={t} out of range for {len(bars)} bars")
        object.__setattr__(self, "_bars", bars)
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_frozen", True)

    # -- immutability -----------------------------------------------------
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MarketView is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MarketView is read-only")

    # -- boundary ---------------------------------------------------------
    @property
    def t(self) -> int:
        """The current bar index (the future boundary)."""
        return self._t

    def __len__(self) -> int:
        """The number of visible bars, i.e. ``t + 1``."""
        return self._t + 1

    def _resolve(self, index: int) -> int:
        """Translate a (possibly negative) index into an absolute, guarded one."""
        n = self._t + 1
        abs_index = index + n if index < 0 else index
        if abs_index < 0:
            raise IndexError(f"index {index} before the start of the series")
        if abs_index > self._t:
            raise LookaheadError(
                f"look-ahead blocked: requested bar {abs_index} but only bars "
                f"0..{self._t} are visible at t={self._t}"
            )
        return abs_index

    # -- access -----------------------------------------------------------
    def __getitem__(self, index: int | slice) -> Bar | tuple[Bar, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._t + 1)
            # ``slice.indices`` already clamps to the visible range, but an
            # explicit stop beyond the boundary is a look-ahead attempt.
            if index.stop is not None:
                requested_stop = index.stop
                if requested_stop < 0:
                    requested_stop += self._t + 1
                if requested_stop > self._t + 1:
                    raise LookaheadError(
                        f"look-ahead blocked: slice stop {index.stop} reaches past "
                        f"the visible boundary t={self._t}"
                    )
            return tuple(self._bars[i] for i in range(start, stop, step))
        return self._bars[self._resolve(index)]

    def __iter__(self) -> Iterator[Bar]:
        for i in range(self._t + 1):
            yield self._bars[i]

    @property
    def current(self) -> Bar:
        """The current bar (index ``t``)."""
        return self._bars[self._t]

    def last(self, n: int = 1) -> tuple[Bar, ...]:
        """The last ``n`` visible bars, oldest first. Clamped to what exists."""
        if n <= 0:
            raise ValueError("n must be positive")
        start = max(0, self._t + 1 - n)
        return tuple(self._bars[i] for i in range(start, self._t + 1))

    def closes(self, n: int | None = None) -> tuple[float, ...]:
        """Closing prices of the visible window (optionally the last ``n``)."""
        bars = self.last(n) if n is not None else tuple(self)
        return tuple(b.close for b in bars)

    def has(self, n: int) -> bool:
        """Whether at least ``n`` bars are visible so far."""
        return self._t + 1 >= n
