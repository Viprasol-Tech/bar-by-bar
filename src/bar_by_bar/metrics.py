"""Performance metrics computed from an equity curve and the realized trades.

All metrics are pure functions of plain numbers so they are trivial to test
against hand-computed values. :func:`compute_metrics` bundles them into a single
frozen :class:`Metrics` model.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


def _returns(equity: Sequence[float]) -> list[float]:
    """Period-over-period simple returns of an equity curve."""
    out: list[float] = []
    for prev, cur in itertools.pairwise(equity):
        if prev == 0:
            out.append(0.0)
        else:
            out.append((cur - prev) / prev)
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def total_return(equity: Sequence[float]) -> float:
    """Total return of the equity curve, e.g. 0.25 for +25%."""
    if len(equity) < 2 or equity[0] == 0:
        return 0.0
    return (equity[-1] - equity[0]) / equity[0]


def sharpe(equity: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (risk-free rate assumed zero)."""
    rets = _returns(equity)
    sd = _std(rets)
    if sd == 0:
        return 0.0
    return (_mean(rets) / sd) * math.sqrt(periods_per_year)


def sortino(equity: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio (downside-deviation denominator)."""
    rets = _returns(equity)
    if not rets:
        return 0.0
    downside = [min(0.0, r) for r in rets]
    dd = math.sqrt(sum(d**2 for d in downside) / len(downside))
    if dd == 0:
        return 0.0
    return (_mean(rets) / dd) * math.sqrt(periods_per_year)


def max_drawdown(equity: Sequence[float]) -> float:
    """The worst peak-to-trough decline, as a non-negative fraction."""
    peak = -math.inf
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            drop = (peak - value) / peak
            worst = max(worst, drop)
    return worst


def win_rate(trade_pnls: Sequence[float]) -> float:
    """Fraction of closed trades with strictly positive PnL."""
    if not trade_pnls:
        return 0.0
    wins = sum(1 for p in trade_pnls if p > 0)
    return wins / len(trade_pnls)


def exposure(position_flags: Sequence[bool]) -> float:
    """Fraction of bars during which a position was held."""
    if not position_flags:
        return 0.0
    return sum(1 for flag in position_flags if flag) / len(position_flags)


class Metrics(BaseModel):
    """Bundle of performance statistics for a backtest run."""

    model_config = ConfigDict(frozen=True)

    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    exposure: float
    trades: int
    bars: int


def compute_metrics(
    equity: Sequence[float],
    trade_pnls: Sequence[float],
    position_flags: Sequence[bool],
    periods_per_year: int = 252,
) -> Metrics:
    """Compute all metrics from an equity curve, trade PnLs and position flags."""
    return Metrics(
        total_return=total_return(equity),
        sharpe=sharpe(equity, periods_per_year),
        sortino=sortino(equity, periods_per_year),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate(trade_pnls),
        exposure=exposure(position_flags),
        trades=len(trade_pnls),
        bars=len(equity),
    )
