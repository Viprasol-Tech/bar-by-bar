"""bar-by-bar: a framework-agnostic agentic backtesting harness.

Feed any agent (a plain callable, a CrewAI/LangGraph/AutoGen wrapper, or an LLM
behind a function) point-in-time market bars and let it make a decision *bar by
bar*. A hard look-ahead guard makes it impossible for the agent to peek at any
future bar: every attempt raises :class:`LookaheadError`.

Public API
----------
- :class:`Bar`               - a single OHLCV candle.
- :class:`PointInTimeSeries` - iterates a series, yielding frozen market views.
- :class:`MarketView`        - read-only window over bars up to and including ``t``.
- :class:`LookaheadError`    - raised on any attempt to read a future bar.
- :class:`Decision` / :class:`Action` - what an agent returns each bar.
- :class:`Agent`             - the callable protocol an agent implements.
- :func:`sma_cross_agent` / :func:`momentum_agent` - deterministic examples.
- :class:`Harness` / :class:`PaperBook` - run the backtest, track the book.
- :class:`BacktestResult`    - equity curve + metrics + fills.
"""

from __future__ import annotations

from bar_by_bar.agent import (
    Action,
    Agent,
    Decision,
    momentum_agent,
    sma_cross_agent,
)
from bar_by_bar.data import Bar, PointInTimeSeries
from bar_by_bar.engine import BacktestResult, Fill, Harness, PaperBook
from bar_by_bar.guard import LookaheadError, MarketView
from bar_by_bar.metrics import Metrics, compute_metrics
from bar_by_bar.report import metrics_table, result_to_dict

__all__ = [
    "Action",
    "Agent",
    "BacktestResult",
    "Bar",
    "Decision",
    "Fill",
    "Harness",
    "LookaheadError",
    "MarketView",
    "Metrics",
    "PaperBook",
    "PointInTimeSeries",
    "compute_metrics",
    "metrics_table",
    "momentum_agent",
    "result_to_dict",
    "sma_cross_agent",
]

__version__ = "0.1.0"
