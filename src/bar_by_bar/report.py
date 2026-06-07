"""Reporting helpers: turn a :class:`~bar_by_bar.engine.BacktestResult` into a
``rich`` table for the terminal or a plain dict for programmatic use.
"""

from __future__ import annotations

from typing import Any

from rich.table import Table

from bar_by_bar.engine import BacktestResult


def result_to_dict(result: BacktestResult) -> dict[str, Any]:
    """Flatten a backtest result into a JSON-friendly dictionary."""
    m = result.metrics
    return {
        "starting_cash": round(result.starting_cash, 2),
        "final_equity": round(result.final_value, 2),
        "total_return": round(m.total_return, 6),
        "sharpe": round(m.sharpe, 4),
        "sortino": round(m.sortino, 4),
        "max_drawdown": round(m.max_drawdown, 6),
        "win_rate": round(m.win_rate, 4),
        "exposure": round(m.exposure, 4),
        "trades": m.trades,
        "bars": m.bars,
    }


def metrics_table(result: BacktestResult, title: str = "bar-by-bar backtest") -> Table:
    """Build a ``rich`` table summarizing a backtest result."""
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    m = result.metrics
    table.add_row("Starting cash", f"{result.starting_cash:,.2f}")
    table.add_row("Final equity", f"{result.final_value:,.2f}")
    table.add_row("Total return", f"{m.total_return:+.2%}")
    table.add_row("Sharpe", f"{m.sharpe:.2f}")
    table.add_row("Sortino", f"{m.sortino:.2f}")
    table.add_row("Max drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("Win rate", f"{m.win_rate:.2%}")
    table.add_row("Exposure", f"{m.exposure:.2%}")
    table.add_row("Trades", str(m.trades))
    table.add_row("Bars", str(m.bars))
    return table
