"""Render the bar-by-bar headline command output to an SVG for the README hero.

This imports the repo's own agent, engine, and report helpers so the rendered
image matches the real ``bar-by-bar run --agent momentum`` output exactly, then
uses rich's ``record=True`` console to export a colored terminal screenshot.

Run with::

    PYTHONPATH=src python docs/make_demo.py
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from bar_by_bar.agent import momentum_agent
from bar_by_bar.engine import Harness
from bar_by_bar.report import metrics_table
from bar_by_bar.synthetic import synthetic_series

OUT = Path(__file__).resolve().parent / "assets" / "demo.svg"


def main() -> None:
    console = Console(record=True, width=100)

    series = synthetic_series(n=250, seed=7)
    harness = Harness(starting_cash=10_000.0, fee_rate=0.0005, slippage=0.0005)
    result = harness.run(series, momentum_agent())

    console.print("[bold green]$[/bold green] python -m bar_by_bar run --agent momentum\n")
    console.print(f"[bold]Agent:[/bold] momentum   [bold]Bars:[/bold] {len(series)}")
    console.print(metrics_table(result, title="bar-by-bar :: momentum agent"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    console.save_svg(str(OUT), title="bar-by-bar demo")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
