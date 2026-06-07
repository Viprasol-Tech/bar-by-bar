"""Command-line interface for bar-by-bar.

Commands
--------
- ``run``            backtest a built-in example agent over synthetic bars.
- ``lookahead-demo`` show the look-ahead guard raising when an agent peeks ahead.
"""

from __future__ import annotations

import json
from enum import Enum

import typer
from rich.console import Console

from bar_by_bar.agent import Decision, momentum_agent, sma_cross_agent
from bar_by_bar.engine import Harness
from bar_by_bar.guard import LookaheadError, MarketView
from bar_by_bar.report import metrics_table, result_to_dict
from bar_by_bar.synthetic import synthetic_series

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Framework-agnostic agentic backtesting harness with a hard look-ahead guard.",
)
console = Console()


class AgentName(str, Enum):
    sma = "sma"
    momentum = "momentum"


@app.command()
def run(
    agent: AgentName = typer.Option(AgentName.sma, help="Which example agent to run."),
    bars: int = typer.Option(250, min=2, help="Number of synthetic bars."),
    seed: int = typer.Option(7, help="Seed for the synthetic series."),
    cash: float = typer.Option(10_000.0, help="Starting cash."),
    fee: float = typer.Option(0.0005, help="Proportional fee rate."),
    slippage: float = typer.Option(0.0005, help="Proportional slippage."),
    as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
) -> None:
    """Backtest an example agent over deterministic synthetic bars."""
    series = synthetic_series(n=bars, seed=seed)
    strategy = sma_cross_agent() if agent is AgentName.sma else momentum_agent()
    harness = Harness(starting_cash=cash, fee_rate=fee, slippage=slippage)
    result = harness.run(series, strategy)

    if as_json:
        console.print_json(json.dumps(result_to_dict(result)))
        return

    console.print(f"[bold]Agent:[/bold] {agent.value}   [bold]Bars:[/bold] {len(series)}")
    console.print(metrics_table(result, title=f"bar-by-bar :: {agent.value} agent"))


@app.command("lookahead-demo")
def lookahead_demo() -> None:
    """Demonstrate the look-ahead guard blocking access to future bars."""
    series = synthetic_series(n=20, seed=1)

    console.print("[bold]A well-behaved agent only reads the current bar:[/bold]")
    view = series.view_at(5)
    console.print(
        f"  view at t=5 -> sees {len(view)} bars, current close = {view.current.close:.2f}"
    )

    console.print("\n[bold]A cheating agent tries to read one bar into the future:[/bold]")

    def cheating_agent(v: MarketView) -> Decision:
        # t + 1 is the very next (future) bar -- this must be blocked.
        _future = v[v.t + 1]
        return Decision.buy()

    try:
        cheating_agent(view)
    except LookaheadError as exc:
        console.print(f"  [bold red]LookaheadError raised:[/bold red] {exc}")

    console.print("\n[bold]Slicing past the boundary is blocked too:[/bold]")
    try:
        _ = view[: view.t + 5]
    except LookaheadError as exc:
        console.print(f"  [bold red]LookaheadError raised:[/bold red] {exc}")

    console.print(
        "\n[green]The guard makes look-ahead bias a hard error, not a silent bug.[/green]"
    )


if __name__ == "__main__":
    app()
