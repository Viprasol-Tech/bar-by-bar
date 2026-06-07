"""Tests for reporting, synthetic data, and the CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from bar_by_bar.agent import sma_cross_agent
from bar_by_bar.cli import app
from bar_by_bar.engine import Harness
from bar_by_bar.report import metrics_table, result_to_dict
from bar_by_bar.synthetic import synthetic_series

runner = CliRunner()


def test_synthetic_series_is_deterministic() -> None:
    a = synthetic_series(n=50, seed=42)
    b = synthetic_series(n=50, seed=42)
    assert [bar.close for bar in a.bars] == [bar.close for bar in b.bars]


def test_synthetic_series_length() -> None:
    s = synthetic_series(n=120, seed=3)
    assert len(s) == 120


def test_synthetic_series_requires_two_bars() -> None:
    import pytest

    with pytest.raises(ValueError):
        synthetic_series(n=1)


def test_result_to_dict_shape() -> None:
    s = synthetic_series(n=60, seed=5)
    result = Harness().run(s, sma_cross_agent())
    d = result_to_dict(result)
    for key in (
        "starting_cash",
        "final_equity",
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "exposure",
        "trades",
        "bars",
    ):
        assert key in d


def test_metrics_table_builds() -> None:
    s = synthetic_series(n=60, seed=5)
    result = Harness().run(s, sma_cross_agent())
    table = metrics_table(result)
    assert table.row_count == 10


def test_cli_run_sma() -> None:
    res = runner.invoke(app, ["run", "--agent", "sma", "--bars", "60"])
    assert res.exit_code == 0
    assert "bar-by-bar" in res.stdout


def test_cli_run_momentum_json() -> None:
    res = runner.invoke(app, ["run", "--agent", "momentum", "--bars", "60", "--json"])
    assert res.exit_code == 0
    assert "total_return" in res.stdout


def test_cli_lookahead_demo_raises_and_reports() -> None:
    res = runner.invoke(app, ["lookahead-demo"])
    assert res.exit_code == 0
    assert "LookaheadError raised" in res.stdout
