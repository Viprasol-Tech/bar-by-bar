"""Tests for the PaperBook and the Harness engine."""

from __future__ import annotations

import pytest

from bar_by_bar.agent import Action, Decision
from bar_by_bar.engine import Harness, PaperBook
from bar_by_bar.guard import LookaheadError, MarketView

from .conftest import make_bar, series_from_closes


def test_paperbook_initial_state() -> None:
    book = PaperBook(starting_cash=1000.0)
    assert book.cash == 1000.0
    assert book.position == 0.0
    assert book.equity(50.0) == 1000.0


def test_paperbook_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        PaperBook(starting_cash=0.0)
    with pytest.raises(ValueError):
        PaperBook(fee_rate=-0.1)
    with pytest.raises(ValueError):
        PaperBook(slippage=-0.1)


def test_buy_with_no_fees_or_slippage() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    bar = make_bar(1, 100.0)
    book.buy(0, bar, 1.0)
    assert book.position == pytest.approx(10.0)
    assert book.cash == pytest.approx(0.0)
    assert book.equity(100.0) == pytest.approx(1000.0)


def test_buy_charges_fee_and_slippage() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.01, slippage=0.0)
    bar = make_bar(1, 100.0)
    book.buy(0, bar, 1.0)
    # spend == cash; quantity = 1000 / (100 * 1.01) = 9.9009...
    assert book.position == pytest.approx(1000.0 / (100.0 * 1.01))
    assert book.cash == pytest.approx(0.0, abs=1e-9)
    assert len(book.fills) == 1
    assert book.fills[0].fee > 0


def test_slippage_raises_buy_price() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.02)
    bar = make_bar(1, 100.0)
    book.buy(0, bar, 1.0)
    assert book.fills[0].price == pytest.approx(102.0)


def test_partial_buy_uses_fraction_of_cash() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    book.buy(0, make_bar(1, 100.0), 0.5)
    assert book.cash == pytest.approx(500.0)
    assert book.position == pytest.approx(5.0)


def test_sell_returns_cash_and_records_pnl() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    book.buy(0, make_bar(1, 100.0), 1.0)
    book.sell(1, make_bar(2, 110.0), 1.0)
    assert book.position == 0.0
    assert book.cash == pytest.approx(1100.0)
    assert len(book.trade_pnls) == 1
    assert book.trade_pnls[0] == pytest.approx(100.0)


def test_partial_sell_keeps_remaining_position() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    book.buy(0, make_bar(1, 100.0), 1.0)
    book.sell(1, make_bar(2, 100.0), 0.5)
    assert book.position == pytest.approx(5.0)
    assert book.cash == pytest.approx(500.0)


def test_sell_without_position_is_noop() -> None:
    book = PaperBook(starting_cash=1000.0)
    book.sell(0, make_bar(1, 100.0), 1.0)
    assert book.cash == 1000.0
    assert not book.fills


def test_buy_without_cash_is_noop() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    book.buy(0, make_bar(1, 100.0), 1.0)
    book.buy(1, make_bar(2, 100.0), 1.0)
    assert len(book.fills) == 1


def test_apply_routes_actions() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    bar = make_bar(1, 100.0)
    book.apply(0, bar, Decision.hold())
    assert not book.fills
    book.apply(0, bar, Decision.buy())
    assert book.position > 0


def test_harness_equity_curve_length() -> None:
    s = series_from_closes([100.0 + i for i in range(15)])
    result = Harness().run(s, lambda v: Decision.hold())
    assert len(result.equity_curve) == 15
    assert len(result.decisions) == 15


def test_harness_hold_keeps_cash_flat() -> None:
    s = series_from_closes([100.0, 110.0, 120.0])
    result = Harness(starting_cash=5000.0).run(s, lambda v: Decision.hold())
    assert all(e == pytest.approx(5000.0) for e in result.equity_curve)
    assert result.metrics.trades == 0
    assert result.metrics.exposure == 0.0


def test_harness_known_buy_and_hold_pnl() -> None:
    # buy at first bar, hold; equity tracks price with no fees/slippage
    s = series_from_closes([100.0, 100.0, 200.0])

    def buy_once(v: MarketView) -> Decision:
        return Decision.buy() if v.t == 0 else Decision.hold()

    result = Harness(starting_cash=1000.0, fee_rate=0.0, slippage=0.0).run(s, buy_once)
    assert result.equity_curve[0] == pytest.approx(1000.0)
    assert result.equity_curve[-1] == pytest.approx(2000.0)
    assert result.metrics.total_return == pytest.approx(1.0)


def test_harness_rejects_non_decision_return() -> None:
    s = series_from_closes([100.0, 101.0])
    with pytest.raises(TypeError):
        Harness().run(s, lambda v: "buy")  # type: ignore[arg-type,return-value]


def test_harness_blocks_lookahead_agent() -> None:
    s = series_from_closes([100.0, 101.0, 102.0, 103.0])

    def cheater(v: MarketView) -> Decision:
        # peeking one bar ahead must blow up inside the harness
        _ = v[v.t + 1]
        return Decision.buy()

    with pytest.raises(LookaheadError):
        Harness().run(s, cheater)


def test_result_final_value() -> None:
    s = series_from_closes([100.0, 100.0])
    result = Harness(starting_cash=2000.0).run(s, lambda v: Decision.hold())
    assert result.final_value == pytest.approx(2000.0)


def test_fill_records_action_type() -> None:
    book = PaperBook(starting_cash=1000.0, fee_rate=0.0, slippage=0.0)
    book.buy(0, make_bar(1, 100.0), 1.0)
    assert book.fills[0].action is Action.BUY
    book.sell(1, make_bar(2, 100.0), 1.0)
    assert book.fills[1].action is Action.SELL
