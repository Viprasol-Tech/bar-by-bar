"""The backtest engine: a :class:`PaperBook`, the :class:`Harness`, and results.

The :class:`Harness` walks a :class:`~bar_by_bar.data.PointInTimeSeries` one bar
at a time. On each step it:

1. builds a frozen :class:`~bar_by_bar.guard.MarketView` for the current index,
2. asks the agent for a :class:`~bar_by_bar.agent.Decision`,
3. applies that decision to a :class:`PaperBook` (cash, position, fees, slippage),
4. marks the book to the current close and records one equity-curve point.

Because the agent only ever receives a point-in-time view, look-ahead is
structurally impossible -- not merely discouraged.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bar_by_bar.agent import Action, Agent, Decision
from bar_by_bar.data import Bar, PointInTimeSeries
from bar_by_bar.guard import MarketView
from bar_by_bar.metrics import Metrics, compute_metrics


class Fill(BaseModel):
    """A single executed trade."""

    model_config = ConfigDict(frozen=True)

    bar_index: int
    timestamp: int
    action: Action
    quantity: float
    price: float
    fee: float
    cash_after: float
    position_after: float


class PaperBook:
    """A minimal long-only paper-trading book.

    Tracks ``cash`` and a single-asset ``position`` (units held). Buys spend
    cash, sells return cash. A proportional ``fee_rate`` is charged on notional,
    and ``slippage`` worsens the fill price (paid up on buys, received down on
    sells). The book is long-only: it never goes short and never spends cash it
    does not have.
    """

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0005,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if fee_rate < 0 or slippage < 0:
            raise ValueError("fee_rate and slippage must be non-negative")
        self.starting_cash = starting_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.cash = starting_cash
        self.position = 0.0
        self._cost_basis = 0.0  # total cash paid for the open position (incl. fees)
        self.fills: list[Fill] = []
        self.trade_pnls: list[float] = []

    def equity(self, mark_price: float) -> float:
        """Total account value when the position is marked at ``mark_price``."""
        return self.cash + self.position * mark_price

    def buy(self, bar_index: int, bar: Bar, fraction: float) -> None:
        """Deploy ``fraction`` of available cash at the current bar's close."""
        if fraction <= 0 or self.cash <= 0:
            return
        spend = self.cash * min(fraction, 1.0)
        fill_price = bar.close * (1.0 + self.slippage)
        # quantity such that quantity*fill_price*(1+fee_rate) == spend
        quantity = spend / (fill_price * (1.0 + self.fee_rate))
        if quantity <= 0:
            return
        notional = quantity * fill_price
        fee = notional * self.fee_rate
        self.cash -= notional + fee
        self.position += quantity
        self._cost_basis += notional + fee
        self._record(bar_index, bar, Action.BUY, quantity, fill_price, fee)

    def sell(self, bar_index: int, bar: Bar, fraction: float) -> None:
        """Close ``fraction`` of the open position at the current bar's close."""
        if fraction <= 0 or self.position <= 0:
            return
        quantity = self.position * min(fraction, 1.0)
        fill_price = bar.close * (1.0 - self.slippage)
        notional = quantity * fill_price
        fee = notional * self.fee_rate
        proceeds = notional - fee
        # realized PnL on the closed slice, relative to its share of cost basis
        closed_fraction = quantity / self.position
        cost_of_slice = self._cost_basis * closed_fraction
        self.trade_pnls.append(proceeds - cost_of_slice)
        self.cash += proceeds
        self.position -= quantity
        self._cost_basis -= cost_of_slice
        if self.position <= 1e-12:
            self.position = 0.0
            self._cost_basis = 0.0
        self._record(bar_index, bar, Action.SELL, quantity, fill_price, fee)

    def apply(self, bar_index: int, bar: Bar, decision: Decision) -> None:
        """Route a decision to :meth:`buy` / :meth:`sell` (HOLD is a no-op)."""
        if decision.action is Action.BUY:
            self.buy(bar_index, bar, decision.size)
        elif decision.action is Action.SELL:
            self.sell(bar_index, bar, decision.size)

    def _record(
        self,
        bar_index: int,
        bar: Bar,
        action: Action,
        quantity: float,
        price: float,
        fee: float,
    ) -> None:
        self.fills.append(
            Fill(
                bar_index=bar_index,
                timestamp=bar.timestamp,
                action=action,
                quantity=quantity,
                price=price,
                fee=fee,
                cash_after=self.cash,
                position_after=self.position,
            )
        )


class BacktestResult(BaseModel):
    """Everything produced by a single backtest run."""

    model_config = ConfigDict(frozen=True)

    equity_curve: list[float]
    decisions: list[Decision]
    fills: list[Fill]
    metrics: Metrics
    starting_cash: float
    final_equity: float = Field(default=0.0)

    @property
    def final_value(self) -> float:
        """The last point on the equity curve."""
        return self.equity_curve[-1] if self.equity_curve else self.starting_cash


class Harness:
    """Runs an agent over a point-in-time series, bar by bar."""

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage: float = 0.0005,
        periods_per_year: int = 252,
    ) -> None:
        self.starting_cash = starting_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.periods_per_year = periods_per_year

    def run(self, series: PointInTimeSeries, agent: Agent) -> BacktestResult:
        """Backtest ``agent`` over ``series`` and return a :class:`BacktestResult`."""
        book = PaperBook(
            starting_cash=self.starting_cash,
            fee_rate=self.fee_rate,
            slippage=self.slippage,
        )
        equity_curve: list[float] = []
        decisions: list[Decision] = []
        position_flags: list[bool] = []

        for view in series.walk():
            decision = self._ask(agent, view)
            decisions.append(decision)
            current_bar = series.bars[view.t]
            book.apply(view.t, current_bar, decision)
            equity_curve.append(book.equity(current_bar.close))
            position_flags.append(book.position > 0)

        metrics = compute_metrics(
            equity_curve,
            book.trade_pnls,
            position_flags,
            periods_per_year=self.periods_per_year,
        )
        return BacktestResult(
            equity_curve=equity_curve,
            decisions=decisions,
            fills=book.fills,
            metrics=metrics,
            starting_cash=self.starting_cash,
            final_equity=equity_curve[-1] if equity_curve else self.starting_cash,
        )

    @staticmethod
    def _ask(agent: Agent, view: MarketView) -> Decision:
        decision = agent(view)
        if not isinstance(decision, Decision):
            raise TypeError(f"agent must return a Decision, got {type(decision).__name__}")
        return decision
