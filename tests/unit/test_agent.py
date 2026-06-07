"""Tests for decisions and the example agents."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bar_by_bar.agent import Action, Agent, Decision, momentum_agent, sma_cross_agent

from .conftest import series_from_closes


def test_decision_factories() -> None:
    assert Decision.buy().action is Action.BUY
    assert Decision.sell(0.5).action is Action.SELL
    assert Decision.hold().action is Action.HOLD
    assert Decision.hold().size == 0.0


def test_decision_size_bounds() -> None:
    with pytest.raises(ValidationError):
        Decision(action=Action.BUY, size=1.5)
    with pytest.raises(ValidationError):
        Decision(action=Action.BUY, size=-0.1)


def test_decision_is_frozen() -> None:
    d = Decision.buy()
    with pytest.raises(ValidationError):
        d.size = 0.2  # type: ignore[misc]


def test_agent_protocol_runtime_check() -> None:
    agent = sma_cross_agent()
    assert isinstance(agent, Agent)


def test_sma_cross_validates_windows() -> None:
    with pytest.raises(ValueError):
        sma_cross_agent(fast=20, slow=5)
    with pytest.raises(ValueError):
        sma_cross_agent(fast=0, slow=5)


def test_sma_cross_holds_during_warmup() -> None:
    agent = sma_cross_agent(fast=2, slow=4)
    s = series_from_closes([10.0, 11.0])
    decision = agent(s.view_at(1))
    assert decision.action is Action.HOLD


def test_sma_cross_buys_in_uptrend() -> None:
    agent = sma_cross_agent(fast=2, slow=4)
    s = series_from_closes([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    decision = agent(s.view_at(5))
    assert decision.action is Action.BUY


def test_sma_cross_sells_in_downtrend() -> None:
    agent = sma_cross_agent(fast=2, slow=4)
    s = series_from_closes([20.0, 19.0, 18.0, 17.0, 16.0, 15.0])
    decision = agent(s.view_at(5))
    assert decision.action is Action.SELL


def test_momentum_validates_params() -> None:
    with pytest.raises(ValueError):
        momentum_agent(lookback=0)
    with pytest.raises(ValueError):
        momentum_agent(threshold=-0.1)


def test_momentum_holds_during_warmup() -> None:
    agent = momentum_agent(lookback=5)
    s = series_from_closes([10.0, 11.0, 12.0])
    assert agent(s.view_at(2)).action is Action.HOLD


def test_momentum_buys_on_positive_return() -> None:
    agent = momentum_agent(lookback=3, threshold=0.0)
    s = series_from_closes([10.0, 10.5, 11.0, 12.0, 13.0])
    assert agent(s.view_at(4)).action is Action.BUY


def test_momentum_sells_on_negative_return() -> None:
    agent = momentum_agent(lookback=3, threshold=0.0)
    s = series_from_closes([13.0, 12.0, 11.0, 10.0, 9.0])
    assert agent(s.view_at(4)).action is Action.SELL


def test_momentum_holds_within_threshold() -> None:
    agent = momentum_agent(lookback=2, threshold=0.5)
    s = series_from_closes([10.0, 10.05, 10.1])
    assert agent(s.view_at(2)).action is Action.HOLD


def test_example_agents_are_deterministic() -> None:
    s = series_from_closes([10.0 + i for i in range(20)])
    a1 = sma_cross_agent()
    a2 = sma_cross_agent()
    for t in range(len(s)):
        view = s.view_at(t)
        assert a1(view).action == a2(view).action
