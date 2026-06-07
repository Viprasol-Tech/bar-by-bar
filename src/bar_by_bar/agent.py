"""Agents and their decisions.

An *agent* is anything callable that maps a :class:`~bar_by_bar.guard.MarketView`
to a :class:`Decision`. That deliberately covers a plain function, a class with
``__call__``, or a thin adapter around a CrewAI / LangGraph / AutoGen crew or an
LLM call. The harness never inspects what is inside -- it only feeds the agent a
point-in-time view and reads back a decision, so any framework plugs in.

Two deterministic example agents are provided so the library is useful (and
testable) fully offline, with no LLM or network:

- :func:`sma_cross_agent` -- classic fast/slow simple-moving-average crossover.
- :func:`momentum_agent`   -- buys positive look-back momentum, sells negative.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from bar_by_bar.guard import MarketView


class Action(str, Enum):
    """What an agent wants to do on the current bar."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Decision(BaseModel):
    """An agent's instruction for the current bar.

    ``size`` is a fraction in ``[0, 1]`` of the relevant side:

    - for :attr:`Action.BUY` it is the fraction of *available cash* to deploy;
    - for :attr:`Action.SELL` it is the fraction of the *current position* to
      close;
    - for :attr:`Action.HOLD` it is ignored.
    """

    model_config = ConfigDict(frozen=True)

    action: Action
    size: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = ""

    @classmethod
    def buy(cls, size: float = 1.0, reason: str = "") -> Decision:
        return cls(action=Action.BUY, size=size, reason=reason)

    @classmethod
    def sell(cls, size: float = 1.0, reason: str = "") -> Decision:
        return cls(action=Action.SELL, size=size, reason=reason)

    @classmethod
    def hold(cls, reason: str = "") -> Decision:
        return cls(action=Action.HOLD, size=0.0, reason=reason)


@runtime_checkable
class Agent(Protocol):
    """The agent interface: a callable ``(MarketView) -> Decision``."""

    def __call__(self, view: MarketView) -> Decision: ...


def _sma(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def sma_cross_agent(fast: int = 5, slow: int = 20) -> Agent:
    """Build a simple-moving-average crossover agent.

    Goes (fully) long when the fast SMA is above the slow SMA, and flat
    otherwise. Holds until enough bars exist to compute the slow SMA.
    """

    if fast < 1 or slow < 1:
        raise ValueError("SMA windows must be positive")
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")

    def agent(view: MarketView) -> Decision:
        if not view.has(slow):
            return Decision.hold(reason=f"warming up ({len(view)}/{slow} bars)")
        fast_sma = _sma(view.closes(fast))
        slow_sma = _sma(view.closes(slow))
        if fast_sma > slow_sma:
            return Decision.buy(size=1.0, reason=f"fast {fast_sma:.2f} > slow {slow_sma:.2f}")
        return Decision.sell(size=1.0, reason=f"fast {fast_sma:.2f} <= slow {slow_sma:.2f}")

    return agent


def momentum_agent(lookback: int = 10, threshold: float = 0.0) -> Agent:
    """Build a momentum agent.

    Compares the current close against the close ``lookback`` bars ago. Buys
    when the return exceeds ``threshold``, sells when it falls below
    ``-threshold``, and holds in the neutral band between.
    """

    if lookback < 1:
        raise ValueError("lookback must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    def agent(view: MarketView) -> Decision:
        if not view.has(lookback + 1):
            return Decision.hold(reason=f"warming up ({len(view)}/{lookback + 1} bars)")
        window = view.closes(lookback + 1)
        past, now = window[0], window[-1]
        ret = (now - past) / past
        if ret > threshold:
            return Decision.buy(size=1.0, reason=f"momentum {ret:+.2%}")
        if ret < -threshold:
            return Decision.sell(size=1.0, reason=f"momentum {ret:+.2%}")
        return Decision.hold(reason=f"flat momentum {ret:+.2%}")

    return agent
