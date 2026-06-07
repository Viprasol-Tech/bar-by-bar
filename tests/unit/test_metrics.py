"""Tests for the metrics module against hand-computed values."""

from __future__ import annotations

import math

import pytest

from bar_by_bar.metrics import (
    compute_metrics,
    exposure,
    max_drawdown,
    sharpe,
    sortino,
    total_return,
    win_rate,
)


def test_total_return_basic() -> None:
    assert total_return([100.0, 150.0]) == pytest.approx(0.5)
    assert total_return([100.0, 80.0]) == pytest.approx(-0.2)


def test_total_return_degenerate() -> None:
    assert total_return([100.0]) == 0.0
    assert total_return([]) == 0.0


def test_max_drawdown_known() -> None:
    # peak 120 then trough 90 -> (120-90)/120 = 0.25
    curve = [100.0, 120.0, 90.0, 110.0]
    assert max_drawdown(curve) == pytest.approx(0.25)


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_win_rate() -> None:
    assert win_rate([10.0, -5.0, 3.0, -1.0]) == pytest.approx(0.5)
    assert win_rate([1.0, 2.0]) == pytest.approx(1.0)
    assert win_rate([]) == 0.0


def test_exposure() -> None:
    assert exposure([True, True, False, False]) == pytest.approx(0.5)
    assert exposure([False, False]) == 0.0
    assert exposure([]) == 0.0


def test_sharpe_zero_when_flat() -> None:
    assert sharpe([100.0, 100.0, 100.0]) == 0.0


def test_sharpe_positive_for_steady_gains() -> None:
    # constant positive returns -> zero std -> sharpe 0 by convention
    assert sharpe([100.0, 110.0, 121.0]) == 0.0


def test_sharpe_finite_for_mixed_returns() -> None:
    value = sharpe([100.0, 110.0, 105.0, 115.0])
    assert math.isfinite(value)
    assert value != 0.0


def test_sortino_zero_when_no_downside() -> None:
    assert sortino([100.0, 110.0, 121.0]) == 0.0


def test_sortino_finite_with_downside() -> None:
    value = sortino([100.0, 90.0, 120.0, 110.0])
    assert math.isfinite(value)


def test_compute_metrics_bundles_everything() -> None:
    equity = [100.0, 120.0, 90.0, 110.0]
    pnls = [10.0, -5.0]
    flags = [True, True, False, True]
    m = compute_metrics(equity, pnls, flags)
    assert m.total_return == pytest.approx(0.1)
    assert m.max_drawdown == pytest.approx(0.25)
    assert m.win_rate == pytest.approx(0.5)
    assert m.exposure == pytest.approx(0.75)
    assert m.trades == 2
    assert m.bars == 4


def test_metrics_model_is_frozen() -> None:
    m = compute_metrics([100.0, 110.0], [1.0], [True])
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        m.trades = 99  # type: ignore[misc]
