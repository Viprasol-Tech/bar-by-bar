"""Deterministic synthetic bar generation for demos and tests.

Uses a seeded :class:`random.Random` so output is fully reproducible and fully
offline -- no market data download, no network, no API keys.
"""

from __future__ import annotations

import math
import random

from bar_by_bar.data import Bar, PointInTimeSeries


def synthetic_series(
    n: int = 250,
    start_price: float = 100.0,
    seed: int = 7,
    drift: float = 0.0004,
    volatility: float = 0.012,
    cycle: float = 40.0,
    cycle_amp: float = 0.004,
) -> PointInTimeSeries:
    """Generate ``n`` reproducible OHLCV bars as a :class:`PointInTimeSeries`.

    The path combines a small upward drift, a sine-wave cycle (so trend-following
    agents have something to catch) and seeded Gaussian noise.
    """
    if n < 2:
        raise ValueError("need at least two bars")
    rng = random.Random(seed)
    bars: list[Bar] = []
    price = start_price
    ts = 1_700_000_000  # fixed epoch base for reproducible timestamps
    for i in range(n):
        cyclic = cycle_amp * math.sin(2.0 * math.pi * i / cycle)
        shock = rng.gauss(0.0, volatility)
        ret = drift + cyclic + shock
        new_close = max(0.01, price * (1.0 + ret))
        high = max(price, new_close) * (1.0 + abs(rng.gauss(0.0, volatility / 2)))
        low = min(price, new_close) * (1.0 - abs(rng.gauss(0.0, volatility / 2)))
        volume = round(1000.0 + abs(rng.gauss(0.0, 250.0)), 2)
        bars.append(
            Bar(
                timestamp=ts + i * 86_400,
                open=round(price, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(new_close, 4),
                volume=volume,
            )
        )
        price = new_close
    return PointInTimeSeries(bars)
