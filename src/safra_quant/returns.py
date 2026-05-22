"""Return calculations.

The v1 notebook computed daily returns on a *reversed* DataFrame, which made
`.diff()` subtract the newer row from the older row — every return came out
with the wrong sign. Here we always compute returns on a chronologically
sorted series and let the caller reorder for display.
"""

from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
import pandas as pd

Frequency = Literal["D", "W", "ME", "QE", "YE"]
PandasObj = TypeVar("PandasObj", pd.DataFrame, pd.Series)


def _ensure_sorted(prices: PandasObj) -> PandasObj:
    if isinstance(prices.index, pd.DatetimeIndex) and not prices.index.is_monotonic_increasing:
        return prices.sort_index()
    return prices


def log_returns(prices: PandasObj) -> PandasObj:
    """log(P_t / P_{t-1}). Always sorts chronologically before differencing."""
    sorted_prices = _ensure_sorted(prices)
    log_prices = np.log(sorted_prices)
    if isinstance(sorted_prices, pd.DataFrame):
        return pd.DataFrame(
            log_prices, index=sorted_prices.index, columns=sorted_prices.columns
        ).diff()
    return pd.Series(log_prices, index=sorted_prices.index, name=sorted_prices.name).diff()


def pct_returns(prices: PandasObj) -> PandasObj:
    """(P_t - P_{t-1}) / P_{t-1}. Always sorts chronologically first."""
    sorted_prices = _ensure_sorted(prices)
    return sorted_prices.pct_change()


def resample_returns(
    prices: PandasObj,
    freq: Frequency,
    *,
    kind: Literal["log", "pct"] = "log",
) -> PandasObj:
    """Aggregate daily prices to ``freq`` and compute period returns.

    The v1 notebook re-downloaded monthly/quarterly bars from Yahoo and then
    interpolated NaN-filled retorno columns, which propagated the same value
    across many months and distorted the statistics. The correct way is to
    resample the daily series and compute returns on the resampled prices.
    """
    sorted_prices = _ensure_sorted(prices)
    period_prices = sorted_prices.resample(freq).last()
    if kind == "log":
        return log_returns(period_prices)
    return pct_returns(period_prices)
