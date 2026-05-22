"""Technical indicators.

The v1 notebook implemented RSI inline inside ``get_rsi`` with a partial
Wilder smoothing that mixed the simple moving average with a single
Wilder-style update. Here we provide both the classic SMA-based RSI and the
Wilder recursive smoothing, well-tested and importable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

RSIMethod = Literal["wilder", "sma"]


def rsi(close: pd.Series, window: int = 14, method: RSIMethod = "wilder") -> pd.Series:
    """Relative Strength Index.

    ``method='wilder'`` uses the standard Wilder exponential smoothing
    (alpha = 1/window). ``method='sma'`` uses a simple moving average of the
    gains/losses, which matches some textbook definitions.
    """
    if window < 2:
        raise ValueError("`window` must be >= 2")

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    if method == "wilder":
        avg_gain = gains.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        avg_loss = losses.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    else:
        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + rs))
    # Resolve the conventional corner cases in priority order:
    #   - flat (no gains AND no losses)  -> undefined, keep as NaN
    #   - all gains, no losses           -> 100
    #   - all losses, no gains           -> 0
    flat_mask = (avg_gain == 0) & (avg_loss == 0)
    rsi_values = rsi_values.where(avg_loss != 0, 100.0)
    rsi_values = rsi_values.where(avg_gain != 0, 0.0)
    rsi_values = rsi_values.mask(flat_mask, np.nan)
    return rsi_values.rename("RSI")


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()
