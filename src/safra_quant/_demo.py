"""Synthetic data fallback for environments without Yahoo Finance access.

Activated via the ``SAFRA_QUANT_DEMO=1`` environment variable, or via the
:func:`get_prices_with_fallback` helper which tries real yfinance and falls
back automatically. Used by the notebook so it can render outputs even in
sandboxed CI environments.

This data is **synthetic** — never present it as real market data.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from safra_quant.data import get_prices


def is_demo_mode() -> bool:
    return os.getenv("SAFRA_QUANT_DEMO", "").lower() in ("1", "true", "yes")


def _seed_for(ticker: str) -> int:
    return abs(hash(ticker)) % (2**32 - 1)


def synthetic_prices(
    tickers: Iterable[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate a deterministic geometric-Brownian-motion price panel."""
    idx = pd.bdate_range(start, end, freq="B")
    cols: dict[str, np.ndarray] = {}
    for ticker in tickers:
        rng = np.random.default_rng(_seed_for(ticker))
        mu = rng.uniform(0.0001, 0.0006)
        sigma = rng.uniform(0.012, 0.025)
        p0 = rng.uniform(10.0, 80.0)
        returns = rng.normal(mu, sigma, size=len(idx))
        cols[ticker] = p0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(cols, index=idx)


def get_prices_with_fallback(
    tickers: str | Iterable[str],
    start: str,
    end: str,
) -> tuple[pd.DataFrame, bool]:
    """Return (prices, used_demo). Tries real yfinance unless demo mode is on."""
    tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
    if is_demo_mode():
        return synthetic_prices(tickers_list, start, end), True
    try:
        return get_prices(tickers_list, start=start, end=end), False
    except Exception:
        return synthetic_prices(tickers_list, start, end), True
