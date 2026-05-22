"""Shared pytest fixtures.

All fixtures here are deterministic and offline. We never call yfinance from
the test suite; the few tests that exercise the data loader use a fake
downloader injected via the `downloader=` argument of `get_prices`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Two synthetic tickers, 500 business days starting 2018-01-02."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-02", periods=500, freq="B")
    a = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=len(idx))))
    b = 20 * np.exp(np.cumsum(rng.normal(0.0003, 0.020, size=len(idx))))
    return pd.DataFrame({"AAAA3.SA": a, "BBBB4.SA": b}, index=idx)


@pytest.fixture
def monotonic_up_prices() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=80, freq="B")
    return pd.Series(np.linspace(100, 200, len(idx)), index=idx, name="UP")


@pytest.fixture
def monotonic_down_prices() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=80, freq="B")
    return pd.Series(np.linspace(200, 100, len(idx)), index=idx, name="DOWN")


@pytest.fixture
def fake_yf_downloader(sample_prices: pd.DataFrame):
    """Return a callable with yfinance.download's signature."""

    def _download(
        tickers: str | Iterable[str],
        start: str,
        end: str,
        progress: bool = False,
        auto_adjust: bool = False,
    ) -> pd.DataFrame:
        tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
        sliced = sample_prices.loc[start:end, [t for t in tickers_list if t in sample_prices]]
        # Emulate yfinance multi-ticker shape: columns are (field, ticker).
        out = pd.concat({"Adj Close": sliced, "Close": sliced}, axis=1)
        out.columns = pd.MultiIndex.from_tuples(list(out.columns))
        return out

    return _download


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "data_cache"
    cache.mkdir()
    return cache
