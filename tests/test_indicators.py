from __future__ import annotations

import numpy as np
import pandas as pd

from safra_quant.indicators import rsi, sma


def test_rsi_monotonic_up_saturates_at_100(monotonic_up_prices: pd.Series) -> None:
    r = rsi(monotonic_up_prices, window=14)
    # After warm-up, all gains and no losses → RSI = 100.
    assert r.dropna().iloc[-1] == 100.0


def test_rsi_monotonic_down_saturates_at_zero(monotonic_down_prices: pd.Series) -> None:
    r = rsi(monotonic_down_prices, window=14)
    assert r.dropna().iloc[-1] == 0.0


def test_rsi_warmup_is_nan() -> None:
    """Wilder RSI needs `window` non-NaN deltas; first delta is itself NaN."""
    idx = pd.bdate_range("2020-01-01", periods=30, freq="B")
    series = pd.Series(np.linspace(100, 110, 30), index=idx)
    r = rsi(series, window=14)
    assert r.iloc[:14].isna().all()
    assert not np.isnan(r.iloc[14])


def test_rsi_bounded() -> None:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=250, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 250))), index=idx)
    r = rsi(prices, window=14).dropna()
    assert (r >= 0).all()
    assert (r <= 100).all()


def test_rsi_window_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        rsi(pd.Series([1.0, 2.0, 3.0]), window=1)


def test_wilder_differs_from_sma() -> None:
    """The two smoothing methods produce different values past warm-up."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", periods=200, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 200))), index=idx)
    r_wilder = rsi(prices, window=21, method="wilder").dropna()
    r_sma = rsi(prices, window=21, method="sma").dropna()
    # They share the same support after warm-up but should differ in values.
    common = r_wilder.index.intersection(r_sma.index)
    diffs = (r_wilder.loc[common] - r_sma.loc[common]).abs()
    assert diffs.mean() > 0.1


def test_sma_basic() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, window=3).dropna()
    assert list(out) == [2.0, 3.0, 4.0]
