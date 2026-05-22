from __future__ import annotations

import numpy as np
import pandas as pd

from safra_quant.returns import log_returns, pct_returns, resample_returns


def test_reversed_input_does_not_flip_sign(sample_prices: pd.DataFrame) -> None:
    """Regression test for the v1 bug.

    In the 2022 notebook the DataFrame was reversed with `[::-1]` *before*
    calling `.diff()`, which subtracted the newer row from the older row and
    flipped every return's sign. `log_returns` must always sort first.
    """
    forward = log_returns(sample_prices)
    reversed_input = sample_prices.iloc[::-1]
    reversed_output = log_returns(reversed_input).sort_index()
    pd.testing.assert_frame_equal(forward, reversed_output, check_freq=False)


def test_log_and_pct_returns_approximately_equal_for_small_moves(
    sample_prices: pd.DataFrame,
) -> None:
    lr = log_returns(sample_prices).dropna()
    pr = pct_returns(sample_prices).dropna()
    small_moves = pr.abs() < 0.01
    # Differences are O(r^2) for small r.
    diff = (lr - pr)[small_moves].abs().max().max()
    assert diff < 1e-3


def test_first_row_is_nan(sample_prices: pd.DataFrame) -> None:
    assert log_returns(sample_prices).iloc[0].isna().all()
    assert pct_returns(sample_prices).iloc[0].isna().all()


def test_pct_returns_matches_pandas_pct_change(sample_prices: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(pct_returns(sample_prices), sample_prices.pct_change())


def test_resample_returns_uses_period_end_prices() -> None:
    """Resampling on prices then computing returns ≠ averaging daily returns."""
    idx = pd.bdate_range("2020-01-01", periods=66, freq="B")  # ~3 months
    prices = pd.Series(np.linspace(100, 130, len(idx)), index=idx, name="X")
    monthly = resample_returns(prices, "ME", kind="pct")
    monthly_clean = monthly.dropna()
    assert len(monthly_clean) >= 2
    # Each monthly return must match the ratio of end-of-month prices, NOT the
    # interpolation-distorted result the v1 notebook produced.
    end_of_month = prices.resample("ME").last()
    expected = end_of_month.pct_change().dropna()
    pd.testing.assert_series_equal(monthly_clean, expected, check_names=False)
