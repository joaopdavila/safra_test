from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from safra_quant.data import get_prices


def test_cache_miss_calls_downloader_and_writes_parquet(tmp_cache_dir, fake_yf_downloader):
    spy = MagicMock(side_effect=fake_yf_downloader)
    out = get_prices(
        ["AAAA3.SA"], "2018-01-02", "2018-06-30", cache_dir=tmp_cache_dir, downloader=spy
    )
    assert spy.call_count == 1
    assert not out.isna().values.any()
    assert list(tmp_cache_dir.glob("*.parquet"))


def test_cache_hit_skips_downloader(tmp_cache_dir, fake_yf_downloader):
    spy = MagicMock(side_effect=fake_yf_downloader)
    get_prices(["AAAA3.SA"], "2018-01-02", "2018-06-30", cache_dir=tmp_cache_dir, downloader=spy)
    get_prices(["AAAA3.SA"], "2018-01-02", "2018-06-30", cache_dir=tmp_cache_dir, downloader=spy)
    assert spy.call_count == 1


def test_returns_chronologically_sorted_and_no_nan(tmp_cache_dir, fake_yf_downloader):
    out = get_prices(
        ["AAAA3.SA", "BBBB4.SA"],
        "2018-01-02",
        "2019-06-30",
        cache_dir=tmp_cache_dir,
        downloader=fake_yf_downloader,
    )
    assert out.index.is_monotonic_increasing
    assert not out.isna().values.any()
    assert set(out.columns) == {"AAAA3.SA", "BBBB4.SA"}


def test_empty_tickers_raises(tmp_cache_dir):
    import pytest

    with pytest.raises(ValueError):
        get_prices([], "2018-01-02", "2019-01-01", cache_dir=tmp_cache_dir)


def test_fills_internal_nan_via_interpolation(tmp_cache_dir):
    """Inject a downloader that returns NaNs and confirm interpolation fills them."""
    idx = pd.bdate_range("2020-01-02", periods=10, freq="B")
    base = pd.Series(
        [10.0, 10.1, None, None, 10.4, 10.5, None, 10.7, 10.8, 10.9], index=idx, name="X.SA"
    )
    payload = pd.concat({"Adj Close": base.to_frame()}, axis=1)
    payload.columns = pd.MultiIndex.from_tuples(list(payload.columns))

    def downloader(*_a, **_k):
        return payload

    out = get_prices(
        ["X.SA"], "2020-01-02", "2020-01-20", cache_dir=tmp_cache_dir, downloader=downloader
    )
    assert not out.isna().values.any()
    # Linear interpolation between 10.1 and 10.4 over two missing days.
    assert abs(out["X.SA"].iloc[2] - 10.2) < 1e-9
    assert abs(out["X.SA"].iloc[3] - 10.3) < 1e-9
