"""Price loader with local Parquet cache.

The v1 notebook called ``yf.download`` once per question, hitting Yahoo five
times and recomputing the same series. Here we centralize fetching: results
are cached as Parquet under ``data/`` and reused on subsequent calls.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_FIELD = "Adj Close"


class Downloader(Protocol):
    def __call__(self, tickers: Any, **kwargs: Any) -> pd.DataFrame: ...


def _cache_key(tickers: Iterable[str], start: str, end: str, field: str) -> str:
    raw = "|".join(sorted(tickers)) + f"|{start}|{end}|{field}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _normalize(df: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    """Return a tickers-as-columns DataFrame indexed by tz-naive dates."""
    out: pd.DataFrame
    if isinstance(df.columns, pd.MultiIndex):
        if field in df.columns.get_level_values(0):
            slice_: pd.DataFrame | pd.Series = df[field]
        else:
            slice_ = df.xs(field, axis=1, level=-1)
        out = slice_.copy() if isinstance(slice_, pd.DataFrame) else slice_.to_frame()
    elif field in df.columns:
        out = df[[field]].copy()
        out.columns = pd.Index(tickers[:1])
    else:
        out = df.copy()

    if len(tickers) == 1 and len(out.columns) == 1:
        out.columns = pd.Index(tickers)

    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    return out.sort_index()


def get_prices(
    tickers: str | Iterable[str],
    start: str,
    end: str,
    *,
    field: str = DEFAULT_FIELD,
    fill_neighbors: bool = True,
    cache_dir: Path | None = None,
    downloader: Downloader = yf.download,
) -> pd.DataFrame:
    """Fetch close prices and fill missing values with neighbor averages.

    ``downloader`` is injected so tests can monkeypatch without monkey-patching
    the global ``yfinance`` module.
    """
    tickers_list = [tickers] if isinstance(tickers, str) else list(tickers)
    if not tickers_list:
        raise ValueError("`tickers` cannot be empty")

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(tickers_list, start, end, field)}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = downloader(
        tickers_list if len(tickers_list) > 1 else tickers_list[0],
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {tickers_list} in {start}..{end}")

    prices = _normalize(raw, tickers_list, field)

    if fill_neighbors:
        # Linear interpolation across the *time* axis approximates "fill with
        # neighbor mean". `limit_direction='both'` also fills NaNs at the edges
        # (the v1 notebook used 'backward' and left forward-edge NaNs untouched).
        prices = prices.interpolate(method="linear", limit_direction="both", axis=0)

    prices.to_parquet(cache_file)
    return prices
