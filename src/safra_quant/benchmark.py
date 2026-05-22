"""IBOVESPA loader and helpers for relative performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from safra_quant.data import get_prices
from safra_quant.universe import IBOVESPA_TICKER


def get_ibovespa(start: str, end: str, *, cache_dir: Path | None = None) -> pd.Series:
    df = get_prices([IBOVESPA_TICKER], start=start, end=end, cache_dir=cache_dir)
    series = pd.Series(df[IBOVESPA_TICKER].to_numpy(), index=df.index, name="IBOV")
    return series


def rebase_to(series: pd.Series, base: float = 100.0) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return clean
    rebased: pd.Series = (series / clean.iloc[0]) * base
    return rebased
