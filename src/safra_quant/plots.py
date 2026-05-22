"""Plot helpers — keep notebooks thin and consistent."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _freedman_diaconis_bins(series: pd.Series) -> int:
    clean = series.dropna().to_numpy()
    if clean.size < 2:
        return 10
    q75, q25 = np.percentile(clean, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 30
    bin_width = 2 * iqr * clean.size ** (-1 / 3)
    if bin_width == 0:
        return 30
    return max(10, int(np.ceil((clean.max() - clean.min()) / bin_width)))


def price_variation_distribution(
    prices: pd.Series,
    returns: pd.Series,
    *,
    title_prefix: str = "",
) -> Figure:
    """Three stacked panels: price, daily returns, return distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    axes[0].plot(prices.index, prices.to_numpy(), color="#1f77b4")
    axes[0].set_title(f"{title_prefix} — Histórico de preço (Adj Close)".strip(" —"))
    axes[0].set_ylabel("Preço (BRL)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(returns.index, returns.to_numpy(), color="#ff7f0e", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title(f"{title_prefix} — Retornos diários (log)".strip(" —"))
    axes[1].set_ylabel("Retorno")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        returns.dropna(), bins=_freedman_diaconis_bins(returns), color="#2ca02c", edgecolor="white"
    )
    axes[2].set_title(f"{title_prefix} — Distribuição de retornos diários".strip(" —"))
    axes[2].set_xlabel("Retorno diário")
    axes[2].set_ylabel("Frequência")
    axes[2].grid(True, alpha=0.3)

    return fig


def equity_curves(curves: dict[str, pd.Series], *, title: str = "Equity curves") -> Figure:
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for label, series in curves.items():
        ax.plot(series.index, series.to_numpy(), label=label, linewidth=1.3)
    ax.set_title(title)
    ax.set_ylabel("Índice (base 100)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return fig
