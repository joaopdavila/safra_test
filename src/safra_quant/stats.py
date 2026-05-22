"""Performance and distributional statistics.

The v1 notebook only printed mean / median / variance / quartiles. This module
adds the risk-adjusted metrics that any asset-manager-style evaluation will
expect: Sharpe, Sortino, Calmar, max drawdown, beta, Jensen's alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class DistributionStats:
    mean: float
    median: float
    variance: float
    q1: float
    q3: float

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "mean": self.mean,
                "median": self.median,
                "variance": self.variance,
                "q1": self.q1,
                "q3": self.q3,
            }
        )


def distribution(series: pd.Series) -> DistributionStats:
    clean = series.dropna()
    return DistributionStats(
        mean=float(clean.mean()),
        median=float(clean.median()),
        variance=float(clean.var()),
        q1=float(clean.quantile(0.25)),
        q3=float(clean.quantile(0.75)),
    )


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    clean = equity.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return float("nan")
    total_return = clean.iloc[-1] / clean.iloc[0]
    years = len(clean) / periods_per_year
    return float(total_return ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.dropna().std() * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    clean = returns.dropna()
    excess = clean - rf / periods_per_year
    std = excess.std()
    if np.isnan(std) or std < 1e-12:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    clean = returns.dropna()
    excess = clean - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    dd_std = downside.std()
    if np.isnan(dd_std) or dd_std < 1e-12:
        return float("nan")
    return float(np.sqrt(periods_per_year) * excess.mean() / dd_std)


def max_drawdown(equity: pd.Series) -> float:
    clean = equity.dropna()
    if clean.empty:
        return float("nan")
    running_max = clean.cummax()
    drawdown = clean / running_max - 1
    return float(drawdown.min())


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(equity, periods_per_year) / abs(mdd))


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = pd.Index(["r", "b"])
    var_b = aligned["b"].var()
    if var_b == 0 or np.isnan(var_b):
        return float("nan")
    cov = aligned["r"].cov(aligned["b"])
    return float(cov / var_b)


def jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    b = beta(returns, benchmark_returns)
    if np.isnan(b):
        return float("nan")
    r_ann = returns.dropna().mean() * periods_per_year
    bm_ann = benchmark_returns.dropna().mean() * periods_per_year
    return float(r_ann - (rf + b * (bm_ann - rf)))


def performance_summary(
    equity: pd.Series,
    *,
    benchmark_equity: pd.Series | None = None,
    rf: float = 0.0,
) -> pd.Series:
    """Bundle the headline metrics into a single ``Series`` for display."""
    strat_returns = equity.pct_change().dropna()
    summary: dict[str, float] = {
        "CAGR": cagr(equity),
        "Vol (ann.)": annualized_volatility(strat_returns),
        "Sharpe": sharpe(strat_returns, rf=rf),
        "Sortino": sortino(strat_returns, rf=rf),
        "Max DD": max_drawdown(equity),
        "Calmar": calmar(equity),
    }
    if benchmark_equity is not None:
        bench_returns = benchmark_equity.pct_change().dropna()
        summary["Beta vs BM"] = beta(strat_returns, bench_returns)
        summary["Alpha vs BM (ann.)"] = jensens_alpha(strat_returns, bench_returns, rf=rf)
    return pd.Series(summary)
