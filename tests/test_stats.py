from __future__ import annotations

import math

import numpy as np
import pandas as pd

from safra_quant.stats import (
    beta,
    cagr,
    calmar,
    distribution,
    jensens_alpha,
    max_drawdown,
    performance_summary,
    sharpe,
    sortino,
)


def test_distribution_matches_pandas() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    d = distribution(s)
    assert d.mean == 3.0
    assert d.median == 3.0
    assert math.isclose(d.variance, s.var())
    assert d.q1 == 2.0
    assert d.q3 == 4.0


def test_max_drawdown_known_path() -> None:
    equity = pd.Series([100, 120, 90, 110, 80, 130])
    # Peak 120 → trough 80 → drawdown = -1/3
    assert math.isclose(max_drawdown(equity), -1 / 3, abs_tol=1e-12)


def test_cagr_one_year_double() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    equity = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    assert abs(cagr(equity) - 1.0) < 0.01  # ~100% over one trading year


def test_sharpe_handles_zero_volatility() -> None:
    returns = pd.Series([0.001] * 100)
    assert math.isnan(sharpe(returns)) or math.isinf(sharpe(returns))


def test_sortino_greater_than_sharpe_for_low_downside_volatility() -> None:
    """When most negative moves are small and positive moves are larger,
    Sortino should exceed Sharpe (denominator excludes the larger upside)."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.005, 1000)  # small symmetric noise
    base[::20] = abs(base[::20]) * 4 + 0.02  # occasional large upside
    rets = pd.Series(base)
    assert sortino(rets) > sharpe(rets)


def test_beta_identical_series_is_one() -> None:
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 250))
    assert math.isclose(beta(bench, bench), 1.0, abs_tol=1e-12)


def test_jensens_alpha_zero_when_returns_equal_benchmark() -> None:
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0, 0.01, 250))
    assert abs(jensens_alpha(bench, bench)) < 1e-9


def test_calmar_positive_for_growing_equity() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    equity = pd.Series(np.linspace(100, 150, len(idx)), index=idx)
    # Monotonic growth → max DD ≈ 0 → calmar tends to infinity or large.
    val = calmar(equity)
    assert math.isnan(val) or val > 1e6


def test_performance_summary_keys() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    equity = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    bench = pd.Series(np.linspace(100, 115, len(idx)), index=idx)
    summary = performance_summary(equity, benchmark_equity=bench)
    expected = {
        "CAGR",
        "Vol (ann.)",
        "Sharpe",
        "Sortino",
        "Max DD",
        "Calmar",
        "Beta vs BM",
        "Alpha vs BM (ann.)",
    }
    assert expected.issubset(set(summary.index))
