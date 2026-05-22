from __future__ import annotations

import numpy as np
import pandas as pd

from safra_quant.strategy import Backtest, RSIStrategy, buy_and_hold


def _synthetic_oversold_overbought() -> pd.Series:
    """A series that drops to trigger a buy, then rallies to trigger a sell."""
    down = np.linspace(100, 70, 40)  # falling for 40 days → RSI low
    flat = np.full(5, 70.0)
    up = np.linspace(70, 130, 40)  # rising for 40 days → RSI high
    arr = np.concatenate([down, flat, up])
    idx = pd.bdate_range("2020-01-01", periods=len(arr), freq="B")
    return pd.Series(arr, index=idx, name="X")


def test_no_signals_keeps_initial_cash() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100, freq="B")
    flat = pd.Series(100.0, index=idx)
    bt = Backtest(RSIStrategy(window=14), prices=flat, initial_cash=50_000.0).run()
    assert bt.equity.iloc[-1] == 50_000.0
    assert bt.n_trades == 0


def test_oversold_then_overbought_produces_buy_and_sell() -> None:
    prices = _synthetic_oversold_overbought()
    bt = Backtest(
        RSIStrategy(
            buy_threshold=25, sell_threshold=75, window=14, trade_size=100, max_position=500
        ),
        prices=prices,
        initial_cash=100_000.0,
    ).run()
    sides = bt.trades["side"].tolist()
    assert "BUY" in sides
    assert any(s in ("SELL", "CLOSE") for s in sides)


def test_open_position_is_marked_to_market_at_end() -> None:
    """Regression test for the v1 PnL bug: equity must reflect open positions."""
    down = pd.Series(
        np.linspace(100, 70, 60), index=pd.bdate_range("2020-01-01", periods=60, freq="B")
    )
    bt = Backtest(
        RSIStrategy(buy_threshold=25, sell_threshold=75, window=14),
        prices=down,
        initial_cash=100_000.0,
    ).run()
    # The strategy buys but never sells (no overbought signal). The CLOSE
    # trade liquidates the position at the final price; equity must equal
    # initial cash minus the realized loss on the trades.
    assert bt.equity.iloc[-1] < 100_000.0
    # And the loss must NOT be the trivial "sum of buys" (which would give a
    # negative absolute value far below realistic levels).
    assert bt.equity.iloc[-1] > 0


def test_max_position_is_respected() -> None:
    prices = _synthetic_oversold_overbought()
    bt = Backtest(
        RSIStrategy(
            buy_threshold=60, sell_threshold=90, window=14, trade_size=100, max_position=200
        ),
        prices=prices,
        initial_cash=200_000.0,
    ).run()
    assert bt.positions.max() <= 200


def test_commission_reduces_equity() -> None:
    prices = _synthetic_oversold_overbought()
    no_fee = Backtest(
        RSIStrategy(commission_bps=0.0, window=14),
        prices=prices,
        initial_cash=100_000.0,
    ).run()
    with_fee = Backtest(
        RSIStrategy(commission_bps=20.0, window=14),
        prices=prices,
        initial_cash=100_000.0,
    ).run()
    if no_fee.n_trades > 0:
        assert with_fee.equity.iloc[-1] <= no_fee.equity.iloc[-1]


def test_trend_filter_blocks_buys_in_downtrend() -> None:
    down = pd.Series(
        np.linspace(200, 50, 250), index=pd.bdate_range("2020-01-01", periods=250, freq="B")
    )
    bt = Backtest(
        RSIStrategy(buy_threshold=30, sell_threshold=70, window=14, trend_filter_window=200),
        prices=down,
        initial_cash=100_000.0,
    ).run()
    buys = bt.trades.query("side == 'BUY'") if not bt.trades.empty else bt.trades
    assert len(buys) == 0


def test_buy_and_hold_matches_price_ratio() -> None:
    idx = pd.bdate_range("2020-01-01", periods=50, freq="B")
    prices = pd.Series(np.linspace(100, 150, 50), index=idx)
    eq = buy_and_hold(prices, initial_cash=1000.0)
    assert abs(eq.iloc[-1] - 1500.0) < 1e-9
