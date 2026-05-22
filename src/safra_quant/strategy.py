"""Backtest engine and RSI strategy.

The v1 notebook computed PnL as `sold_amount - bought_amount` in BRL of
notional, never closed open positions, and did not produce an equity curve.
Here we keep a running cash + position state, mark-to-market every day, and
return a proper equity curve for performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from safra_quant.indicators import rsi as compute_rsi
from safra_quant.indicators import sma as compute_sma


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    initial_cash: float

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_cash - 1)

    @property
    def n_trades(self) -> int:
        return len(self.trades)


@dataclass
class RSIStrategy:
    """Long-only RSI mean-reversion strategy.

    Buys ``trade_size`` shares whenever RSI crosses below ``buy_threshold``
    (and current position is below ``max_position``); sells ``trade_size``
    shares whenever RSI crosses above ``sell_threshold`` (and position > 0).
    """

    buy_threshold: float = 25.0
    sell_threshold: float = 75.0
    window: int = 21
    trade_size: int = 100
    max_position: int = 500
    trend_filter_window: int | None = None
    commission_bps: float = 0.0

    def signals(self, close: pd.Series) -> pd.DataFrame:
        rsi_values = compute_rsi(close, window=self.window, method="wilder")
        out = pd.DataFrame({"close": close, "rsi": rsi_values})
        if self.trend_filter_window is not None:
            out["sma"] = compute_sma(close, self.trend_filter_window)
        return out


@dataclass
class Backtest:
    strategy: RSIStrategy
    prices: pd.Series
    initial_cash: float = 100_000.0
    trade_log: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> BacktestResult:
        signals = self.strategy.signals(self.prices)
        cash = self.initial_cash
        position = 0
        equity_curve: list[float] = []
        positions_curve: list[int] = []
        trades: list[dict[str, Any]] = []

        for ts, row in signals.iterrows():
            price = float(row["close"])
            rsi_val = float(row["rsi"]) if not np.isnan(row["rsi"]) else None
            sma_val = float(row.get("sma", np.nan)) if "sma" in signals.columns else None

            trend_ok_long = sma_val is None or (not np.isnan(sma_val) and price > sma_val)

            if rsi_val is not None:
                # Buy signal
                if (
                    rsi_val < self.strategy.buy_threshold
                    and position + self.strategy.trade_size <= self.strategy.max_position
                    and trend_ok_long
                ):
                    notional = price * self.strategy.trade_size
                    fee = notional * self.strategy.commission_bps / 10_000
                    if cash >= notional + fee:
                        cash -= notional + fee
                        position += self.strategy.trade_size
                        trades.append(
                            {
                                "ts": ts,
                                "side": "BUY",
                                "price": price,
                                "qty": self.strategy.trade_size,
                                "fee": fee,
                            }
                        )
                # Sell signal
                elif rsi_val > self.strategy.sell_threshold and position > 0:
                    qty = min(self.strategy.trade_size, position)
                    notional = price * qty
                    fee = notional * self.strategy.commission_bps / 10_000
                    cash += notional - fee
                    position -= qty
                    trades.append(
                        {"ts": ts, "side": "SELL", "price": price, "qty": qty, "fee": fee}
                    )

            equity_curve.append(cash + position * price)
            positions_curve.append(position)

        # Close any remaining position at last available price (mark to market
        # via equity, but also produce an explicit closing trade for clarity).
        if position > 0:
            last_price = float(self.prices.iloc[-1])
            notional = last_price * position
            fee = notional * self.strategy.commission_bps / 10_000
            cash += notional - fee
            trades.append(
                {
                    "ts": self.prices.index[-1],
                    "side": "CLOSE",
                    "price": last_price,
                    "qty": position,
                    "fee": fee,
                }
            )
            position = 0
            equity_curve[-1] = cash

        equity = pd.Series(equity_curve, index=signals.index, name="equity")
        positions = pd.Series(positions_curve, index=signals.index, name="position")
        trade_df = pd.DataFrame(trades)
        if not trade_df.empty:
            trade_df = trade_df.set_index("ts")
        return BacktestResult(
            equity=equity,
            positions=positions,
            trades=trade_df,
            initial_cash=self.initial_cash,
        )


def buy_and_hold(prices: pd.Series, initial_cash: float = 100_000.0) -> pd.Series:
    """Reference equity curve for a 100% allocation at t=0."""
    units = initial_cash / float(prices.iloc[0])
    return (prices * units).rename("equity")
