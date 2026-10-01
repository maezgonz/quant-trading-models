"""SMA crossover backtest on ingested market data, dark-mode charts."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_ingestion import MarketDataError, fetch_daily_ohlcv

TRADING_DAYS = 252


def sma_signals(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    """Build an SMA crossover signal frame.

    The signal is 1.0 (long) while the fast SMA is above the slow SMA and
    0.0 (flat) otherwise; rows before the slow SMA exists are flat.

    Args:
        close: Close prices indexed by date.
        fast: Fast SMA window in days.
        slow: Slow SMA window in days.

    Returns:
        DataFrame with ``sma_fast``, ``sma_slow`` and ``signal`` columns.

    Raises:
        ValueError: If windows are non-positive or fast >= slow.
    """
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("require 0 < fast < slow")

    frame = pd.DataFrame({"close": close})
    frame["sma_fast"] = close.rolling(fast).mean()
    frame["sma_slow"] = close.rolling(slow).mean()
    frame["signal"] = np.where(frame["sma_fast"] > frame["sma_slow"], 1.0, 0.0)
    return frame


def backtest(signals: pd.DataFrame, capital: float = 10_000.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a vectorized long/flat backtest of the signal column.

    Positions are taken on the day after the signal fires (shift by one),
    so no lookahead bias is introduced. Transaction costs are ignored.

    Args:
        signals: Frame as produced by :func:`sma_signals`.
        capital: Initial capital.

    Returns:
        Tuple of (frame enriched with ``position``, ``strategy_return``,
        ``equity`` and ``buy_hold_equity`` columns, metrics dict).
    """
    frame = signals.copy()
    returns = frame["close"].pct_change().fillna(0.0)
    frame["position"] = frame["signal"].shift(1).fillna(0.0)
    frame["strategy_return"] = frame["position"] * returns

    frame["equity"] = capital * (1.0 + frame["strategy_return"]).cumprod()
    frame["buy_hold_equity"] = capital * (1.0 + returns).cumprod()

    total_return = frame["equity"].iloc[-1] / capital - 1.0
    buy_hold_return = frame["buy_hold_equity"].iloc[-1] / capital - 1.0

    strategy_std = frame["strategy_return"].std()
    sharpe = 0.0
    if strategy_std is not None and strategy_std > 0.0:
        sharpe = frame["strategy_return"].mean() / strategy_std * (TRADING_DAYS**0.5)

    drawdown = frame["equity"] / frame["equity"].cummax() - 1.0
    metrics = {
        "total_return": float(total_return),
        "buy_hold_return": float(buy_hold_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "exposure": float(frame["position"].mean()),
    }
    return frame, metrics


def plot_backtest(frame: pd.DataFrame, symbol: str, output: str) -> None:
    """Render a dark-mode chart of signals and equity curves.

    Args:
        frame: Enriched backtest frame from :func:`backtest`.
        symbol: Ticker symbol used in the chart title.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, (ax_price, ax_equity) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=120, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_price.plot(frame.index, frame["close"], color="#58a6ff", linewidth=1.4, label="Close")
    ax_price.plot(frame.index, frame["sma_fast"], color="#f78166", linewidth=1.0, label="SMA fast")
    ax_price.plot(frame.index, frame["sma_slow"], color="#d2a8ff", linewidth=1.0, label="SMA slow")

    trades = frame["position"].diff()
    entries = frame.index[trades == 1.0]
    exits = frame.index[trades == -1.0]
    ax_price.scatter(entries, frame.loc[entries, "close"], marker="^", color="#3fb950", s=60, zorder=3, label="Entry")
    ax_price.scatter(exits, frame.loc[exits, "close"], marker="v", color="#f85149", s=60, zorder=3, label="Exit")

    ax_price.set_title(f"{symbol} - SMA crossover signals")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left", framealpha=0.2)

    ax_equity.plot(frame.index, frame["equity"], color="#3fb950", linewidth=1.4, label="Strategy")
    ax_equity.plot(frame.index, frame["buy_hold_equity"], color="#8b949e", linewidth=1.0, label="Buy & hold")
    ax_equity.set_ylabel("Equity")
    ax_equity.grid(alpha=0.2)
    ax_equity.legend(loc="upper left", framealpha=0.2)

    fig.tight_layout()
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SMA crossover backtest with dark-mode charts")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL, IBE.MC)")
    parser.add_argument("--period", default="1y", help="Lookback window (1mo, 6mo, 1y, ...)")
    parser.add_argument("--fast", type=int, default=20, help="Fast SMA window (default: 20)")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window (default: 50)")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--output", default="results/figures/backtest.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        ohlcv = fetch_daily_ohlcv(args.symbol, args.period)
        signals = sma_signals(ohlcv["close"].dropna(), args.fast, args.slow)
        frame, metrics = backtest(signals, args.capital)
        plot_backtest(frame, args.symbol, args.output)
    except (MarketDataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"symbol={args.symbol} fast={args.fast} slow={args.slow} bars={len(frame)}")
    for key, value in metrics.items():
        if key in ("total_return", "buy_hold_return", "max_drawdown"):
            print(f"{key}={value:.2%}")
        else:
            print(f"{key}={value:.2f}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
