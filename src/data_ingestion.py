"""Market data ingestion from public REST APIs with dark-mode charting."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

CHART_API = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) quant-trading-models"


class MarketDataError(Exception):
    """Raised when market data cannot be fetched or parsed."""


def fetch_daily_ohlcv(symbol: str, period: str = "1y", interval: str = "1d", timeout: float = 15.0) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a symbol from the Yahoo chart API.

    Args:
        symbol: Ticker symbol (e.g. ``AAPL``, ``IBE.MC``).
        period: Lookback window (e.g. ``1mo``, ``6mo``, ``1y``).
        interval: Bar interval (``1d`` by default).
        timeout: HTTP timeout in seconds.

    Returns:
        DataFrame indexed by date with ``open``, ``high``, ``low``, ``close``
        and ``volume`` columns.

    Raises:
        MarketDataError: On HTTP failure, empty payload or malformed response.
    """
    try:
        response = requests.get(
            CHART_API.format(symbol=symbol),
            params={"range": period, "interval": interval},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise MarketDataError(f"request failed for {symbol}: {exc}") from exc
    except ValueError as exc:
        raise MarketDataError(f"non-JSON response for {symbol}") from exc

    try:
        result = payload["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise MarketDataError(f"malformed chart payload for {symbol}") from exc

    frame = pd.DataFrame(
        {
            "date": [datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in timestamps],
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    ).set_index("date")

    if frame.empty:
        raise MarketDataError(f"no bars returned for {symbol} ({period})")
    return frame


def compute_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns from a close-price series.

    Args:
        close: Close prices indexed by date.

    Returns:
        Series of daily log returns (first row dropped).
    """
    return np.log(close / close.shift(1)).dropna()


def plot_close_series(frame: pd.DataFrame, symbol: str, output: str) -> None:
    """Render a dark-mode chart of close prices and volume.

    Args:
        frame: OHLCV DataFrame as produced by :func:`fetch_daily_ohlcv`.
        symbol: Ticker symbol used in the chart title.
        output: Destination path for the PNG figure.
    """
    close = frame["close"].dropna()

    plt.style.use("dark_background")
    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=120, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(close.index, close.values, color="#58a6ff", linewidth=1.5)
    mean_line = close.rolling(window=20).mean()
    ax_price.plot(close.index, mean_line.values, color="#f78166", linewidth=1.2, label="SMA-20")
    ax_price.set_title(f"{symbol} - daily close")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.2)
    ax_price.legend(loc="upper left", framealpha=0.2)

    ax_volume.bar(close.index, frame["volume"].reindex(close.index).fillna(0).values, color="#8b949e", alpha=0.6)
    ax_volume.set_ylabel("Volume")
    ax_volume.grid(alpha=0.2)

    fig.tight_layout()
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fetch market data and render dark-mode charts")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL, IBE.MC)")
    parser.add_argument("--period", default="1y", help="Lookback window (1mo, 6mo, 1y, ...)")
    parser.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    parser.add_argument("--output", default="results/figures/ingestion.png", help="Output PNG path")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        frame = fetch_daily_ohlcv(args.symbol, args.period, args.interval, args.timeout)
        plot_close_series(frame, args.symbol, args.output)
    except MarketDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    close = frame["close"].dropna()
    returns = compute_returns(close)
    annualized_vol = returns.std() * (252**0.5)
    print(f"symbol={args.symbol} bars={len(frame)}")
    print(f"first={frame.index[0]} last={frame.index[-1]}")
    print(f"last_close={close.iloc[-1]:.2f} annualized_vol={annualized_vol:.2%}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
