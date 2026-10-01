"""Portfolio risk metrics with dark-mode drawdown charts."""

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


def total_return(close: pd.Series) -> float:
    """Return the total price return of the series.

    Args:
        close: Prices indexed by date.

    Returns:
        Total return as a fraction (e.g. 0.15 for +15%).
    """
    if len(close) < 2:
        return 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Return the annualized volatility of daily returns.

    Args:
        returns: Daily returns indexed by date.

    Returns:
        Annualized volatility as a fraction.
    """
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Return the annualized Sharpe ratio of daily returns.

    Args:
        returns: Daily returns indexed by date.
        risk_free: Annualized risk-free rate.

    Returns:
        Sharpe ratio, or NaN when volatility is zero.
    """
    std = returns.std()
    if std is None or std == 0.0:
        return float("nan")
    excess = returns.mean() - risk_free / TRADING_DAYS
    return float(excess / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Return the annualized Sortino ratio using downside deviation.

    Args:
        returns: Daily returns indexed by date.
        risk_free: Annualized risk-free rate.

    Returns:
        Sortino ratio, or NaN when there is no downside deviation.
    """
    excess = returns - risk_free / TRADING_DAYS
    downside = np.minimum(excess, 0.0)
    downside_std = float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS))
    if downside_std == 0.0:
        return float("nan")
    return float(excess.mean() * TRADING_DAYS / downside_std)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Return beta against a benchmark, aligned on common dates.

    Args:
        returns: Daily returns of the asset.
        benchmark_returns: Daily returns of the benchmark.

    Returns:
        Beta value, or None when dates do not overlap or variance is zero.
    """
    common = returns.index.intersection(benchmark_returns.index)
    if common.empty:
        return None
    aligned = returns.loc[common]
    benchmark = benchmark_returns.loc[common]
    variance = float(benchmark.var())
    if variance == 0.0:
        return None
    return float(aligned.cov(benchmark) / variance)


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Return the historical VaR as a positive loss fraction.

    Args:
        returns: Daily returns indexed by date.
        level: Confidence level (e.g. 0.95).

    Returns:
        VaR as a positive fraction of capital (e.g. 0.03 for -3%).
    """
    return float(-returns.quantile(1.0 - level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Return the historical CVaR (expected shortfall) as a positive loss fraction.

    Args:
        returns: Daily returns indexed by date.
        level: Confidence level (e.g. 0.95).

    Returns:
        Mean loss of the tail beyond VaR, as a positive fraction.
    """
    var = returns.quantile(1.0 - level)
    tail = returns[returns <= var]
    if tail.empty:
        return float(-var)
    return float(-tail.mean())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the drawdown curve of the compounded equity curve.

    Args:
        returns: Daily returns indexed by date.

    Returns:
        Series of drawdowns (0 = at a new high, negative = below peak).
    """
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def plot_drawdown(returns: pd.Series, symbol: str, output: str) -> None:
    """Render a dark-mode drawdown chart with the maximum drawdown marked.

    Args:
        returns: Daily returns indexed by date.
        symbol: Ticker symbol used in the chart title.
        output: Destination path for the PNG figure.
    """
    drawdown = drawdown_series(returns)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)

    ax.fill_between(drawdown.index, drawdown.values, 0.0, color="#f85149", alpha=0.35)
    ax.plot(drawdown.index, drawdown.values, color="#f85149", linewidth=1.2)
    trough_date = drawdown.idxmin()
    trough = drawdown.min()
    ax.axhline(trough, color="#8b949e", linestyle="--", alpha=0.7)
    ax.annotate(
        f"max drawdown {trough:.1%}",
        xy=(trough_date, trough),
        xytext=(10, -12),
        textcoords="offset points",
        color="#e6edf3",
    )

    ax.set_title(f"{symbol} - drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio risk metrics with dark-mode charts")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL, IBE.MC)")
    parser.add_argument("--period", default="1y", help="Lookback window (1mo, 6mo, 1y, ...)")
    parser.add_argument("--benchmark", default=None, help="Benchmark ticker for beta")
    parser.add_argument("--risk-free", type=float, default=0.0, help="Annualized risk-free rate")
    parser.add_argument("--output", default="results/figures/drawdown.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        ohlcv = fetch_daily_ohlcv(args.symbol, args.period)
        close = ohlcv["close"].dropna()
        returns = close.pct_change().dropna()
        benchmark_returns = None
        if args.benchmark:
            benchmark = fetch_daily_ohlcv(args.benchmark, args.period)
            benchmark_returns = benchmark["close"].dropna().pct_change().dropna()
        plot_drawdown(returns, args.symbol, args.output)
    except (MarketDataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"symbol={args.symbol} bars={len(close)}")
    print(f"total_return={total_return(close):.2%}")
    print(f"annualized_volatility={annualized_volatility(returns):.2%}")
    print(f"sharpe_ratio={sharpe_ratio(returns, args.risk_free):.2f}")
    print(f"sortino_ratio={sortino_ratio(returns, args.risk_free):.2f}")
    if benchmark_returns is not None:
        asset_beta = beta(returns, benchmark_returns)
        if asset_beta is not None:
            print(f"beta_vs_{args.benchmark}={asset_beta:.2f}")
    print(f"var_95={value_at_risk(returns):.2%}")
    print(f"cvar_95={conditional_value_at_risk(returns):.2%}")
    print(f"max_drawdown={drawdown_series(returns).min():.2%}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
