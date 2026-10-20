"""Rolling Hurst exponent and execution regime classification.

Implements the framework from Kulkarni & Nayar (arXiv:2610.23749):
the optimal execution schedule depends on the Hurst exponent H of the
price series and the impact elasticity. When H > 0.55 (momentum /
superdiffusive), front-load execution; when H < 0.45 (reversive /
subdiffusive), slow down and let the market absorb impact.
"""

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
REGIME_REVERSIVE = "REVERSIVE (subdiffusive)"
REGIME_RANDOM = "RANDOM WALK"
REGIME_MOMENTUM = "MOMENTUM (superdiffusive)"


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent via rescaled range (R/S) analysis.

    Expects an INCREMENT series (returns, diffs), not price levels.
    For price series, use :func:`rolling_hurst` which converts internally.

    Args:
        series: Increment/return series.
        max_lag: Maximum lag for the R/S computation.

    Returns:
        Hurst exponent H (0 = reversive, 0.5 = random walk, 1 = trending).

    Raises:
        ValueError: If the series is too short or max_lag exceeds its length.
    """
    if len(series) < max_lag * 2:
        raise ValueError(f"series needs at least {max_lag * 2} points, got {len(series)}")
    lags = np.arange(2, max_lag)
    rs_values = []
    for lag in lags:
        chunks = np.array_split(series.values, len(series) // lag)
        chunk_rs = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = np.mean(chunk)
            deviations = np.cumsum(chunk - mean)
            rng = np.max(deviations) - np.min(deviations)
            std = np.std(chunk)
            if std > 0:
                chunk_rs.append(rng / std)
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
    if len(rs_values) < 5:
        raise ValueError("not enough valid R/S points for regression")
    log_lags = np.log(lags[: len(rs_values)])
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(
    series: pd.Series, window: int = 252, step: int = 21, max_lag: int = 50
) -> pd.Series:
    """Compute the rolling Hurst exponent over a price series.

    Args:
        series: Price series indexed by date.
        window: Lookback window in observations (default: 1 year of daily data).
        step: Step between windows (default: 1 month).
        max_lag: Maximum lag for each R/S computation.

    Returns:
        Series of Hurst exponents indexed by the window end date.

    Raises:
        ValueError: If window is too small or series too short.
    """
    if window < max_lag * 2:
        raise ValueError(f"window must be >= {max_lag * 2}, got {window}")
    if len(series) < window:
        raise ValueError(f"series needs at least {window} points, got {len(series)}")

    results = {}
    returns = np.log(series / series.shift(1)).dropna()
    for end in range(window, len(returns) + 1, step):
        subset = returns.iloc[end - window : end]
        date = series.index[end]
        try:
            results[date] = hurst_exponent(subset, max_lag)
        except ValueError:
            continue
    return pd.Series(results, dtype=float)


def classify_regime(h: float) -> str:
    """Classify the market regime from the Hurst exponent.

    Args:
        h: Hurst exponent value.

    Returns:
        Human-readable regime label.
    """
    if h < 0.45:
        return REGIME_REVERSIVE
    if h > 0.55:
        return REGIME_MOMENTUM
    return REGIME_RANDOM


def execution_schedule(
    h: float, beta: float = 0.6, horizon_steps: int = 20
) -> np.ndarray:
    """Compute a front-loaded execution schedule per arXiv:2610.23749.

    The optimal schedule is proportional to (lambda - w(t))^+(1/beta),
    where w(t) = t^H is the Hurst-scaled envelope cost. Higher H
    (momentum) produces more front-loaded schedules.

    Args:
        h: Hurst exponent of the current regime.
        beta: Impact elasticity (typical range 0.4-0.8).
        horizon_steps: Number of execution time slices.

    Returns:
        Normalized schedule weights (sum to 1.0).
    """
    t = np.linspace(0, 1, horizon_steps)
    w = t**h
    lam = 1.0
    raw = np.maximum(lam - w, 0) ** (1.0 / beta)
    total = raw.sum()
    if total == 0:
        return np.full(horizon_steps, 1.0 / horizon_steps)
    return raw / total


def recommend_pace(h: float) -> str:
    """Produce a plain-English execution recommendation from H.

    Args:
        h: Hurst exponent value.

    Returns:
        Actionable recommendation string.
    """
    regime = classify_regime(h)
    if regime == REGIME_MOMENTUM:
        return (
            f"H={h:.3f} — MOMENTUM regime. Front-load execution: "
            "work the order faster in early slices to avoid directional slippage."
        )
    if regime == REGIME_REVERSIVE:
        return (
            f"H={h:.3f} — REVERSIVE regime. Slow down: "
            "spread execution evenly (or delay) to let the market absorb impact."
        )
    return (
        f"H={h:.3f} — RANDOM WALK. Standard TWAP is near-optimal; "
        "no urgency to deviate from a uniform schedule."
    )


def plot_hurst_analysis(
    close: pd.Series, hurst: pd.Series, symbol: str, output: str
) -> None:
    """Render a dark-mode chart with price and rolling Hurst exponent.

    Args:
        close: Close prices indexed by date.
        hurst: Rolling Hurst exponent indexed by date.
        symbol: Ticker symbol for the title.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, (ax_price, ax_hurst) = plt.subplots(
        2, 1, figsize=(12, 7), dpi=120, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax_price.plot(close.index, close.values, color="#4C9AFF", linewidth=1.4)
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{symbol} — rolling Hurst exponent and execution regime")
    ax_price.grid(alpha=0.2)

    ax_hurst.plot(hurst.index, hurst.values, color="#FF7A59", linewidth=1.2)
    ax_hurst.axhline(0.55, color="#9ECB45", linestyle="--", alpha=0.6, label="Momentum threshold (0.55)")
    ax_hurst.axhline(0.45, color="#4C9AFF", linestyle="--", alpha=0.6, label="Reversive threshold (0.45)")
    ax_hurst.axhline(0.5, color="#8b949e", linestyle=":", alpha=0.4, label="Random walk (0.5)")
    ax_hurst.fill_between(hurst.index, 0.45, 0.55, color="#8b949e", alpha=0.08)
    ax_hurst.set_ylabel("Hurst exponent H")
    ax_hurst.set_ylim(0.2, 0.9)
    ax_hurst.set_xlabel("Date")
    ax_hurst.grid(alpha=0.2)
    ax_hurst.legend(loc="lower left", framealpha=0.2, fontsize=8)

    fig.tight_layout()
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Rolling Hurst exponent analysis")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL, BTC-USD)")
    parser.add_argument("--period", default="2y", help="Lookback window (2y, 5y, max)")
    parser.add_argument("--window", type=int, default=252, help="Rolling window (default: 252 = 1y)")
    parser.add_argument("--max-lag", type=int, default=50, help="Max lag for R/S (default: 50)")
    parser.add_argument("--output", default="results/figures/hurst.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        ohlcv = fetch_daily_ohlcv(args.symbol, args.period)
        close = ohlcv["close"].dropna()
        hurst = rolling_hurst(close, window=args.window, max_lag=args.max_lag)
        if hurst.empty:
            print("error: not enough data for rolling Hurst", file=sys.stderr)
            return 1
        latest = hurst.iloc[-1]
        print(f"symbol={args.symbol} windows={len(hurst)}")
        print(f"latest_h={latest:.4f} regime={classify_regime(latest)}")
        print(recommend_pace(latest))
        schedule = execution_schedule(latest)
        front_pct = schedule[: len(schedule) // 4].sum() * 100
        print(f"front_quarter_weight={front_pct:.1f}% (TWAP would be 25.0%)")
        plot_hurst_analysis(close, hurst, args.symbol, args.output)
        print(f"figure={args.output}")
    except (MarketDataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
