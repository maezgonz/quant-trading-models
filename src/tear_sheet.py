"""Performance tear sheets in HTML via quantstats."""

from __future__ import annotations

import argparse
import sys

import quantstats as qs

from src.data_ingestion import MarketDataError, fetch_daily_ohlcv


def build_tear_sheet(
    returns, benchmark_returns, symbol: str, output: str, risk_free: float = 0.0
) -> None:
    """Generate a full HTML tear sheet from daily returns.

    Args:
        returns: Daily returns indexed by date.
        benchmark_returns: Daily benchmark returns (or None).
        symbol: Ticker symbol used as the report title.
        output: Destination path for the HTML report.
        risk_free: Annualized risk-free rate.
    """
    qs.reports.html(
        returns,
        benchmark=benchmark_returns,
        title=f"{symbol} - performance tear sheet",
        output=output,
        rf=risk_free,
        download_filename=output,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantstats HTML tear sheet")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g. AAPL, GGAL.BA)")
    parser.add_argument("--period", default="2y", help="Lookback window (1mo, 6mo, 2y, ...)")
    parser.add_argument("--benchmark", default=None, help="Benchmark ticker for comparison")
    parser.add_argument("--risk-free", type=float, default=0.0, help="Annualized risk-free rate")
    parser.add_argument("--output", default="results/tear_sheet.html", help="Output HTML path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        ohlcv = fetch_daily_ohlcv(args.symbol, args.period)
        returns = ohlcv["close"].dropna().pct_change().dropna()
        benchmark_returns = None
        if args.benchmark:
            benchmark = fetch_daily_ohlcv(args.benchmark, args.period)
            benchmark_returns = benchmark["close"].dropna().pct_change().dropna()
        build_tear_sheet(returns, benchmark_returns, args.symbol, args.output, args.risk_free)
    except (MarketDataError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"symbol={args.symbol} bars={len(returns) + 1} benchmark={args.benchmark or 'none'}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
