"""Monte Carlo projection of asset prices under Geometric Brownian Motion."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class GBMParams:
    """Parameters of the Geometric Brownian Motion model."""

    s0: float
    mu: float
    sigma: float
    horizon_days: int = TRADING_DAYS
    n_paths: int = 50_000
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.s0 <= 0:
            raise ValueError("s0 must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")


def simulate(params: GBMParams) -> np.ndarray:
    """Simulate GBM price paths with a vectorized NumPy kernel.

    Args:
        params: Model parameters.

    Returns:
        Array of shape ``(n_paths, horizon_days)`` with simulated prices.
    """
    rng = np.random.default_rng(params.seed)
    dt = 1.0 / TRADING_DAYS
    shocks = rng.standard_normal((params.n_paths, params.horizon_days))
    drift = (params.mu - 0.5 * params.sigma**2) * dt
    diffusion = params.sigma * np.sqrt(dt) * shocks
    return params.s0 * np.exp(np.cumsum(drift + diffusion, axis=1))


def summarize(paths: np.ndarray) -> dict[str, float]:
    """Return terminal price statistics for the simulated paths.

    Args:
        paths: Simulated price paths of shape ``(n_paths, horizon_days)``.

    Returns:
        Mapping of statistic name to value.
    """
    terminal = paths[:, -1]
    p05, p50, p95 = np.percentile(terminal, [5, 50, 95])
    return {
        "expected_terminal": float(terminal.mean()),
        "median_terminal": float(p50),
        "p05_terminal": float(p05),
        "p95_terminal": float(p95),
    }


def plot_projection(paths: np.ndarray, params: GBMParams, output: str) -> None:
    """Render a dark-mode projection chart with percentile bands.

    Args:
        paths: Simulated price paths of shape ``(n_paths, horizon_days)``.
        params: Model parameters used for labels and the day-0 anchor.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    timeline = np.arange(paths.shape[1] + 1)
    bands = np.column_stack(
        [np.full(5, params.s0), np.percentile(paths, [5, 25, 50, 75, 95], axis=0)]
    )

    ax.fill_between(timeline, bands[0], bands[4], color="#58a6ff", alpha=0.15, label="5th-95th pct")
    ax.fill_between(timeline, bands[1], bands[3], color="#58a6ff", alpha=0.25, label="25th-75th pct")
    ax.plot(timeline, bands[2], color="#f78166", linewidth=1.8, label="Median path")

    for path in paths[:30]:
        ax.plot(timeline[1:], path, color="#8b949e", alpha=0.15, linewidth=0.6)

    ax.set_title(f"Monte Carlo projection - {params.n_paths:,} GBM paths")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.2)
    fig.tight_layout()

    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo GBM price projection with dark-mode charts"
    )
    parser.add_argument("--ticker", default="ASSET", help="Ticker label for the chart title")
    parser.add_argument("--s0", type=float, default=100.0, help="Initial price")
    parser.add_argument("--mu", type=float, default=0.08, help="Annualized drift")
    parser.add_argument("--sigma", type=float, default=0.25, help="Annualized volatility")
    parser.add_argument("--horizon", type=int, default=TRADING_DAYS, help="Horizon in trading days")
    parser.add_argument("--paths", type=int, default=50_000, help="Number of Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--output", default="results/figures/projection.png", help="Output PNG path"
    )
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        params = GBMParams(
            s0=args.s0,
            mu=args.mu,
            sigma=args.sigma,
            horizon_days=args.horizon,
            n_paths=args.paths,
            seed=args.seed,
        )
        paths = simulate(params)
        stats = summarize(paths)
        plot_projection(paths, params, args.output)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ticker={args.ticker}")
    for key, value in stats.items():
        print(f"{key}={value:.2f}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
