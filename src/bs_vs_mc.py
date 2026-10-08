"""Black-Scholes vs Monte Carlo pricing comparison with convergence study."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.black_scholes import bsCall
from src.monte_carlo import GBMParams, simulate

TRADING_DAYS = 252
CONVERGENCE_GRID = (100, 1_000, 10_000, 100_000, 1_000_000)


def mc_price(
    s0: float,
    strike: float,
    rate: float,
    years: float,
    sigma: float,
    n_paths: int,
    option_type: str = "call",
    seed: int = 42,
) -> float:
    """Price a European option by Monte Carlo using the GBM kernel.

    Args:
        s0: Spot price.
        strike: Strike price.
        rate: Annualized risk-free rate.
        years: Time to expiration in years.
        sigma: Annualized volatility.
        n_paths: Number of Monte Carlo paths.
        option_type: ``"call"`` or ``"put"``.
        seed: RNG seed for reproducibility.

    Returns:
        Discounted Monte Carlo price estimate.
    """
    params = GBMParams(
        s0=s0,
        mu=rate,
        sigma=sigma,
        horizon_days=round(years * TRADING_DAYS),
        n_paths=n_paths,
        seed=seed,
    )
    paths = simulate(params)
    terminal = paths[:, -1]
    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    return float(np.exp(-rate * years) * payoff.mean())


def convergence_study(
    s0: float,
    strike: float,
    rate: float,
    years: float,
    sigma: float,
    reference_price: float,
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Measure Monte Carlo absolute error against the analytic price.

    Args:
        s0: Spot price.
        strike: Strike price.
        rate: Annualized risk-free rate.
        years: Time to expiration in years.
        sigma: Annualized volatility.
        reference_price: Analytic (Black-Scholes) price.
        seed: RNG seed for reproducibility.

    Returns:
        List of (n_paths, absolute error) pairs.
    """
    return [
        (n_paths, abs(mc_price(s0, strike, rate, years, sigma, n_paths, seed=seed) - reference_price))
        for n_paths in CONVERGENCE_GRID
    ]


def plot_convergence(errors: list[tuple[int, float]], output: str) -> None:
    """Render a dark-mode log-log plot of MC error versus path count.

    Args:
        errors: (n_paths, absolute error) pairs from :func:`convergence_study`.
        output: Destination path for the PNG figure.
    """
    counts = [pair[0] for pair in errors]
    abs_errors = [max(pair[1], 1e-12) for pair in errors]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    ax.loglog(counts, abs_errors, "o-", color="#58a6ff", linewidth=1.6, label="Monte Carlo error")
    reference = abs_errors[0] * (counts[0] / np.array(counts, dtype=float)) ** 0.5
    ax.loglog(counts, reference, "--", color="#8b949e", alpha=0.7, label="O(1/sqrt(N))")

    ax.set_title("Monte Carlo convergence vs Black-Scholes reference")
    ax.set_xlabel("Paths (N)")
    ax.set_ylabel("Absolute error")
    ax.grid(alpha=0.2, which="both")
    ax.legend(framealpha=0.2)
    fig.tight_layout()

    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Black-Scholes vs Monte Carlo comparison")
    parser.add_argument("--s0", type=float, default=100.0, help="Spot price")
    parser.add_argument("--strike", type=float, default=100.0, help="Strike price")
    parser.add_argument("--rate", type=float, default=0.05, help="Annualized risk-free rate")
    parser.add_argument("--years", type=float, default=0.5, help="Time to expiration in years")
    parser.add_argument("--sigma", type=float, default=0.20, help="Annualized volatility")
    parser.add_argument("--output", default="results/figures/bs_vs_mc.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()

    call_reference = bsCall(args.s0, args.strike, args.rate, args.years, args.sigma)["price"]
    put_reference_price = bsCall(args.s0, args.strike, args.rate, args.years, args.sigma)["put"]

    call_mc = mc_price(args.s0, args.strike, args.rate, args.years, args.sigma, 1_000_000)
    put_mc = mc_price(args.s0, args.strike, args.rate, args.years, args.sigma, 1_000_000, "put")
    errors = convergence_study(args.s0, args.strike, args.rate, args.years, args.sigma, call_reference)
    plot_convergence(errors, args.output)

    print(f"spot={args.s0} strike={args.strike} T={args.years}y sigma={args.sigma}")
    print(f"call: black_scholes={call_reference:.4f} montecarlo={call_mc:.4f} abs_error={abs(call_mc - call_reference):.2e}")
    print(f"put:  black_scholes={put_reference_price:.4f} montecarlo={put_mc:.4f} abs_error={abs(put_mc - put_reference_price):.2e}")
    for n_paths, error in errors:
        print(f"convergence: N={n_paths:>9,} abs_error={error:.2e}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
