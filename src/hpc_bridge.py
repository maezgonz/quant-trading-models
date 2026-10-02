"""Bridge between the C GBM engine output and Python analytics (dark mode)."""

from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.monte_carlo import GBMParams, simulate

REQUIRED_COLUMNS = {"day", "mean", "std", "p05", "p50", "p95"}


def read_engine_csv(csv_path: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Parse a gbm_engine CSV: '# key=value' metadata header plus day rows.

    Args:
        csv_path: Path to the CSV emitted by the C GBM engine.

    Returns:
        Tuple of (metadata mapping with float values, band DataFrame).

    Raises:
        ValueError: If the band table is missing required columns.
        OSError: If the file cannot be read.
    """
    meta: dict[str, float] = {}
    data_lines: list[str] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                for token in line.lstrip("#").split():
                    if "=" in token:
                        key, _, value = token.partition("=")
                        try:
                            meta[key] = float(value)
                        except ValueError:
                            pass
            elif line.strip():
                data_lines.append(line)

    frame = pd.read_csv(io.StringIO("".join(data_lines)))
    if not REQUIRED_COLUMNS.issubset(frame.columns):
        raise ValueError(f"engine CSV must contain columns {sorted(REQUIRED_COLUMNS)}")
    return meta, frame


def compare_with_numpy(meta: dict[str, float], n_paths: int = 50_000) -> dict[str, float]:
    """Re-run the NumPy GBM kernel with the engine parameters and compare.

    Args:
        meta: Engine metadata with ``s0``, ``mu``, ``sigma``, ``horizon``.
        n_paths: Path count for the NumPy comparison sample.

    Returns:
        Mapping with NumPy terminal statistics and relative differences
        against the engine's exact values (``exact_terminal_*`` keys).
    """
    params = GBMParams(
        s0=meta["s0"],
        mu=meta["mu"],
        sigma=meta["sigma"],
        horizon_days=int(meta["horizon"]),
        n_paths=n_paths,
        seed=int(meta.get("seed", 42)),
    )
    paths = simulate(params)
    terminal = paths[:, -1]

    numpy_mean = float(terminal.mean())
    numpy_std = float(terminal.std())
    exact_mean = meta.get("exact_terminal_mean")
    exact_std = meta.get("exact_terminal_std")

    result = {
        "numpy_terminal_mean": numpy_mean,
        "numpy_terminal_std": numpy_std,
    }
    if exact_mean:
        result["mean_rel_diff"] = abs(numpy_mean - exact_mean) / exact_mean
    if exact_std:
        result["std_rel_diff"] = abs(numpy_std - exact_std) / exact_std
    return result


def plot_bands(meta: dict[str, float], frame: pd.DataFrame, output: str) -> None:
    """Render a dark-mode chart of the engine's per-day percentile bands.

    Args:
        meta: Engine metadata used for the chart title.
        frame: Band DataFrame as produced by :func:`read_engine_csv`.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    timeline = frame["day"]
    ax.fill_between(timeline, frame["p05"], frame["p95"], color="#58a6ff", alpha=0.18, label="5th-95th pct")
    ax.plot(timeline, frame["mean"], color="#f78166", linewidth=1.8, label="Mean (C engine)")
    ax.plot(timeline, frame["p50"], color="#d2a8ff", linewidth=1.2, label="Median")

    n_paths = int(meta.get("n_paths", 0))
    ax.set_title(f"GBM engine - {n_paths:,} paths (antithetic variates)")
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
        description="Bridge C GBM engine output into Python analytics (dark mode)"
    )
    parser.add_argument("--csv", required=True, help="Bands CSV emitted by gbm_engine")
    parser.add_argument("--paths", type=int, default=50_000, help="NumPy comparison sample size")
    parser.add_argument("--output", default="results/figures/hpc_bands.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        meta, frame = read_engine_csv(args.csv)
        comparison = compare_with_numpy(meta, args.paths)
        plot_bands(meta, frame, args.output)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"engine_paths={int(meta.get('n_paths', 0)):,} horizon={int(meta.get('horizon', 0))}")
    if "exact_terminal_mean" in meta:
        print(f"engine_exact_terminal_mean={meta['exact_terminal_mean']:.2f}")
        print(f"engine_exact_terminal_std={meta['exact_terminal_std']:.2f}")
    print(f"numpy_terminal_mean={comparison['numpy_terminal_mean']:.2f}")
    print(f"numpy_terminal_std={comparison['numpy_terminal_std']:.2f}")
    if "mean_rel_diff" in comparison:
        print(f"mean_rel_diff={comparison['mean_rel_diff']:.2%}")
        print(f"std_rel_diff={comparison['std_rel_diff']:.2%}")
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
