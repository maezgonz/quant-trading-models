"""Unit tests for the GBM Monte Carlo kernel (deterministic, no network)."""

import numpy as np
import pytest

from src.monte_carlo import GBMParams, simulate, summarize


def test_gbm_params_validation():
    """Invalid parameters must raise ValueError."""
    with pytest.raises(ValueError):
        GBMParams(s0=0.0, mu=0.08, sigma=0.25)
    with pytest.raises(ValueError):
        GBMParams(s0=100.0, mu=0.08, sigma=-0.1)
    with pytest.raises(ValueError):
        GBMParams(s0=100.0, mu=0.08, sigma=0.25, horizon_days=0)
    with pytest.raises(ValueError):
        GBMParams(s0=100.0, mu=0.08, sigma=0.25, n_paths=0)


def test_gbm_shapes():
    """simulate() must return an array of shape (n_paths, horizon_days)."""
    params = GBMParams(s0=100.0, mu=0.08, sigma=0.25, horizon_days=30, n_paths=500, seed=42)
    paths = simulate(params)
    assert paths.shape == (500, 30)
    assert np.all(paths > 0)


def test_gbm_statistics_sanity():
    """With a fixed seed the median terminal must match the log-normal reference."""
    params = GBMParams(s0=100.0, mu=0.08, sigma=0.25, horizon_days=252, n_paths=20000, seed=42)
    stats = summarize(simulate(params))
    years = 252 / 252
    expected_median = 100.0 * np.exp((0.08 - 0.5 * 0.25**2) * years)
    assert abs(stats["median_terminal"] - expected_median) / expected_median < 0.05
    assert stats["p05_terminal"] < stats["median_terminal"] < stats["p95_terminal"]


def test_gbm_reproducible():
    """Same seed must produce identical paths."""
    params = GBMParams(s0=100.0, mu=0.08, sigma=0.25, horizon_days=30, n_paths=200, seed=7)
    assert np.array_equal(simulate(params), simulate(params))
