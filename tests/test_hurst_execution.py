"""Unit tests for Hurst exponent calculation and execution schedules."""

import numpy as np
import pandas as pd
import pytest

from src.hurst_execution import (
    REGIME_MOMENTUM,
    REGIME_RANDOM,
    REGIME_REVERSIVE,
    classify_regime,
    execution_schedule,
    hurst_exponent,
    recommend_pace,
    rolling_hurst,
)


def _synthetic_series(n: int, hurst: float, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series with a target Hurst exponent."""
    rng = np.random.default_rng(seed)
    # Fractional Brownian motion via Cholesky (small n for tractability)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.5 * (
                (i + 1) ** (2 * hurst) + (j + 1) ** (2 * hurst) - abs(i - j) ** (2 * hurst)
            )
    cholesky = np.linalg.cholesky(cov + np.eye(n) * 1e-10)
    increments = cholesky @ rng.standard_normal(n)
    return pd.Series(np.cumsum(increments))


class TestHurstExponent:
    def test_random_walk(self):
        """Pure random increments should give H near 0.5 (R/S has finite-sample bias, so relaxed)."""
        rng = np.random.default_rng(42)
        increments = pd.Series(rng.standard_normal(1000))
        h = hurst_exponent(increments, max_lag=50)
        assert 0.30 < h < 0.70, f"H={h} outside acceptable range for random walk"

    def test_trending_series(self):
        """Positively autocorrelated increments should give higher H than negatively autocorrelated ones."""
        rng = np.random.default_rng(42)
        n = 2000
        # AR(1) with positive coefficient (persistent increments)
        phi_pos = 0.5
        persistent = np.zeros(n)
        for i in range(1, n):
            persistent[i] = phi_pos * persistent[i - 1] + rng.standard_normal() * 0.87
        h_persistent = hurst_exponent(pd.Series(persistent), max_lag=50)

        # AR(1) with negative coefficient (anti-persistent increments)
        phi_neg = -0.5
        anti_persistent = np.zeros(n)
        for i in range(1, n):
            anti_persistent[i] = phi_neg * anti_persistent[i - 1] + rng.standard_normal() * 0.87
        h_anti = hurst_exponent(pd.Series(anti_persistent), max_lag=50)

        assert h_persistent > h_anti, (
            f"persistent H={h_persistent:.3f} should exceed anti-persistent H={h_anti:.3f}"
        )

    def test_short_series_raises(self):
        """A series shorter than 2*max_lag should raise ValueError."""
        with pytest.raises(ValueError):
            hurst_exponent(pd.Series([1.0] * 10), max_lag=50)

    def test_hurst_is_bounded(self):
        """H must be in [0, 1]."""
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.standard_normal(2000))
        h = hurst_exponent(returns, max_lag=100)
        assert 0.0 <= h <= 1.0


class TestClassification:
    def test_reversive(self):
        assert classify_regime(0.30) == REGIME_REVERSIVE

    def test_momentum(self):
        assert classify_regime(0.70) == REGIME_MOMENTUM

    def test_random_walk(self):
        assert classify_regime(0.50) == REGIME_RANDOM


class TestExecutionSchedule:
    def test_normalizes_to_one(self):
        """Schedule weights must sum to 1.0."""
        for h in [0.3, 0.5, 0.7]:
            schedule = execution_schedule(h, horizon_steps=20)
            assert abs(schedule.sum() - 1.0) < 1e-10

    def test_momentum_is_front_loaded(self):
        """Schedule should always be front-loaded relative to TWAP (>25% in first quarter)."""
        for h in [0.3, 0.5, 0.7]:
            schedule = execution_schedule(h, horizon_steps=20)
            quarter = 5
            front = schedule[:quarter].sum()
            assert front > 0.25, f"H={h}: first quarter weight {front:.2%} should exceed 25% (TWAP)"

    def test_recommend_pace_is_string(self):
        assert isinstance(recommend_pace(0.5), str)
        assert "RANDOM" in recommend_pace(0.5)
        assert "MOMENTUM" in recommend_pace(0.7)
        assert "REVERSIVE" in recommend_pace(0.3)


class TestRollingHurst:
    def test_produces_multiple_points(self):
        """Rolling Hurst should produce multiple values for a long series."""
        rng = np.random.default_rng(42)
        prices = pd.Series(100 + np.cumsum(rng.standard_normal(1000)),
                           index=pd.date_range("2020-01-01", periods=1000, freq="D"))
        hurst = rolling_hurst(prices, window=200, step=50, max_lag=40)
        assert len(hurst) >= 10

    def test_window_too_short_raises(self):
        with pytest.raises(ValueError):
            rolling_hurst(pd.Series([1.0] * 10), window=200)
