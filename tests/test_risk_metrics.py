"""Unit tests for the risk metrics module (pure functions, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    annualized_volatility,
    beta,
    conditional_value_at_risk,
    drawdown_series,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    value_at_risk,
)


def _series(values):
    return pd.Series(values, dtype=float)


def test_total_return():
    assert total_return(_series([100.0, 110.0, 120.0])) == pytest.approx(0.20)


def test_annualized_volatility():
    returns = _series([0.01, -0.01, 0.01, -0.01])
    assert annualized_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_sharpe_ratio_zero_volatility():
    """Constant zero returns have exactly zero volatility -> NaN, not a crash."""
    sharpe = sharpe_ratio(_series([0.0] * 10))
    assert np.isnan(sharpe)


def test_sortino_ratio_positive():
    """Sortino must exceed Sharpe when downside deviation is smaller than total vol."""
    rng = np.random.default_rng(42)
    returns = _series(rng.normal(0.0005, 0.01, 500))
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_var_cvar_known_series():
    """Historical VaR/CVaR against hand-computed reference values."""
    returns = _series([-0.01, -0.02, 0.01, 0.005, -0.03])
    assert value_at_risk(returns, level=0.95) == pytest.approx(0.028)
    assert conditional_value_at_risk(returns, level=0.95) == pytest.approx(0.03)
    assert conditional_value_at_risk(returns, level=0.95) >= value_at_risk(returns, level=0.95)


def test_drawdown_series():
    drawdown = drawdown_series(_series([0.10, -0.10, 0.10]))
    assert drawdown.iloc[0] == pytest.approx(0.0)
    assert drawdown.iloc[1] == pytest.approx(-0.10)
    assert drawdown.min() == pytest.approx(-0.10)


def test_beta_two_x_relationship():
    """For y = 2x: cov(x,y)/var(y) = 0.5 and cov(y,x)/var(x) = 2.0."""
    rng = np.random.default_rng(42)
    x = _series(rng.normal(0.0, 0.01, 500))
    y = 2.0 * x
    assert beta(x, y) == pytest.approx(0.5, rel=0.01)
    assert beta(y, x) == pytest.approx(2.0, rel=0.01)


def test_beta_no_overlap():
    """Non-overlapping dates must return None, not a crash."""
    x = _series([0.01, -0.01])
    y = pd.Series([0.01, -0.01], index=["2024-01-01", "2024-01-02"])
    assert beta(x, y) is None
