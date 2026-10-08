import numpy as np
import pytest

from src.black_scholes import bs_call, bs_put, implied_volatility


def test_black_scholes_pricing_scalar():
    """Test standard scalar pricing values against exact mathematical reference."""
    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 0.5
    sigma = 0.2
    q = 0.0

    # Computed standard call & put prices
    call_res = bs_call(S0, K, r, T, sigma, q)
    put_res = bs_put(S0, K, r, T, sigma, q)

    # Reference values (verified using financial calculators)
    # Call value should be ~6.8887, Delta ~0.5977
    assert isinstance(call_res["price"], float)
    assert abs(call_res["price"] - 6.888729) < 1e-5
    assert abs(call_res["delta"] - 0.597734) < 1e-5

    # Put value should be ~4.4197, Delta ~-0.40226
    assert isinstance(put_res["price"], float)
    assert abs(put_res["price"] - 4.419720) < 1e-5
    assert abs(put_res["delta"] - (-0.402265)) < 1e-5


def test_black_scholes_pricing_vectorized():
    """Test vectorized inputs using NumPy arrays."""
    spots = np.array([100.0, 110.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0])
    r = 0.05
    T = 0.5
    vol = 0.2

    call_res = bs_call(spots, strikes, r, T, vol)

    assert isinstance(call_res["price"], np.ndarray)
    assert len(call_res["price"]) == 3
    # Check that in-the-money options are priced higher
    assert call_res["price"][2] > call_res["price"][1] > call_res["price"][0]


def test_black_scholes_boundary_conditions():
    """Test expired options (T = 0) to ensure zero division is handled and returns intrinsic value."""
    S0 = 105.0
    K = 100.0
    r = 0.05
    T = 0.0
    sigma = 0.2

    # Call should equal intrinsic value (105 - 100 = 5.0) and Delta should be 1.0
    call_res = bs_call(S0, K, r, T, sigma)
    assert call_res["price"] == 5.0
    assert call_res["delta"] == 1.0
    assert call_res["vega"] == 0.0

    # Put should equal intrinsic value (max(0, 100 - 105) = 0.0) and Delta should be 0.0
    put_res = bs_put(S0, K, r, T, sigma)
    assert put_res["price"] == 0.0
    assert put_res["delta"] == 0.0


def test_black_scholes_invalid_inputs():
    """Ensure invalid parameters raise an appropriate ValueError exception."""
    with pytest.raises(ValueError):
        bs_call(-100.0, 100.0, 0.05, 0.5, 0.2)  # Negative spot

    with pytest.raises(ValueError):
        bs_call(100.0, 100.0, 0.05, -0.5, 0.2)  # Negative time to maturity


def test_implied_volatility_solver():
    """Test convergence of Newton-Raphson and Bisection solver for implied volatility."""
    S0 = 100.0
    K = 100.0
    r = 0.05
    T = 0.25
    observed_price = 4.25

    # Compute IV
    iv = implied_volatility(S0, K, r, T, observed_price)

    # Reprice to ensure correct round-trip
    repriced_res = bs_call(S0, K, r, T, iv)
    assert abs(repriced_res["price"] - observed_price) < 1e-5


def test_implied_volatility_bisection_fallback():
    """Test option with near-zero Vega to force bisection fallback (deep out-of-the-money)."""
    S0 = 60.0
    K = 100.0
    r = 0.05
    T = 0.1
    # Deep OTM call option has very small price, but if price > 0 there should be a valid (albeit high) IV
    observed_price = 0.5

    iv = implied_volatility(S0, K, r, T, observed_price)
    repriced_res = bs_call(S0, K, r, T, iv)
    assert abs(repriced_res["price"] - observed_price) < 1e-4
