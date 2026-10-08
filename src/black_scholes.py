"""Black-Scholes pricing, Greeks and implied volatility (vectorized NumPy)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

# Define types for scalar or vectorized inputs
Numeric = float | np.ndarray


def bs_pricing(
    S0: Numeric,
    K: Numeric,
    r: float,
    T: Numeric,
    sigma: Numeric,
    q: float = 0.0,
    option_type: str = "C"
) -> dict[str, Any]:
    """
    Computes Black-Scholes price and all first-order Greeks (Delta, Gamma, Vega, Theta, Rho)
    using vectorized NumPy calculations. This supports both scalar floats and NumPy arrays.

    Args:
        S0 (Numeric): Underlying asset price (spot).
        K (Numeric): Strike price.
        r (float): Risk-free interest rate (annualized).
        T (Numeric): Time to expiration in years.
        sigma (Numeric): Volatility (annualized).
        q (float): Continuous dividend yield.
        option_type (str): "C" for Call, "P" for Put.

    Returns:
        Dict[str, Any]: Prices and greeks. Returns dictionary with keys:
                        'price', 'delta', 'gamma', 'vega', 'theta', 'rho'.
                        For backward compatibility, also populates 'call' or 'put' keys.
    """
    # Convert inputs to numpy arrays for vectorization
    S0 = np.asarray(S0, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Validate inputs
    if np.any(S0 <= 0) or np.any(K <= 0) or np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("Prices, strikes, volatility must be positive, and tenor non-negative.")

    # Handle expired options (T = 0)
    t_zero_mask = T == 0
    # Safe division: use temporary positive T to avoid division by zero in d1/d2 calculation
    T_safe = np.where(t_zero_mask, 1.0, T)
    sigma_safe = np.where(sigma == 0, 1e-8, sigma)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * np.sqrt(T_safe))
    d2 = d1 - sigma_safe * np.sqrt(T_safe)

    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    # Calculate Prices
    call_price = S0 * np.exp(-q * T_safe) * cdf_d1 - K * np.exp(-r * T_safe) * cdf_d2
    put_price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S0 * np.exp(-q * T_safe) * norm.cdf(-d1)

    # Calculate Greeks
    delta_call = np.exp(-q * T_safe) * cdf_d1
    delta_put = -np.exp(-q * T_safe) * norm.cdf(-d1)

    gamma = (pdf_d1 * np.exp(-q * T_safe)) / (S0 * sigma_safe * np.sqrt(T_safe))
    vega = 0.01 * S0 * np.exp(-q * T_safe) * pdf_d1 * np.sqrt(T_safe)

    theta_call = (1 / 365.0) * (
        -((S0 * sigma_safe * np.exp(-q * T_safe)) / (2 * np.sqrt(T_safe))) * pdf_d1
        - r * K * np.exp(-r * T_safe) * cdf_d2
        + q * S0 * np.exp(-q * T_safe) * cdf_d1
    )
    theta_put = (1 / 365.0) * (
        -((S0 * sigma_safe * np.exp(-q * T_safe)) / (2 * np.sqrt(T_safe))) * pdf_d1
        + r * K * np.exp(-r * T_safe) * norm.cdf(-d2)
        - q * S0 * np.exp(-q * T_safe) * norm.cdf(-d1)
    )

    rho_call = 0.01 * K * T_safe * np.exp(-r * T_safe) * cdf_d2
    rho_put = -0.01 * K * T_safe * np.exp(-r * T_safe) * norm.cdf(-d2)

    # Set boundary conditions for expired options
    if np.any(t_zero_mask):
        call_price = np.where(t_zero_mask, np.maximum(0.0, S0 - K), call_price)
        put_price = np.where(t_zero_mask, np.maximum(0.0, K - S0), put_price)
        delta_call = np.where(t_zero_mask, np.where(S0 > K, 1.0, np.where(S0 < K, 0.0, 0.5)), delta_call)
        delta_put = np.where(t_zero_mask, np.where(S0 < K, -1.0, np.where(S0 > K, 0.0, -0.5)), delta_put)
        gamma = np.where(t_zero_mask, 0.0, gamma)
        vega = np.where(t_zero_mask, 0.0, vega)
        theta_call = np.where(t_zero_mask, 0.0, theta_call)
        theta_put = np.where(t_zero_mask, 0.0, theta_put)
        rho_call = np.where(t_zero_mask, 0.0, rho_call)
        rho_put = np.where(t_zero_mask, 0.0, rho_put)

    # Compile result mapping option type
    is_call = option_type.upper() == "C"
    price = np.where(is_call, call_price, put_price)
    delta = np.where(is_call, delta_call, delta_put)
    theta = np.where(is_call, theta_call, theta_put)
    rho = np.where(is_call, rho_call, rho_put)

    # Return structure matching any list/array input or clean scalars
    def to_output(val: np.ndarray) -> Any:
        return val.item() if val.ndim == 0 else val

    ret = {
        "price": to_output(price),
        "delta": to_output(delta),
        "gamma": to_output(gamma),
        "vega": to_output(vega),
        "theta": to_output(theta),
        "rho": to_output(rho),
    }

    # Backward compatible keys for original files
    ret["call"] = to_output(call_price)
    ret["put"] = to_output(put_price)

    return ret


# Backward compatible aliases
def bs_call(S0: Numeric, K: Numeric, r: float, T: Numeric, sigma: Numeric, q: float = 0.0) -> dict[str, Any]:
    return bs_pricing(S0, K, r, T, sigma, q, option_type="C")


def bs_put(S0: Numeric, K: Numeric, r: float, T: Numeric, sigma: Numeric, q: float = 0.0) -> dict[str, Any]:
    return bs_pricing(S0, K, r, T, sigma, q, option_type="P")


def implied_volatility(
    S0: float,
    K: float,
    r: float,
    T: float,
    price: float,
    q: float = 0.0,
    option_type: str = "C",
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Computes option Implied Volatility (IV) using the industry-standard Newton-Raphson
    method with analytical Vega as the derivative. Falls back to Bisection search if
    Newton-Raphson diverges or encounters flat spots (near-zero Vega).

    Args:
        S0 (float): Spot price.
        K (float): Strike price.
        r (float): Risk-free rate.
        T (float): Time to expiration in years.
        price (float): Observed market price of the option.
        q (float): Dividend yield.
        option_type (str): "C" for Call, "P" for Put.
        max_iterations (int): Maximum iterations for solvers.
        tolerance (float): Precision tolerance.

    Returns:
        float: Implied volatility as a decimal (e.g. 0.25 for 25%).
    """
    if price <= 0:
        return 0.0

    # Intrinsic value check
    intrinsic = max(0.0, S0 * np.exp(-q * T) - K * np.exp(-r * T)) if option_type.upper() == "C" else max(0.0, K * np.exp(-r * T) - S0 * np.exp(-q * T))
    if price < intrinsic - tolerance:
        return 0.0

    # 1. Newton-Raphson Method
    sigma = 0.5  # Standard initial guess (50% IV)
    for _ in range(max_iterations):
        res = bs_pricing(S0, K, r, T, sigma, q, option_type)
        diff = res["price"] - price
        if abs(diff) < tolerance:
            return float(sigma)

        # Vega is the derivative of option price with respect to volatility
        # Multiply by 100 because our pricing vega is scaled to 1% change (0.01 * ...)
        vega = res["vega"] * 100.0
        if abs(vega) < 1e-5:
            # Derivative is too small, break to fallback to bisection
            break

        sigma -= diff / vega

        # Keep sigma within realistic bounds during iteration
        if sigma <= 0 or sigma > 5.0:
            break
    else:
        # If converged but out of bounds
        if 0 < sigma < 5.0:
            return float(sigma)

    # 2. Bisection Fallback Method (robust guaranteed convergence)
    low_sigma, high_sigma = 1e-6, 5.0
    for _ in range(max_iterations):
        mid_sigma = 0.5 * (low_sigma + high_sigma)
        res = bs_pricing(S0, K, r, T, mid_sigma, q, option_type)
        diff = res["price"] - price

        if abs(diff) < tolerance:
            return float(mid_sigma)

        if diff > 0:
            high_sigma = mid_sigma
        else:
            low_sigma = mid_sigma

    return float(mid_sigma)


if __name__ == "__main__":
    # Test vectorized capabilities
    spots = np.array([100.0, 110.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0])
    tenor = 30 / 365.0
    vol = 0.25

    results = bs_call(spots, strikes, 0.05, tenor, vol)
    print("Vectorized Spot Prices:", spots)
    print("Vectorized Option Prices:", results["price"])
    print("Vectorized Option Deltas:", results["delta"])

    # Test real implied volatility computation speed and convergence
    target_price = 4.5
    computed_iv = implied_volatility(100.0, 100.0, 0.05, tenor, target_price)
    print(f"\nTarget price: {target_price} -> Computed Implied Volatility: {computed_iv:.4%}")
    repriced = bs_call(100.0, 100.0, 0.05, tenor, computed_iv)["price"]
    print(f"Repriced with computed IV: {repriced:.4f} (Error: {abs(repriced - target_price):.2e})")
