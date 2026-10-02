import math
import time

import numpy as np

from src.black_scholes import bs_pricing


# Define the previous pure-python hand-rolled Black-Scholes implementation for comparison
def fi_legacy(x):
    Pi = 3.141592653589793238
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    L = abs(x)
    k = 1 / (1 + 0.2316419 * L)
    p = 1 - 1 / pow(2 * Pi, 0.5) * math.exp(-pow(L, 2) / 2) * (
        a1 * k + a2 * pow(k, 2) + a3 * pow(k, 3) + a4 * pow(k, 4) + a5 * pow(k, 5)
    )
    if x >= 0:
        return p
    else:
        return 1 - p


def normalInv_legacy(x):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-x * x * 0.5)


def bsCall_legacy(S0, K, r, T, sigma, q=0):
    ret = {}
    if S0 > 0 and K > 0 and r >= 0 and T > 0 and sigma > 0:
        d1 = (math.log(S0 / K) + (r - q + sigma * sigma * 0.5) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        ret["call"] = math.exp(-q * T) * S0 * fi_legacy(d1) - K * math.exp(-r * T) * fi_legacy(d2)
        ret["delta"] = math.exp(-q * T) * fi_legacy(d1)
        ret["gamma"] = (normalInv_legacy(d1) * math.exp(-q * T)) / (S0 * sigma * math.sqrt(T))
        ret["vega"] = 0.01 * S0 * math.exp(-q * T) * normalInv_legacy(d1) * math.sqrt(T)
        ret["theta"] = (1 / 365) * (
            -((S0 * sigma * math.exp(-q * T)) / (2 * math.sqrt(T))) * normalInv_legacy(d1)
            - r * K * (math.exp(-r * T)) * fi_legacy(d2)
            + q * S0 * (math.exp(-q * T)) * fi_legacy(d1)
        )
        ret["rho"] = 0.01 * K * T * math.exp(-r * T) * fi_legacy(d2)
    else:
        ret["errores"] = "Se Ingresaron valores incorrectos"
    return ret


def run_benchmark():
    # Number of options to price in a single run
    N = 100000
    print(f"=== QUANT DEV BENCHMARK: Pricing {N:,} Options ===")

    # Generate random options parameters
    np.random.seed(42)
    spots = np.random.uniform(80.0, 120.0, N)
    strikes = np.random.uniform(90.0, 110.0, N)
    tenors = np.random.uniform(0.1, 2.0, N)
    volatilities = np.random.uniform(0.1, 0.5, N)
    r = 0.05

    # 1. Benchmark Vectorized Implementation
    t0 = time.perf_counter()
    res_vectorized = bs_pricing(spots, strikes, r, tenors, volatilities, option_type="C")
    t_vectorized = time.perf_counter() - t0
    print(f"Vectorized NumPy (SciPy) Time: {t_vectorized:.5f} seconds")
    print(f"  Speed: {N / t_vectorized:,.0f} options / second")
    print(f"  Sample price check: {res_vectorized['price'][0]:.4f}")

    # 2. Benchmark Legacy Pure Python (using loop)
    t0 = time.perf_counter()
    res_legacy = []
    for i in range(N):
        res_legacy.append(
            bsCall_legacy(spots[i], strikes[i], r, tenors[i], volatilities[i])
        )
    t_legacy = time.perf_counter() - t0
    print(f"Legacy Pure Python Loop Time: {t_legacy:.5f} seconds")
    print(f"  Speed: {N / t_legacy:,.0f} options / second")

    # 3. Calculate Speedup
    speedup = t_legacy / t_vectorized
    print(f"Vectorized code is {speedup:.1f}x FASTER than the legacy implementation!")


if __name__ == "__main__":
    run_benchmark()
