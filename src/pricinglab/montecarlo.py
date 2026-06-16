"""
Monte Carlo method for pricing European options.

Assumes:
- European exercise (no early exercise)
- No dividends
- Constant volatility and risk-free rate
- Frictionless market (no transaction costs)

Convergence to the Black-Scholes analytic price improves
as the number of simulations increases.
"""

import numpy as np

def mc_european(S: float, K: float, r: float, sigma: float, T: float, n_sims: int, option_type: str, seed: int = 42) -> float:

    """
    Estimate a European option price by Monte Carlo simulation.

    Parameters
    ----------
    S           : current spot price
    K           : strike price
    r           : continuously-compounded risk-free rate (annual)
    sigma       : annualised volatility
    T           : time to maturity in years
    n_sims      : number of simulated price paths
    option_type : "call" or "put"
    seed        : random seed for reproducibility

    Returns
    -------
    float : estimated option price
    """

    rng = np.random.default_rng(seed) #random number generator
    Z = rng.standard_normal(n_sims) #random shocks
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) #future predicted stock prices in an array, one value for each simulation
 

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    price = np.exp(-r * T) * np.mean(payoffs) #avg of all the payoffs
    return float(price)