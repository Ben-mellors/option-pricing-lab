"""
Binomial tree (lattice) methods for pricing European and American options.

Uses the Cox-Ross-Rubinstein (CRR) parameterisation:
    u = exp(sigma * sqrt(T / N))
    d = 1 / u

Assumes:
- No dividends
- Constant volatility and risk-free rate
- Frictionless market (no transaction costs)

Convergence to the Black-Scholes analytic price is demonstrated
as the number of steps N increases.
"""

import numpy as np

def crr_european(S: float, K: float, r: float, sigma: float, T: float, N: int, option_type: str) -> float:
    """
    Compute crr european option pricing method

    Parameters
    ----------
    S            : current spot price
    K            : strike price
    r            : continuously-compounded risk-free rate (annual)
    sigma        : annualised volatility
    T            : time to maturity in years
    N            : number of steps
    option_type  : "call" or "put"
    """

    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    j = np.arange(N + 1) #creates an aray from 0 to n+1, so all the possible final stock price options we have
    
    ST = S * (u ** j) * (d ** (N - j)) #st = stock price at time T

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    discount = np.exp(-r * dt)
    for i in range(N):
        payoff = discount * (p * payoff[1:] + (1 - p) * payoff[:-1]) 

    return float(payoff[0])
#payoff[1:] means "give me every element from index 1 onwards". So if payoff has 3 elements at indices 0, 1, 2 - you get indices 1 and 2.
#payoff[:-1] means "give me every element up to but not including the last one". So from indices 0, 1, 2 - you get indices 0 and 1.
#That's called array slicing in Python. It's one of numpy's most useful features - grabbing parts of an array without writing a loop.