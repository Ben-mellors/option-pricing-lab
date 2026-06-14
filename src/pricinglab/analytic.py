"""
Analytic Black-Scholes pricing formulae for European call and put options.

Assumes:
- European exercise (no early exercise)
- No dividends
- Constant volatility and risk-free rate
- Frictionless market (no transaction costs)

This module provides the closed-form ground truth that all numerical
methods in this project are benchmarked against.
"""
import numpy as np
from scipy.stats import norm

def bs_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute d1 from the Black-Scholes formula.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    r     : continuously-compounded risk-free rate (annual)
    sigma : annualised volatility
    T     : time to maturity in years
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute d2 from the Black-Scholes formula.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    r     : continuously-compounded risk-free rate (annual)
    sigma : annualised volatility
    T     : time to maturity in years
    """
    return bs_d1(S, K, r, sigma, T) - sigma * np.sqrt(T)

def bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute the Black-Scholes price of a European call option.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    r     : continuously-compounded risk-free rate (annual)
    sigma : annualised volatility
    T     : time to maturity in years

    Returns
    -------
    float : call option price
    """
    return S * norm.cdf(bs_d1(S, K, r, sigma, T)) - K * np.exp(-r * T) * norm.cdf(bs_d2(S, K, r, sigma, T))#

def bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute the Black-Scholes price of a European put option.

    Parameters
    ----------
    S     : current spot price
    K     : strike price
    r     : continuously-compounded risk-free rate (annual)
    sigma : annualised volatility
    T     : time to maturity in years

    Returns
    -------
    float : put option price
    """
    return  K * np.exp(-r * T) * norm.cdf(-bs_d2(S, K, r, sigma, T)) - S * norm.cdf(-bs_d1(S, K, r, sigma, T))