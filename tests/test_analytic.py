import pytest
import numpy as np
from pricinglab.analytic import bs_call, bs_put, bs_d1, bs_d2


def test_bs_call_known_value():
    """Call price matches known Black-Scholes result to 4 decimal places."""
    price = bs_call(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    assert abs(price - 10.4506) < 0.0001


def test_bs_put_known_value():
    """Put price matches known Black-Scholes result to 4 decimal places."""
    price = bs_put(S=100, K=100, r=0.05, sigma=0.2, T=1.0)
    assert abs(price - 5.5735) < 0.0001


def test_put_call_parity():
    """Put-call parity holds: C - P = S - K * exp(-rT)."""
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    call = bs_call(S, K, r, sigma, T)
    put = bs_put(S, K, r, sigma, T)
    lhs = call - put
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 0.0001