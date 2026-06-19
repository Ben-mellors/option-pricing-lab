import numpy as np
import pytest
from pricinglab.analytic import bs_call
from pricinglab.lattice import crr_european , crr_american
from pricinglab.montecarlo import mc_european
from pricinglab.pde import explicit_fd_european


def test_crr_converges_to_bs():
    """Binomial tree with many steps matches Black-Scholes closely."""
    bs = bs_call(100, 100, 0.05, 0.2, 1.0)
    tree = crr_european(100, 100, 0.05, 0.2, 1.0, 2000, "call")
    assert abs(tree - bs) < 0.01


def test_mc_converges_to_bs():
    """Monte Carlo with many simulations matches Black-Scholes within tolerance."""
    bs = bs_call(100, 100, 0.05, 0.2, 1.0)
    mc = mc_european(100, 100, 0.05, 0.2, 1.0, 500000, "call", seed=42)
    assert abs(mc - bs) < 0.05


def test_pde_matches_black_scholes():
    """Explicit FD price should agree with the analytic BS price.

    Use a stable (M, N) pair well inside the CFL condition - if this
    fails because of instability rather than genuine error, the N here
    is the first thing to check (see scripts/stability_pde.py).
    """
    S, K, r, sigma, T, S_max = 100, 100, 0.05, 0.2, 1, 300
    M, N = 200, 2000

    pde_price = explicit_fd_european(
        S=S, K=K, r=r, sigma=sigma, T=T, M=M, N=N, S_max=S_max,
        option_type="call",
    )
    bs_price = bs_call(S, K, r, sigma, T)

    assert pde_price == pytest.approx(bs_price, abs=0.05)

def test_american_call_equals_european_no_dividends():
    """No dividends => early exercise of a call is never optimal,
    so American and European prices should be identical."""
    params = dict(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500, option_type="call")
    euro = crr_european(**params)
    amer = crr_american(**params)
    assert amer == pytest.approx(euro, abs=1e-8)


def test_american_put_exceeds_european():
    """Early exercise can be optimal for a put, so American >= European,
    strictly so when deep in the money."""
    params = dict(S=80, K=100, r=0.05, sigma=0.2, T=1, N=500, option_type="put")
    euro = crr_european(**params)
    amer = crr_american(**params)
    assert amer >= euro
    assert amer > euro  # strict for this deep ITM case