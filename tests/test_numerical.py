import numpy as np
from pricinglab.analytic import bs_call
from pricinglab.lattice import crr_european
from pricinglab.montecarlo import mc_european


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